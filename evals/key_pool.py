"""Groq key rotation with separate daily exhaustion and temporary cooldowns."""

import asyncio
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_groq import ChatGroq
from pydantic import ConfigDict

from eval_utils import atomic_json_write


class AllKeysExhausted(Exception):
    """Every configured key has a recognized daily quota exhaustion or is invalid."""


def wait_seconds(message):
    match = re.search(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+s)?", message, re.I)
    if not match or not any(match.groups()):
        return 30.0
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float((seconds or "0s")[:-1])


def _daily_limit(message):
    text = message.lower()
    return bool(
        re.search(r"\b(per|each)\s+day\b|\bdaily\b|\b(tpd|rpd)\b|requests?\s+per\s+day|tokens?\s+per\s+day", text)
    )


class GroqKeyPool:
    """Synchronous and asynchronous invocation over configured Groq API keys.

    Only a recognized daily quota marks a key exhausted. Other HTTP 429s put
    that key in cooldown until the provider's retry time; long cooldowns are
    never converted into daily exhaustion.
    """

    def __init__(self, max_wait=90.0, state_path=None):
        raw_keys = []
        raw_keys.extend(os.getenv("GROQ_API_KEYS", "").split(","))
        single = os.getenv("GROQ_API_KEY") or ""
        raw_keys.append(single)
        keys = []
        for item in raw_keys:
            if not item:
                continue
            k = item.strip()
            if len(k) >= 2 and ((k[0] == k[-1]) and k[0] in ('"', "'")):
                k = k[1:-1].strip()
            if k and k not in keys:
                keys.append(k)
        if not keys:
            raise RuntimeError("No Groq keys found. Set GROQ_API_KEYS or GROQ_API_KEY in server/.env")
        self.keys = keys
        self.max_wait = max_wait  # retained for callers; never determines daily exhaustion
        self.state_path = Path(state_path) if state_path else Path(__file__).with_name(".groq_key_pool_state.json")
        self.daily_until = {}
        self.cooldown_until = {}
        self.invalid_keys = set()
        self.cur = 0
        self._clients = {}
        self._state_lock = threading.RLock()
        self._load_state()
        print(f"Groq key pool: {len(self.keys)} key(s)")

    def _fingerprint(self, index):
        return hashlib.sha256(self.keys[index].encode("utf-8")).hexdigest()

    def _load_state(self):
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            now = time.time()
            today = datetime.now().astimezone().date().isoformat()
            if saved.get("date") == today:
                until = float(saved.get("expires_at", 0))
                if until > now:
                    fingerprints = set(saved.get("exhausted", []))
                    for index in range(len(self.keys)):
                        if self._fingerprint(index) in fingerprints:
                            self.daily_until[index] = until
                invalid_fps = set(saved.get("invalid", []))
                for index in range(len(self.keys)):
                    if self._fingerprint(index) in invalid_fps:
                        self.invalid_keys.add(index)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError):
            # Invalid state must not silently disable keys or expose credentials.
            print("[Groq key state unreadable; starting with configured keys]", flush=True)

    def _save_state(self):
        with self._state_lock:
            now = time.time()
            exhausted = [self._fingerprint(index) for index, until in self.daily_until.items() if until > now]
            invalid = [self._fingerprint(index) for index in self.invalid_keys]
            state = {
                "date": datetime.now().astimezone().date().isoformat(),
                "expires_at": max(self.daily_until.values(), default=0),
                "exhausted": exhausted,
                "invalid": invalid,
            }
            atomic_json_write(self.state_path, state)

    def _client(self, index, model, temperature, max_tokens):
        cache_key = (index, model, temperature, max_tokens)
        if cache_key not in self._clients:
            kwargs = {
                "model": model,
                "groq_api_key": self.keys[index],
                "temperature": temperature,
                "max_retries": 0,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            self._clients[cache_key] = ChatGroq(**kwargs)
        return self._clients[cache_key]

    def _available(self):
        with self._state_lock:
            now = time.time()
            for offset in range(len(self.keys)):
                index = (self.cur + offset) % len(self.keys)
                if index in self.invalid_keys:
                    continue
                if self.daily_until.get(index, 0) <= now and self.cooldown_until.get(index, 0) <= now:
                    self.cur = (index + 1) % len(self.keys)
                    return index
            if all((index in self.invalid_keys) or (self.daily_until.get(index, 0) > now) for index in range(len(self.keys))):
                raise AllKeysExhausted(f"All {len(self.keys)} Groq key(s) are exhausted or invalid")
            return None

    def _cooldown_wait(self):
        now = time.time()
        deadlines = [
            self.daily_until.get(i, 0) if self.daily_until.get(i, 0) > now else self.cooldown_until.get(i, 0)
            for i in range(len(self.keys))
        ]
        return max(0.1, min(deadline for deadline in deadlines if deadline > now) - now)

    def _mark_daily(self, index):
        with self._state_lock:
            local_now = datetime.now().astimezone()
            tomorrow = (local_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
            self.daily_until[index] = tomorrow.timestamp()
            self.cooldown_until.pop(index, None)
            self._save_state()
        print(f"[Groq key {index + 1}/{len(self.keys)} daily quota exhausted; rotating]", flush=True)

    def _mark_invalid(self, index):
        with self._state_lock:
            self.invalid_keys.add(index)
            self.cooldown_until.pop(index, None)
            self.daily_until.pop(index, None)
            self._save_state()
        print(f"[Groq key {index + 1}/{len(self.keys)} invalid; rotating]", flush=True)

    def _mark_temporary(self, index, message, retry_after=None):
        delay = wait_seconds(message) if retry_after is None else max(0.1, float(retry_after))
        with self._state_lock:
            self.cooldown_until[index] = time.time() + delay
        print(f"[Groq key {index + 1}/{len(self.keys)} temporarily rate limited; cooldown {delay:.0f}s]", flush=True)

    @staticmethod
    def _status(error):
        response = getattr(error, "response", None)
        return getattr(error, "status_code", None) or getattr(response, "status_code", None)

    @staticmethod
    def _retry_after(error):
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _is_invalid_auth(error, message, status):
        if status == 401:
            return True
        text = (message or "").lower()
        return any(
            token in text
            for token in ("invalid api key", "invalid_api_key", "unauthorized", "authentication", "api key is invalid")
        )

    def _handle_error(self, index, error):
        message = str(error)
        status = self._status(error)
        if status == 413 or "request too large" in message.lower():
            raise error
        if self._is_invalid_auth(error, message, status):
            self._mark_invalid(index)
            return True
        if status == 429 or "rate_limit" in message.lower() or "rate limit" in message.lower():
            if _daily_limit(message):
                self._mark_daily(index)
            else:
                self._mark_temporary(index, message, self._retry_after(error))
            return True
        raise error

    def invoke(self, messages, model, temperature=0.0, max_tokens=None, stop=None):
        while True:
            index = self._available()
            if index is None:
                time.sleep(self._cooldown_wait())
                continue
            try:
                return self._client(index, model, temperature, max_tokens).invoke(messages, stop=stop)
            except Exception as error:
                self._handle_error(index, error)

    async def ainvoke(self, messages, model, temperature=0.0, max_tokens=None, stop=None):
        while True:
            index = self._available()
            if index is None:
                await asyncio.sleep(self._cooldown_wait())
                continue
            try:
                return await self._client(index, model, temperature, max_tokens).ainvoke(messages, stop=stop)
            except Exception as error:
                self._handle_error(index, error)

    def current_key(self):
        """Compatibility accessor; internal Ragas integration uses invoke/ainvoke."""
        index = self._available()
        if index is None:
            raise RuntimeError("All Groq keys are temporarily cooling down")
        return self.keys[index]


class GroqPoolChatModel(BaseChatModel):
    """LangChain chat-model adapter that routes Ragas sync/async calls via pool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    pool: GroqKeyPool
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None

    @property
    def _llm_type(self):
        return "groq-key-pool"

    @property
    def _identifying_params(self):
        return {"model": self.model, "temperature": self.temperature}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = self.pool.invoke(
            messages,
            model=self.model,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stop=stop,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result.content))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await self.pool.ainvoke(
            messages,
            model=self.model,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stop=stop,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result.content))])
