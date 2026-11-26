"""Counts LLM calls and tokens for one run, and stops a run that goes over its call budget."""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# USD per million tokens (input, output). Rough list prices, only used for the cost estimate.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "gemini-3.8-flash": (0.30, 2.50),
    "gpt-4o-mini": (0.15, 0.60),
}
DEFAULT_PRICE = (0.50, 1.50)


class BudgetExceeded(RuntimeError):
    pass


class UsageTracker(BaseCallbackHandler):
    raise_error = True  # let BudgetExceeded stop the run instead of being swallowed

    def __init__(self, max_calls: int = 0) -> None:
        self.max_calls = max_calls
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self._lock = threading.Lock()

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            if self.max_calls and self.calls >= self.max_calls:
                raise BudgetExceeded(f"This run reached its limit of {self.max_calls} LLM calls")
            self.calls += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                model = (getattr(message, "response_metadata", None) or {}).get("model_name", "")
                tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                price_in, price_out = PRICES.get(model, DEFAULT_PRICE)
                with self._lock:
                    self.input_tokens += tokens_in
                    self.output_tokens += tokens_out
                    self.cost += (tokens_in * price_in + tokens_out * price_out) / 1_000_000

    def summary(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost, 5),
        }
