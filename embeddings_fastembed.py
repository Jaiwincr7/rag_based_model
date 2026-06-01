"""Call Hugging Face Inference Providers router with Bearer token auth."""

from __future__ import annotations

from typing import List, Optional

import requests
from langchain_core.language_models.llms import LLM


class HFRouterLLM(LLM):
    model_id: str
    api_key: str
    max_tokens: int = 80
    timeout: int = 90
    router_base: str = "https://router.huggingface.co/v1"

    @property
    def _llm_type(self) -> str:
        return "hf_router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        url = f"{self.router_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
        }

        resp = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )

        if resp.status_code == 401:
            raise RuntimeError(
                "HF router 401: invalid or missing token. On Render, set HF_TOKEN to a "
                "fine-grained token with 'Make calls to Inference Providers' (no quotes). "
                "Or set USE_EXTRACTIVE_ONLY=true to skip the LLM."
            )

        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
