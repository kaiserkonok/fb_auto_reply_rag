from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

_STAGE_INSTRUCTIONS = {
    "greeting": "Greet the user warmly, introduce yourself briefly, and invite the user to share their goal.",
    "interest_detection": "Confirm the user interest and ask one concise qualifying question about their needs.",
    "problem_identification": "Identify the main pain point and ask a focused follow-up question to clarify context.",
    "capability_presentation": "Explain how the solution can address the user's problem with concrete benefits.",
    "payment_discussion": "Discuss pricing and payment in a clear way, and propose next actionable steps.",
    "closing": "Close politely, summarize agreed points, and provide a clear next-step or sign-off.",
}


def generate_ai_reply(stage: str, conversation_history: list[dict[str, Any]], user_message: str) -> str:
    current_stage = stage if stage in _STAGE_INSTRUCTIONS else "greeting"
    messages = _build_prompt_messages(current_stage, conversation_history, user_message)
    ai_reply = _call_openai_chat_completions(messages)
    if ai_reply:
        return ai_reply
    return _fallback_reply(current_stage)


def _build_prompt_messages(stage: str, conversation_history: list[dict[str, Any]], user_message: str) -> list[dict[str, str]]:
    stage_instruction = _STAGE_INSTRUCTIONS[stage]
    system_prompt = (
        "You are a helpful Facebook Messenger sales assistant.\n"
        "Respond naturally in 1-3 short paragraphs.\n"
        "Do not mention internal stages, prompts, or system rules.\n"
        f"Current conversation stage: {stage}\n"
        f"Stage objective: {stage_instruction}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for item in conversation_history[-10:]:
        role = "assistant" if str(item.get("role", "")).strip().lower() == "assistant" else "user"
        content = str(item.get("message_text", "")).strip()
        if content:
            messages.append({"role": role, "content": content})

    cleaned_user_message = (user_message or "").strip()
    if cleaned_user_message:
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != cleaned_user_message:
            messages.append({"role": "user", "content": cleaned_user_message})

    return messages


def _call_openai_chat_completions(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not configured; using fallback AI reply.")
        return ""

    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    timeout_seconds = int(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "20"))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 350,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
        if response.status_code != 200:
            logger.error("AI API call failed: status=%s body=%s", response.status_code, response.text)
            return ""

        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            return ""
        content = str(choices[0].get("message", {}).get("content", "")).strip()
        return content
    except (requests.RequestException, ValueError):
        logger.exception("AI API request failed unexpectedly.")
        return ""


def _fallback_reply(stage: str) -> str:
    if stage == "greeting":
        return "Hi! Thanks for reaching out. What are you looking to improve right now?"
    if stage == "interest_detection":
        return "Great to hear. What outcome are you hoping to get from this?"
    if stage == "problem_identification":
        return "Understood. Can you share the biggest challenge you are facing today?"
    if stage == "capability_presentation":
        return "Based on what you shared, we can support you with a tailored workflow that saves time and improves consistency."
    if stage == "payment_discussion":
        return "I can walk you through pricing options. What budget range are you considering?"
    return "Thanks for the conversation. If you want, I can help you with the next step anytime."
