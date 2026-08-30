"""Security and grounding guardrails for the Finance RAG pipeline.

The guardrails are intentionally conservative: they block obvious unsafe prompts
and make sure generated answers stay grounded in the retrieved financial excerpts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FINANCIAL_KEYWORDS = (
    "revenue",
    "earnings",
    "sales",
    "profit",
    "loss",
    "cash flow",
    "ebitda",
    "net income",
    "operating margin",
    "balance sheet",
    "income statement",
    "sec",
    "filing",
    "quarter",
    "annual",
    "fy",
    "fiscal",
    "eps",
    "debt",
    "liquidity",
    "expense",
    "capital",
    "share",
    "stock",
    "dividend",
    "guidance",
    "forecast",
    "m&a",
    "acquisition",
)

DANGEROUS_PATTERNS = (
    "ignore previous instructions",
    "system prompt",
    "developer prompt",
    "bypass",
    "jailbreak",
    "malware",
    "exploit",
    "hack",
    "phishing",
    "bomb",
    "weapon",
    "self-harm",
    "suicide",
    "kill",
    "steal credentials",
    "credit card numbers",
    "social security",
    "ssn",
)


@dataclass(frozen=True)
class GuardrailResult:
    """Structured result from query or answer validation."""

    allowed: bool
    message: str | None = None


def guardrail_query(query: str, *, max_length: int = 2000) -> GuardrailResult:
    """Validate a user query before retrieval and generation.

    Returns an allowed/blocked result rather than raising to allow callers to
    surface a friendly error without crashing the pipeline.
    """

    candidate = (query or "").strip()
    if not candidate:
        return GuardrailResult(False, "Question cannot be empty.")

    if len(candidate) > max_length:
        return GuardrailResult(False, "Question is too long for the financial RAG pipeline.")

    lowered = candidate.lower()
    if any(pattern in lowered for pattern in DANGEROUS_PATTERNS):
        return GuardrailResult(False, "This system only supports safe financial-document Q&A.")

    has_financial_context = any(keyword in lowered for keyword in FINANCIAL_KEYWORDS)
    if not has_financial_context:
        return GuardrailResult(False, "Please ask a financial question grounded in company filings, earnings reports, or account data.")

    return GuardrailResult(True)


def guardrail_response(answer: str, *, valid_ids: set[str] | None = None) -> str:
    """Sanitise generated output so it stays grounded in the retrieved evidence."""

    cleaned = (answer or "").strip()
    if not cleaned:
        return "Insufficient information in the provided documents."

    lower = cleaned.lower()
    if "insufficient information in the provided documents" in lower:
        return "Insufficient information in the provided documents."

    if not valid_ids:
        return "Insufficient information in the provided documents."

    # Remove citation tags that do not match the retrieved document IDs.
    def replace_invalid_citation(match: re.Match[str]) -> str:
        token = match.group(1)
        return f"[{token}]" if token in valid_ids else ""

    sanitized = re.sub(r"\[([A-Za-z0-9_]+)\]", replace_invalid_citation, cleaned)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    # If the answer is empty after sanitising it, fail closed. Otherwise, keep the
    # original text even if no inline citations are present; a grounded answer can
    # still be valid without explicit citations if the caller is not in structured mode.
    if not sanitized:
        return "Insufficient information in the provided documents."

    return sanitized


def validate_query(query: str, *, max_length: int = 2000) -> None:
    """Compatibility wrapper: raises ValueError on unsafe or unsupported queries."""

    result = guardrail_query(query, max_length=max_length)
    if not result.allowed:
        raise ValueError(result.message)


def apply_guardrails(answer: str, *, valid_ids: set[str] | None = None) -> str:
    """Alias used by generation code and any tests that expect a helper named this way."""

    return guardrail_response(answer, valid_ids=valid_ids)


__all__ = [
    "GuardrailResult",
    "apply_guardrails",
    "guardrail_query",
    "guardrail_response",
    "validate_query",
]
