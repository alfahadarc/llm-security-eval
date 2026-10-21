from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

VALID_ROLES = ("system", "user", "assistant")


@dataclass
class GenerationResult:
    text: str
    runtime_s: float
    status: str  # "ok" or "error"
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def validate_messages(messages):
    """Raise ValueError unless messages is a non-empty list of {role, content} dicts."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise ValueError(f"message {i} must have exactly the keys 'role' and 'content'")
        if m["role"] not in VALID_ROLES:
            raise ValueError(f"message {i} has invalid role {m['role']!r}")
        if not isinstance(m["content"], str):
            raise ValueError(f"message {i} content must be a string")
    if messages[-1]["role"] == "assistant":
        raise ValueError("last message must be system or user, not assistant")


class ChatModel(ABC):
    @abstractmethod
    def generate(self, messages) -> GenerationResult:
        """Return generated text only (never the prompt), plus runtime and status."""
