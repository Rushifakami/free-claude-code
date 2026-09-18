"""Declarative profiles for Poe Chat provider."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.history_replay import HistoryScope
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import (
    InputModalityBooleanPaths,
    ModelTokenLimitResolver,
    RequiredPathValues,
    live_provider_context_window_consensus,
)

from .request_policy import PoeChatRequestPolicy, PoeChatPostprocessor
from .reasoning import NO_REASONING, ReasoningEncoder


@dataclass(frozen=True, slots=True)
class PoeModelPagination:
    """Bounded numbered-pagination metadata for a model-list endpoint."""

    page_param: str = "page"
    first_page: int = 1
    current_page_path: tuple[str, ...] = ("pagination", "current_page")
    total_pages_path: tuple[str, ...] = ("pagination", "total_pages")
    max_pages: int = 100


@dataclass(frozen=True, slots=True)
class PoeModelListing:
    """Declarative model-list endpoint and response shape."""

    path: str | None = None
    query_params: tuple[tuple[str, str], ...] = ()
    collection_field: str | None = "data"
    id_field: str = "id"
    aliases_field: str | None = None
    additional_model_ids: tuple[str, ...] = ()
    required_path_values: RequiredPathValues = ()
    required_null_field: str | None = None
    required_sequence_items: tuple[tuple[str, str], ...] = ()
    exclude_missing_sequence_fields: bool = False
    tags_field: str | None = None
    thinking_tag: str = "reasoning"
    non_thinking_tag: str | None = None
    thinking_boolean_path: tuple[str, ...] | None = None
    input_modalities_path: tuple[str, ...] | None = None
    thinking_sequence_path: tuple[str, ...] | None = None
    fixed_input_modalities: frozenset[ModelInputModality] | None = None
    input_modality_boolean_paths: InputModalityBooleanPaths = ()
    context_window_tokens_path: tuple[str, ...] | None = None
    max_output_tokens_path: tuple[str, ...] | None = None
    context_window_tokens_resolver: ModelTokenLimitResolver | None = None
    pagination: PoeModelPagination | None = None


@dataclass(frozen=True, slots=True)
class PoeChatProfile:
    """Immutable transport and reasoning behavior for one provider."""

    request_policy: PoeChatRequestPolicy
    reasoning: ReasoningEncoder
    postprocessors: tuple[PoeChatPostprocessor, ...] = ()
    model_ids_are_routable: bool = True
    model_listing: PoeModelListing = PoeModelListing()
    normalize_base_url: bool = False
    reasoning_delta_field: Literal["reasoning_content", "reasoning"] = (
        "reasoning_content"
    )
    reasoning_delta_fallback_field: Literal["reasoning_content", "reasoning"] | None = (
        None
    )
    structured_reasoning_details: bool = False
    history_scope: HistoryScope = HistoryScope.TOOL_CONTINUATION
    user_agent: str | None = None

    @property
    def provider_name(self) -> str:
        return self.request_policy.provider_name

    def base_url(self, configured: str) -> str:
        return configured

    def reasoning_delta(self, delta: Any) -> str | None:
        value = getattr(delta, self.reasoning_delta_field, None)
        if isinstance(value, str) and value:
            return value
        fallback = self.reasoning_delta_fallback_field
        if fallback is None:
            return value if isinstance(value, str) else None
        fallback_value = getattr(delta, fallback, None)
        if isinstance(fallback_value, str):
            return fallback_value
        return value if isinstance(value, str) else None

    def apply_reasoning(
        self,
        body: dict[str, Any],
        _request: MessagesRequest,
        policy: ReasoningPolicy,
    ) -> None:
        self.apply_reasoning_to_body(body, policy)

    def apply_reasoning_to_body(
        self,
        body: dict[str, Any],
        policy: ReasoningPolicy,
    ) -> None:
        """Encode resolved reasoning policy after either client translation."""
        self.reasoning.encode(body, policy)

    @property
    def request_postprocessors(self) -> tuple[PoeChatPostprocessor, ...]:
        return (*self.postprocessors, self.apply_reasoning)


def _poe_policy(provider_name: str) -> PoeChatRequestPolicy:
    return PoeChatRequestPolicy(
        provider_name=provider_name,
        reasoning_replay=ReasoningReplayMode.DISABLED,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    )


POE_CHAT_PROFILE = PoeChatProfile(
    _poe_policy("Poe"),
    NO_REASONING,
)
