"""Chat-model resolution: bring your own, or name one.

wardhook-core is deliberately provider-agnostic. The base install depends on
``langchain-core`` for the :class:`~langchain_core.language_models.BaseChatModel`
interface and on no provider SDK at all, so nothing here locks you to a vendor.

There are three ways to give an agent a model, in decreasing order of control:

1. **Pass an instance.** Any object with ``.invoke()`` is accepted and returned
   untouched -- a real provider client, a LangChain wrapper, or a test double.
   This path requires no provider extra and is what the test suite uses.
2. **Pass a name.** ``"claude-opus-5"`` or ``"openai:gpt-4o"`` resolves to the
   matching provider package, which must be installed via the corresponding
   extra.
3. **Pass nothing.** Resolves to :data:`DEFAULT_MODEL`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for type checkers
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "DEFAULT_MODEL",
    "ModelResolutionError",
    "describe_model",
    "resolve_model",
]

DEFAULT_MODEL = "claude-opus-5"
"""Model used when none is supplied. Requires the ``[anthropic]`` extra."""

_PROVIDER_IMPORTS: dict[str, tuple[str, str, str]] = {
    # provider -> (module, class, pip extra that installs it)
    "anthropic": ("langchain_anthropic", "ChatAnthropic", "anthropic"),
    "openai": ("langchain_openai", "ChatOpenAI", "openai"),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI", "google"),
}

_NAME_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini-", "google_genai"),
)


class ModelResolutionError(RuntimeError):
    """Raised when a model name cannot be turned into a usable client.

    The message always states which package to install, because a missing
    optional extra is by far the most common cause.
    """


def _infer_provider(name: str) -> str:
    """Guess the provider from a bare model name.

    Args:
        name: A model name with no explicit ``provider:`` prefix.

    Returns:
        The provider key.

    Raises:
        ModelResolutionError: If the name matches no known provider prefix.
    """
    lowered = name.lower()
    for prefix, provider in _NAME_PREFIXES:
        if lowered.startswith(prefix):
            return provider
    known = ", ".join(sorted(_PROVIDER_IMPORTS))
    raise ModelResolutionError(
        f"Cannot infer a provider for model {name!r}. "
        f"Prefix it explicitly, for example 'anthropic:{name}' or 'openai:{name}'. "
        f"Known providers: {known}."
    )


def resolve_model(model: Any = None, **model_kwargs: Any) -> BaseChatModel:
    """Turn ``model`` into something an agent can call.

    Args:
        model: One of:

            * ``None`` -- use :data:`DEFAULT_MODEL`.
            * A model instance (anything exposing ``.invoke``) -- returned
              unchanged, with ``model_kwargs`` ignored.
            * A string -- either ``"provider:name"`` or a bare name whose
              provider is inferred from its prefix.
        **model_kwargs: Extra keyword arguments forwarded to the provider
            client constructor, such as ``max_tokens`` or ``timeout``. Ignored
            when an instance is passed.

    Returns:
        A chat model ready to invoke.

    Raises:
        ModelResolutionError: If the provider is unknown, its package is not
            installed, or the client fails to construct.

    Example:
        >>> from langchain_core.language_models.fake_chat_models import (
        ...     GenericFakeChatModel,
        ... )
        >>> fake = GenericFakeChatModel(messages=iter([]))
        >>> resolve_model(fake) is fake
        True
    """
    if model is None:
        model = DEFAULT_MODEL

    # Duck-typed on purpose: test doubles and community wrappers that do not
    # subclass BaseChatModel still work, which keeps the runtime testable
    # without a network call.
    if not isinstance(model, str):
        if hasattr(model, "invoke"):
            return model  # type: ignore[no-any-return]
        raise ModelResolutionError(
            f"Expected a model name or an object with an .invoke() method, "
            f"got {type(model).__name__}."
        )

    provider, _, name = model.partition(":")
    if not name:
        name = provider
        provider = _infer_provider(name)
    provider = provider.lower().replace("-", "_")

    if provider not in _PROVIDER_IMPORTS:
        known = ", ".join(sorted(_PROVIDER_IMPORTS))
        raise ModelResolutionError(
            f"Unknown provider {provider!r}. Known providers: {known}. "
            f"To use a provider Wardhook does not know about, construct its "
            f"client yourself and pass the instance instead of a name."
        )

    module_name, class_name, extra = _PROVIDER_IMPORTS[provider]
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:
        raise ModelResolutionError(
            f"Model {name!r} needs the {module_name!r} package, which is not "
            f"installed. Install it with:\n\n"
            f'    pip install "wardhook-core[{extra}]"\n\n'
            f"Alternatively, construct any chat model yourself and pass the "
            f"instance: AgentGraph(model=my_model)."
        ) from exc

    client_cls = getattr(module, class_name)
    try:
        return client_cls(model=name, **model_kwargs)  # type: ignore[no-any-return]
    except Exception as exc:
        raise ModelResolutionError(
            f"Failed to construct {class_name}(model={name!r}): {exc}. "
            f"Check that the provider's API key is set in the environment."
        ) from exc


def describe_model(model: Any) -> str:
    """Return a short, log-safe identifier for a model.

    Used for trace and audit records, where the model name matters but the
    client's full repr (which can include configuration) does not.

    Args:
        model: A resolved model instance.

    Returns:
        The model's name if it exposes one, otherwise its class name.

    Example:
        >>> class Stub:
        ...     model_name = "claude-opus-5"
        >>> describe_model(Stub())
        'claude-opus-5'
    """
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__
