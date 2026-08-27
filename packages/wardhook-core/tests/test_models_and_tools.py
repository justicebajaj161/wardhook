"""Tests for model resolution and tool normalisation."""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as make_tool

from wardhook.core.models import DEFAULT_MODEL, ModelResolutionError, describe_model, resolve_model
from wardhook.core.tools import ToolRegistrationError, normalize_tools, tool_names


class TestResolveModel:
    def test_passes_an_instance_through_untouched(self, fake_model):
        assert resolve_model(fake_model) is fake_model

    def test_accepts_any_duck_typed_model(self):
        class Duck:
            def invoke(self, messages, config=None): ...

        duck = Duck()
        assert resolve_model(duck) is duck

    def test_rejects_an_object_without_invoke(self):
        with pytest.raises(ModelResolutionError, match="invoke"):
            resolve_model(object())

    def test_unknown_provider_prefix_is_rejected(self):
        with pytest.raises(ModelResolutionError, match="Unknown provider"):
            resolve_model("acme:super-model-9")

    def test_unrecognisable_bare_name_asks_for_a_prefix(self):
        with pytest.raises(ModelResolutionError, match="Cannot infer a provider"):
            resolve_model("super-model-9")

    @pytest.mark.parametrize(
        ("name", "expected_package"),
        [
            ("claude-opus-5", "langchain_anthropic"),
            ("gpt-4o", "langchain_openai"),
            ("gemini-2.0-flash", "langchain_google_genai"),
            ("anthropic:claude-sonnet-5", "langchain_anthropic"),
        ],
    )
    def test_names_map_to_the_right_provider_package(self, name, expected_package):
        # The provider extras are not installed in the test environment, so a
        # correct mapping surfaces as an install hint naming that package.
        # This asserts the routing without requiring any provider SDK.
        try:
            resolve_model(name)
        except ModelResolutionError as exc:
            assert expected_package in str(exc)
        else:  # pragma: no cover - only when the extra happens to be installed
            pytest.skip(f"{expected_package} is installed; nothing to assert")

    def test_missing_provider_error_names_the_extra_to_install(self):
        with pytest.raises(ModelResolutionError, match=r"wardhook-core\[anthropic\]"):
            resolve_model("claude-opus-5")

    def test_default_model_is_used_when_none_is_given(self):
        with pytest.raises(ModelResolutionError) as excinfo:
            resolve_model(None)
        assert DEFAULT_MODEL in str(excinfo.value)


class TestDescribeModel:
    def test_prefers_an_explicit_model_name(self):
        class Named:
            model_name = "claude-opus-5"

        assert describe_model(Named()) == "claude-opus-5"

    def test_falls_back_to_the_class_name(self):
        class Anonymous:
            pass

        assert describe_model(Anonymous()) == "Anonymous"


class TestNormalizeTools:
    def test_none_and_empty_give_an_empty_list(self):
        assert normalize_tools(None) == []
        assert normalize_tools([]) == []

    def test_wraps_a_plain_callable(self, echo_tool):
        tools = normalize_tools([echo_tool])
        assert isinstance(tools[0], BaseTool)
        assert tools[0].name == "lookup_account"

    def test_uses_the_docstring_as_the_model_facing_description(self, echo_tool):
        assert (
            normalize_tools([echo_tool])[0].description == "Look up an account by its identifier."
        )

    def test_undocumented_callable_is_rejected(self):
        def mystery(x: int) -> int:
            return x

        with pytest.raises(ToolRegistrationError, match="docstring"):
            normalize_tools([mystery])

    def test_passes_a_prebuilt_tool_through(self):
        @make_tool
        def ping() -> str:
            """Return pong."""
            return "pong"

        assert normalize_tools([ping])[0] is ping

    def test_duplicate_names_are_rejected(self, echo_tool):
        with pytest.raises(ToolRegistrationError, match="Duplicate tool name"):
            normalize_tools([echo_tool, echo_tool])

    def test_non_callable_entry_is_rejected(self):
        with pytest.raises(ToolRegistrationError, match="BaseTool or a callable"):
            normalize_tools([42])

    def test_preserves_input_order(self):
        def alpha() -> str:
            """First."""
            return "a"

        def beta() -> str:
            """Second."""
            return "b"

        assert tool_names(normalize_tools([alpha, beta])) == ["alpha", "beta"]
