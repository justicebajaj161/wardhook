"""Attaching a tracer to a config the caller already owns.

Kept out of ``test_instrument.py`` deliberately: that file skips entirely when
LangGraph is absent, and none of this needs LangGraph. In a standalone install
these are the only tests covering the attachment logic.
"""

from __future__ import annotations

import uuid

from wardhook.observability import Tracer
from wardhook.observability.callbacks import GraphTraceCallback
from wardhook.observability.instrument import _with_tracing


class Manager:
    """Stands in for a LangChain ``CallbackManager``, which is not a list."""

    def __init__(self, handlers=None):
        self.handlers = list(handlers or [])

    def add_handler(self, handler, inherit=True):
        self.handlers.append(handler)


class TestAttachingToAnExistingConfig:
    def test_a_config_with_no_callbacks_gets_a_fresh_list(self):
        tracer = Tracer()
        merged = _with_tracing(None, tracer)
        assert [type(h) for h in merged["callbacks"]] == [GraphTraceCallback]

    def test_the_callers_own_handlers_are_preserved(self):
        tracer = Tracer()
        mine = object()
        merged = _with_tracing({"callbacks": [mine]}, tracer)

        assert merged["callbacks"][0] is mine
        assert isinstance(merged["callbacks"][1], GraphTraceCallback)

    def test_a_callback_manager_is_added_to_rather_than_replaced(self):
        # A CallbackManager is not a list. Replacing it with one would drop
        # every handler the caller had configured on it.
        tracer = Tracer()
        manager = Manager()
        merged = _with_tracing({"callbacks": manager}, tracer)

        assert merged["callbacks"] is manager
        assert isinstance(manager.handlers[-1], GraphTraceCallback)

    def test_a_manager_without_add_handler_is_left_untouched(self):
        # Nothing safe can be done, so the config passes through rather than
        # raising in the middle of someone else's invocation.
        class Opaque:
            handlers = ()

        tracer = Tracer()
        opaque = Opaque()
        merged = _with_tracing({"callbacks": opaque}, tracer)

        assert merged["callbacks"] is opaque

    def test_a_tracer_already_attached_is_not_attached_twice(self):
        # LangGraph's invoke is built on its own stream. Wrapping both without
        # this check attaches two handlers and double-counts every node.
        tracer = Tracer()
        first = _with_tracing(None, tracer)
        second = _with_tracing(first, tracer)

        assert len(second["callbacks"]) == 1

    def test_a_different_tracer_does_get_its_own_handler(self):
        first = _with_tracing(None, Tracer())
        second = _with_tracing(first, Tracer())

        assert len(second["callbacks"]) == 2


class TestGraphTraceCallbackWithoutAGraph:
    """The handler is driven directly, as a provider's callback stream would.

    LangGraph is not required to exercise it: the handler reads ``metadata``
    for a node name and nothing else, which is the whole point of the design.
    """

    def _handler(self):
        tracer = Tracer()
        return GraphTraceCallback(tracer), tracer

    def test_the_outermost_chain_opens_the_run(self):
        handler, tracer = self._handler()
        run_id = uuid.uuid4()
        handler.on_chain_start({}, {}, run_id=run_id, metadata={})
        handler._close(run_id, None)

        assert tracer.get_trace(str(run_id)) is not None

    def test_a_second_non_node_chain_does_not_open_a_second_run(self):
        # A graph nests plenty of unnamed chains. Each one opening a run would
        # produce a trace per internal runnable instead of one per invocation.
        handler, tracer = self._handler()
        outer, inner = uuid.uuid4(), uuid.uuid4()
        handler.on_chain_start({}, {}, run_id=outer, metadata={})
        handler.on_chain_start({}, {}, run_id=inner, metadata={})
        handler._close(inner, None)
        handler._close(outer, None)

        assert tracer.get_trace(str(outer)) is not None
        assert tracer.get_trace(str(inner)) is None

    def test_a_node_arriving_first_adopts_its_own_run(self):
        # Attaching partway through a run means the first event seen is a node,
        # with no enclosing chain. The node's run is adopted rather than lost.
        handler, tracer = self._handler()
        node_run = uuid.uuid4()
        handler.on_chain_start({}, {}, run_id=node_run, metadata={"langgraph_node": "call_model"})
        handler._close(node_run, None)
        tracer.end_run(str(node_run))

        trace = tracer.get_trace(str(node_run))
        assert [s.node for s in trace.steps] == ["call_model"]

    def test_closing_an_unknown_chain_is_ignored(self):
        handler, tracer = self._handler()
        handler.on_chain_start({}, {}, run_id=uuid.uuid4(), metadata={})
        handler._close(uuid.uuid4(), None)

        assert tracer.get_trace() is None

    def test_a_failed_chain_closes_the_node_with_its_error(self):
        handler, tracer = self._handler()
        outer, node = uuid.uuid4(), uuid.uuid4()
        handler.on_chain_start({}, {}, run_id=outer, metadata={})
        handler.on_chain_start({}, {}, run_id=node, metadata={"langgraph_node": "call_model"})
        handler.on_chain_error(RuntimeError("node blew up"), run_id=node)
        handler._close(outer, None)

        (step,) = tracer.get_trace(str(outer)).steps
        assert step.error == "RuntimeError: node blew up"
