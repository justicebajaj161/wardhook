"""Durable trace storage as JSON Lines.

One trace per line, keys sorted. That format is chosen for a specific reason:
traces are append-only, are usually read in bulk, and frequently end up in a
diff during a code review of a prompt change. JSON Lines appends without
rewriting, streams without loading the whole file, and -- because the keys are
sorted -- two runs of the same agent produce line-for-line comparable output.

The file is plain text with no index, so it is readable with ``tail``, ``jq``,
or any tool a reader already has. There is no database and nothing to run.

.. warning::
   A trace records node names, timings, token counts, and whatever metadata the
   caller supplied. It does **not** record prompts or model output, so a trace
   file is far less sensitive than a transcript -- but ``metadata`` is written
   verbatim, so do not put user text in it.

Example:
    >>> import tempfile, pathlib
    >>> from wardhook.observability.models import Trace
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "traces.jsonl"
    >>> store = JSONLTraceStore(path)
    >>> store.append(Trace("r1"))
    >>> [t.run_id for t in store.read()]
    ['r1']
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from wardhook.observability.models import Trace

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["JSONLTraceStore", "load_traces"]


class JSONLTraceStore:
    """Appends traces to a JSON Lines file and reads them back.

    Args:
        path: Destination file. Parent directories are created on first write.

    Attributes:
        path: The file being written to.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialise the store. See the class docstring for arguments."""
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, trace: Trace) -> None:
        """Append one trace to the file.

        Args:
            trace: The trace to persist.
        """
        line = json.dumps(trace.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read(self) -> list[Trace]:
        """Read every trace in the file.

        Returns:
            The traces, in the order they were written. An empty list when the
            file does not exist yet -- a store that has recorded nothing is not
            an error.
        """
        return list(self.iter_traces())

    def iter_traces(self) -> Iterator[Trace]:
        """Yield traces one at a time without loading the whole file.

        Yields:
            Each trace in file order.

        Raises:
            ValueError: If a line is not valid JSON, naming the line number.
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}:{number} is not valid JSON: {exc.msg}. "
                        f"Each line of a trace file must be one complete JSON object."
                    ) from exc
                yield Trace.from_dict(payload)

    def read_one(self, run_id: str) -> Trace | None:
        """Return the trace for a single run.

        Args:
            run_id: The run to look for.

        Returns:
            The matching trace, or ``None`` if the file holds no such run.
        """
        for trace in self.iter_traces():
            if trace.run_id == run_id:
                return trace
        return None

    def __len__(self) -> int:
        """Return how many traces the file holds."""
        return sum(1 for _ in self.iter_traces())

    def __repr__(self) -> str:
        """Return a debug representation naming the file."""
        return f"JSONLTraceStore(path={str(self.path)!r})"


def load_traces(path: str | Path) -> list[Trace]:
    """Read every trace from a JSON Lines file.

    A convenience for the common read-only case, so callers that only want to
    render a file do not have to construct a store first.

    Args:
        path: The trace file to read.

    Returns:
        The traces it contains.

    Raises:
        FileNotFoundError: If the file does not exist. Unlike
            :meth:`JSONLTraceStore.read`, an explicit read of a named file
            should fail loudly rather than return nothing.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"No trace file at {resolved}")
    return JSONLTraceStore(resolved).read()
