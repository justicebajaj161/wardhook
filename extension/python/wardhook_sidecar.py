"""Long-lived analyzer process behind the Wardhook VS Code extension.

The extension speaks JSON Lines to this script over stdin and stdout: one
request object per line in, one response object per line out, correlated by
an ``id`` field. It exists because importing :mod:`wardhook.guardrails` costs
roughly 160ms, which is unusable per keystroke but negligible once amortised
across a session -- a warm scan of a 38KB file round-trips in about 12ms.

Nothing here writes to stdout except responses, so a stray ``print`` would
corrupt the stream. Diagnostics go to stderr, which the extension surfaces in
its output channel.

**No matched value ever crosses this boundary.** :class:`PIIMatch` stores
offsets rather than the text it matched, so a response describes *where* a
secret is without carrying it. The extension never receives the secret.

Run it directly to try the protocol by hand::

    echo '{"id":1,"op":"ping"}' | python wardhook_sidecar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from wardhook.guardrails import (
    BUILTIN_PACKS,
    EntityPack,
    InjectionDetector,
    PIIDetector,
    PIIRedactor,
    RoleBasedToolPolicy,
    __version__,
    get_pack,
)

PROTOCOL_VERSION = 1

# Detectors are cached per pack: building one compiles every rule's regex, and
# the extension re-sends the same pack on every keystroke.
_DETECTORS: dict[str, PIIDetector] = {}
_REDACTORS: dict[str, PIIRedactor] = {}


def _resolve_pack(pack: str | None, custom_path: str | None) -> tuple[str, Any]:
    """Return a cache key and a pack specification.

    Args:
        pack: Name of a built-in pack, or ``None`` for the default.
        custom_path: Path to a YAML pack, which takes precedence when set.

    Returns:
        A ``(cache_key, spec)`` pair, where ``spec`` is whatever
        :class:`PIIDetector` accepts for its ``pack`` argument.
    """
    if custom_path:
        resolved = Path(custom_path).expanduser()
        loaded = EntityPack.from_yaml(resolved)
        return (f"yaml:{resolved}:{resolved.stat().st_mtime_ns}", loaded)
    name = pack or "default"
    return (f"builtin:{name}", name)


def _detector(pack: str | None, custom_path: str | None) -> PIIDetector:
    """Return a cached detector for the requested pack."""
    key, spec = _resolve_pack(pack, custom_path)
    cached = _DETECTORS.get(key)
    if cached is None:
        cached = PIIDetector(spec)
        _DETECTORS[key] = cached
    return cached


def _redactor(pack: str | None, custom_path: str | None) -> PIIRedactor:
    """Return a cached redactor for the requested pack."""
    key, spec = _resolve_pack(pack, custom_path)
    cached = _REDACTORS.get(key)
    if cached is None:
        cached = PIIRedactor(pack=spec)
        _REDACTORS[key] = cached
    return cached


def _line_index(text: str) -> list[int]:
    """Return the offset at which each line starts.

    VS Code addresses documents by zero-based line and character, while the
    detector reports flat character offsets. One pass here beats calling
    ``str.count`` once per match on a large file.
    """
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _position(starts: list[int], offset: int) -> tuple[int, int]:
    """Convert a flat offset into a zero-based ``(line, character)`` pair."""
    low, high = 0, len(starts) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if starts[mid] <= offset:
            low = mid
        else:
            high = mid - 1
    return (low, offset - starts[low])


def _read_text(payload: dict[str, Any]) -> str:
    """Return the text a request wants analysed.

    A request supplies either ``text`` directly or a ``path`` to read. Reading
    here rather than in the extension keeps large files out of the pipe.
    """
    path = payload.get("path")
    if path:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    return str(payload.get("text", ""))


def op_ping(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    """Report that the analyzer is up, and which library it loaded."""
    return {
        "protocol": PROTOCOL_VERSION,
        "guardrails_version": __version__,
        "python": sys.version.split()[0],
    }


def op_scan(payload: dict[str, Any]) -> dict[str, Any]:
    """Detect entities and return their positions, never their values."""
    text = _read_text(payload)
    detector = _detector(payload.get("pack"), payload.get("customPackPath"))
    starts = _line_index(text)
    matches = []
    for match in detector.detect(text):
        start_line, start_char = _position(starts, match.start)
        end_line, end_char = _position(starts, match.end)
        matches.append(
            {
                "entity": match.entity,
                "start": match.start,
                "end": match.end,
                "length": match.length,
                "severity": match.severity.value,
                "validated": match.validated,
                "replacement": match.replacement,
                "startLine": start_line,
                "startChar": start_char,
                "endLine": end_line,
                "endChar": end_char,
            }
        )
    return {"matches": matches, "length": len(text)}


def op_redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace every detected entity with its typed placeholder."""
    text = _read_text(payload)
    result = _redactor(payload.get("pack"), payload.get("customPackPath")).redact(text)
    return {
        "text": result.text,
        "counts": dict(result.counts),
        "matchCount": len(result.matches),
        "maxSeverity": result.max_severity.value,
        "originalLength": result.original_length,
    }


def op_injection(payload: dict[str, Any]) -> dict[str, Any]:
    """Score text for prompt-injection signals.

    The report carries no offsets, only a whole-text score, so the extension
    renders it as a file-level finding rather than a highlighted span.
    """
    text = _read_text(payload)
    threshold = payload.get("threshold")
    detector = InjectionDetector(threshold=threshold) if threshold else InjectionDetector()
    return detector.score(text).to_dict()


def op_packs(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    """List the built-in packs and the entities each one detects."""
    packs = []
    for name in BUILTIN_PACKS:
        pack = get_pack(name)
        packs.append(
            {
                "name": name,
                "entities": pack.entity_names(),
                "ruleCount": len(pack.rules),
            }
        )
    return {"packs": packs}


def op_explain(payload: dict[str, Any]) -> dict[str, Any]:
    """Describe one entity rule: severity, validator, and required context."""
    entity = str(payload.get("entity", ""))
    pack = get_pack(payload.get("pack") or "default")
    for rule in pack.rules:
        if rule.entity.upper() == entity.upper():
            return {
                "found": True,
                "entity": rule.entity,
                "severity": rule.severity.value,
                "validator": rule.validator,
                "replacement": rule.redaction(),
                "contextWords": list(rule.context_words),
                "description": rule.description,
                "pack": pack.name,
            }
    return {"found": False, "entity": entity, "known": pack.entity_names(), "pack": pack.name}


def op_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Check whether a role may call a named tool.

    The policy is deny-by-default, and roles arrive on the principal as a
    list, so a single role name is wrapped before evaluation.
    """
    policy = payload.get("policy") or {}
    role = str(payload.get("role", ""))
    tool = str(payload.get("tool", ""))
    guard = RoleBasedToolPolicy(policy)
    context = {"principal": {"roles": [role]}}
    result = guard.on_tool_call(tool, {}, context)
    return {
        "role": role,
        "tool": tool,
        "allowed": result.allowed,
        "reason": result.reason,
        "grantedPatterns": guard.permitted_tools([role]),
    }


HANDLERS = {
    "ping": op_ping,
    "scan": op_scan,
    "redact": op_redact,
    "injection": op_injection,
    "packs": op_packs,
    "explain": op_explain,
    "policy": op_policy,
}


def handle(request: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one request and wrap the outcome in a response envelope.

    A handler that raises becomes an error response rather than killing the
    process: one malformed file should not take the analyzer down mid-session.
    """
    req_id = request.get("id")
    handler = HANDLERS.get(str(request.get("op")))
    if handler is None:
        return {"id": req_id, "ok": False, "error": f"unknown op: {request.get('op')!r}"}
    try:
        return {"id": req_id, "ok": True, "result": handler(request)}
    # A broad catch on purpose: the response envelope is the error channel.
    except Exception as exc:
        return {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    """Read requests until stdin closes, writing one response per line.

    The default pack is built before announcing readiness. Loading a pack is
    what first needs PyYAML, so doing it here means "ready" states that the
    analyzer works -- rather than deferring the failure to the first scan.
    """
    _detector(None, None)
    ready = {"ok": True, "result": op_ping({}), "id": None, "event": "ready"}
    sys.stdout.write(json.dumps(ready) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response: dict[str, Any] = {"id": None, "ok": False, "error": f"bad JSON: {exc}"}
        else:
            response = handle(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
