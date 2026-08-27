#!/usr/bin/env bash
# Validate the GitHub Actions workflows without pushing them.
#
# Two checks, both for mistakes that are invisible until a runner picks the
# workflow up:
#   1. The YAML parses.
#   2. Every `run:` block is syntactically valid bash.
#
# The second one exists because an unterminated quote in a `run:` block is
# perfectly valid YAML -- it is just a broken shell script, and the only way to
# find out is to burn a CI run. `bash -n` parses without executing.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "$root" <<'PY'
import pathlib
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("check-workflows: PyYAML is required (it is in the dev group; try `uv run`)")

root = pathlib.Path(sys.argv[1])
workflows = sorted((root / ".github" / "workflows").glob("*.yml"))
if not workflows:
    sys.exit("check-workflows: no workflows found -- wrong directory?")

problems = 0
for workflow in workflows:
    try:
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"  INVALID YAML  {workflow.name}: {exc}")
        problems += 1
        continue

    steps = 0
    for job_name, job in (document.get("jobs") or {}).items():
        for index, step in enumerate(job.get("steps") or []):
            script = step.get("run")
            if not script:
                continue
            steps += 1
            result = subprocess.run(
                ["bash", "-n"], input=script, text=True, capture_output=True
            )
            if result.returncode != 0:
                name = step.get("name", f"step {index}")
                first = result.stderr.strip().splitlines()[0] if result.stderr else "syntax error"
                print(f"  BAD SHELL     {workflow.name} :: {job_name} :: {name}")
                print(f"                {first}")
                problems += 1

    print(f"  OK            {workflow.name} ({steps} run blocks)")

sys.exit(1 if problems else 0)
PY
