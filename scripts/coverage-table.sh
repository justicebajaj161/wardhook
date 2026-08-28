#!/usr/bin/env bash
# Print the per-package test count and coverage as the README's status table.
#
# The README quotes exact numbers. A reviewer who runs the suite and gets
# different ones stops trusting the rest of the page, so the table has to be
# reproducible by anyone in one command rather than transcribed by hand.
#
#   make cov-table
#
# Coverage is branch coverage (see [tool.coverage.run] in pyproject.toml),
# which reads lower than line coverage and is the stricter number to publish.

set -euo pipefail

PACKAGES=(core guardrails observability evals)
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

printf '| Package | Status | Tests | Coverage |\n'
printf '| --- | --- | --- | --- |\n'

total=0
for module in "${PACKAGES[@]}"; do
    package="wardhook-$module"
    output="$workdir/$package.txt"

    # Both the tests and the source tree: --doctest-modules turns every
    # docstring example into a test, and those count too.
    COVERAGE_FILE="$workdir/.coverage-$module" uv run pytest \
        "$root/packages/$package/tests" "$root/packages/$package/src" \
        --cov="wardhook.$module" --cov-report=term \
        -q -p no:cacheprovider >"$output" 2>&1 || { cat "$output"; exit 1; }

    tests=$(grep -oE '^[0-9]+ passed' "$output" | head -1 | cut -d' ' -f1)
    coverage=$(grep -E '^TOTAL' "$output" | awk '{print $NF}')
    total=$((total + tests))

    printf '| %-24s | ✅ Complete | %s | %s |\n' "$package" "$tests" "$coverage"
done

printf '| %-24s | ✅ Complete | — | n/a |\n' "wardhook (meta)"
echo
echo "$total tests across the four packages."
