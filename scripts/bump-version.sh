#!/usr/bin/env bash
# Set the version across all four packages.
#
# The four packages are versioned in lockstep while the project is pre-1.0, and
# each declares its version in two places: its `pyproject.toml` and the
# `__version__` in its `__init__.py`. Missing one of the eight is easy and the
# release workflow will reject the tag, so this does all eight at once.
#
#   ./scripts/bump-version.sh 0.2.0

set -euo pipefail

PACKAGES=(wardhook-core wardhook-guardrails wardhook-observability wardhook-evals)

if [ $# -ne 1 ]; then
    echo "usage: $0 <version>    e.g. $0 0.2.0" >&2
    exit 2
fi

version="$1"
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([abrc.-][0-9a-zA-Z.]*)?$ ]]; then
    echo "error: '$version' does not look like a PEP 440 version" >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for package in "${PACKAGES[@]}"; do
    module="${package#wardhook-}"
    pyproject="$root/packages/$package/pyproject.toml"
    init="$root/packages/$package/src/wardhook/$module/__init__.py"

    for file in "$pyproject" "$init"; do
        [ -f "$file" ] || { echo "error: missing $file" >&2; exit 1; }
    done

    # Only the first `version = ` line: later ones belong to dependency pins.
    perl -0pi -e "s/^version = \"[^\"]+\"/version = \"$version\"/m unless \$done{'v'}++" "$pyproject"
    perl -pi -e "s/^__version__ = \"[^\"]+\"/__version__ = \"$version\"/" "$init"

    printf '  %-24s %s\n' "$package" "$version"
done

echo
echo "Now update CHANGELOG.md, then:"
echo "  git commit -am 'Release $version' && git tag -a v$version -m 'Release $version'"
