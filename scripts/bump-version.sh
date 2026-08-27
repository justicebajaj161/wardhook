#!/usr/bin/env bash
# Set the version across all four packages.
#
# The packages are versioned in lockstep while the project is pre-1.0. Each of
# the four real ones declares its version twice -- in its `pyproject.toml` and
# as `__version__` in its `__init__.py` -- and the `wardhook` meta-package
# declares it once plus pins all four dependencies to it exactly. Missing any
# one of those eleven is easy, and the release workflow will reject the tag, so
# this does the lot at once.
#
#   ./scripts/bump-version.sh 0.2.0

set -euo pipefail

PACKAGES=(wardhook-core wardhook-guardrails wardhook-observability wardhook-evals)
META=wardhook

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

# The meta-package has no source, but it pins each of the four to an exact
# version. Bumping its own version and leaving the pins behind would publish a
# `wardhook` that installs the previous release.
meta_pyproject="$root/packages/$META/pyproject.toml"
[ -f "$meta_pyproject" ] || { echo "error: missing $meta_pyproject" >&2; exit 1; }
perl -0pi -e "s/^version = \"[^\"]+\"/version = \"$version\"/m unless \$done{'v'}++" "$meta_pyproject"
perl -pi -e "s/(wardhook-[a-z]+(?:\[[a-z]+\])?)==[^\"]+/\$1==$version/g" "$meta_pyproject"
printf '  %-24s %s (and its four pins)\n' "$META" "$version"

echo
echo "Now update CHANGELOG.md, then:"
echo "  git commit -am 'Release $version' && git tag -a v$version -m 'Release $version'"
