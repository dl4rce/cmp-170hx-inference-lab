#!/usr/bin/env bash
# Install the house gitleaks pre-commit hook into this repo's .git/hooks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK="${ROOT}/.git/hooks/pre-commit"
SRC="${ROOT}/scripts/gitleaks-pre-commit.sh"
[ -f "${SRC}" ] || { echo "missing ${SRC}" >&2; exit 1; }
if [ -f "${HOOK}" ] && ! grep -q "gitleaks pre-commit guard" "${HOOK}"; then
  cp "${HOOK}" "${HOOK}.bak_$(date +%Y%m%dT%H%M%S%z)"
fi
cp "${SRC}" "${HOOK}"
chmod +x "${HOOK}"
echo "installed ${HOOK}"
echo "needs: brew install gitleaks"
