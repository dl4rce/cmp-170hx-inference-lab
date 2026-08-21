#!/usr/bin/env bash
# gitleaks pre-commit guard — scans staged blobs only.
set -euo pipefail

if [ "${SKIP_GITLEAKS:-0}" = "1" ]; then
  echo "gitleaks: SKIPPED via SKIP_GITLEAKS=1" >&2
  exit 0
fi

if ! command -v gitleaks >/dev/null 2>&1; then
  echo "gitleaks not installed — secret scan SKIPPED." >&2
  echo "Install:  brew install gitleaks" >&2
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_ARG=()
if [ -f "${REPO_ROOT}/.gitleaks.toml" ]; then
  CONFIG_ARG=(--config "${REPO_ROOT}/.gitleaks.toml")
fi

if git diff --cached --quiet --diff-filter=ACM; then
  exit 0
fi

TMP="$(mktemp -d)"
cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

git diff --cached --name-only --diff-filter=ACM -z | while IFS= read -r -d '' f; do
  dest="${TMP}/${f}"
  mkdir -p "$(dirname "${dest}")"
  git show ":${f}" > "${dest}" 2>/dev/null || true
done

if gitleaks dir "${TMP}" --no-banner --redact "${CONFIG_ARG[@]}"; then
  exit 0
else
  echo "" >&2
  echo "gitleaks BLOCKED this commit — a secret was found in staged changes." >&2
  echo "Remove it, or SKIP_GITLEAKS=1 git commit ... (you own the risk)." >&2
  exit 1
fi
