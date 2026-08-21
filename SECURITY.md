# Security — public lab repo

This repository is **public**. Treat every commit as world-readable.

## Never commit

- Cloud or LAN **IPs**, SSH ports, hostnames, account IDs
- API keys, Hugging Face tokens (`hf_…`), `.env` files
- Private checkpoints or weight hashes that are not on a public Hub page
- Personal names, emails, or contact details

Scripts in this tree talk to `http://127.0.0.1:8000` only.

## Gitleaks

- Config: [`.gitleaks.toml`](.gitleaks.toml)
- CI: [`.github/workflows/gitleaks.yml`](.github/workflows/gitleaks.yml) (full git history)
- Local pre-commit: `brew install gitleaks` then `bash scripts/install-gitleaks-hook.sh`

```bash
gitleaks detect --source . --config .gitleaks.toml
```

Do not use `gitleaks detect --no-git` as the routine check: it scans the working tree, including gitignored files.

## GitHub

Turn on **Secret scanning** and **Push protection** under Settings → Code security. Public repos get native scanning; Gitleaks CI is the extra net for IPs and lab-specific patterns.
