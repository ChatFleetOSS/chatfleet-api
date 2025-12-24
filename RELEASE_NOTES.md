# ChatFleet API — Release Notes

## v0.1.6
- Auth: Immediate admin on first login if an installer-created promotion intent exists for the email. No more polling delay.
- Startup: Ensure `admin_promotions` indexes (unique email, TTL on `expires_at`).
- Scripts: Added `scripts/verify_admin_promotion.py` to validate end-to-end behavior.
- Docs: API reference notes promotion at register/login.

## v0.1.4
- CI: Build and push multi-arch images (linux/amd64, linux/arm64).
- CI: Optional GHCR PAT login; default to GITHUB_TOKEN when allowed.
- CI: Trivy, Gitleaks, SBOM run non-blocking for first release; Cosign signing.

## v0.1.3
- CI: Fixed invalid `if: secrets.*` expression; moved PAT login to shell step.
- Infra: Tag publishing enabled after org Actions/Packages settings were updated.

## v0.1.0
- Initial public commit with cleaned history and Dockerized backend.
