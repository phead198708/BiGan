# Pinned Polymarket model artifacts

This directory is a repository-local, content-addressed store for immutable model
artifacts required by reproducible development diagnostics.

Each artifact lives below `sha256/<digest>/`, and its protocol descriptor repeats
the same SHA-256 digest. Runtime validation resolves the repository-relative path
and verifies the file bytes before any diagnostic opens outcomes.

These artifacts are development inputs only. Vendoring them does not authorize
training, paper trading, live trading, promotion, wallet access, or writes.
