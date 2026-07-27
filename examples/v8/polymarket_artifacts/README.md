# Pinned Polymarket model artifacts

This directory is a repository-local, content-addressed store for immutable model
artifacts required by reproducible development diagnostics.

Independent artifacts live below `sha256/<artifact-digest>/`. Artifacts with
relative dependencies live together below `sha256/<bundle-digest>/`, with a
`bundle_manifest.json` that pins every member and the dependency graph. Runtime
validation resolves paths from the repository root, verifies the bundle identity,
then resolves dependencies relative to the verified bundle directory. It never
uses the process working directory.

These artifacts are development inputs only. Vendoring them does not authorize
training, paper trading, live trading, promotion, wallet access, or writes.
