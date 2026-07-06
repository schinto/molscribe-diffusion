# Project instructions

## Project goal

Extend MolScribe experimentally with diffusion-based decoders while
preserving the existing baseline and checkpoint compatibility.

## Working rules

- Do not change baseline behavior unless explicitly requested.
- Keep the existing MLP graph predictor available.
- Implement new functionality behind configuration flags.
- Prefer small, reviewable changes.
- Add tests for every new module.
- Run relevant tests after each change.
- Do not silently change dataset formats or public model outputs.
- Document tensor shapes and masking semantics.
- Do not commit datasets, checkpoints, model weights, or secrets.

## Python

- Follow the repository's existing Python style.
- Add type hints where they improve clarity.
- Avoid new dependencies unless they are clearly necessary.
- Use PyTorch implementations instead of external diffusion libraries.

## Git

- Keep unrelated refactoring out of feature changes.
- Use focused commits.
- Never force-push or delete branches without explicit approval.

