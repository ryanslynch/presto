# conda-forge recipe (reference copy)

This directory holds a reference copy of PRESTO's conda-forge recipe. It is **not** used by
the normal meson/pixi build; it documents how PRESTO is packaged for conda-forge.

- `meta.yaml` / `build.sh` — the recipe. It builds `libpresto` + the C tools with meson,
  then the Python package with pip, and vendors ERFA (which has no conda-forge feedstock) as
  a second source.

## First submission

Copy these two files into a fork of
[`conda-forge/staged-recipes`](https://github.com/conda-forge/staged-recipes) at
`recipes/presto/`, fill in the `sha256` of the v6.0.0 GitHub tarball
(`curl -sL https://github.com/scottransom/presto/archive/refs/tags/v6.0.0.tar.gz | sha256sum`),
test locally with `python build-locally.py`, and open a PR. See `../RELEASE.md` for the full
checklist.

## After the feedstock exists

Once `presto-feedstock` is created, its `recipe/meta.yaml` becomes authoritative and
conda-forge's autotick bot proposes version bumps automatically. Keep this copy roughly in
sync when the packaging itself changes (deps, build steps, vendored ERFA version).
