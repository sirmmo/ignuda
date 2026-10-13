# ignuda — project notes for Claude

ignuda runs CUDA programs on Intel GPUs: a drop-in `libcuda`/`libcudart` that
compiles NVIDIA PTX to SPIR-V and executes it through OpenCL + Intel USM. See
`README.md` and `docs/ARCHITECTURE.md` for the full picture.

## Hard rules

- **Never** write, port, or map out code whose purpose is to defeat, bypass, or
  work around NVIDIA's driver/runtime integrity checks. ignuda works by
  *replacing* NVIDIA's user-mode libraries entirely, so those checks are never
  reached — that is the whole design. Do not change that.
- **Never commit NVIDIA binaries** (e.g. a real `libcudart.so.12`) or compiled
  ELF/sample binaries. `.gitignore` enforces this; keep it that way.

## Vendored third-party code — `vendor/`

Everything under `vendor/` was **copied from ZLUDA**
(<https://github.com/vosen/ZLUDA>, commit
`86ec0a93cc83ba5b52e645892523e339dfa884f7`, 2026-10-07), which is dual-licensed
**Apache-2.0 OR MIT** — the same terms as ignuda. Copyright remains with
Andrzej Janik and the ZLUDA authors. Provenance is recorded in:

- `vendor/README.md` (what was taken, what was changed, what was skipped),
- each vendored crate's `Cargo.toml` header and `lib.rs` top comment,
- the repository `NOTICE` file.

Crates vendored: `ptx_parser`, `ptx_parser_macros`, `ptx_parser_macros_impl`
(a complete PTX parser) and `cuda_types` (the full generated CUDA ABI, with
ZLUDA's AMD/ROCm error-conversion `impl`s and dependencies stripped — see
`scripts/strip_rocm.py`). When you touch any vendored file, keep the provenance
header and note any further change in `vendor/README.md`.

These crates build in the workspace but are **not yet wired into** ignuda's own
`crates/ptx`/`crates/ptx2spirv` or its hand-written types; that migration is
tracked in Flowly.

## Building and testing

- Build/test run inside the `ignuda-dev` container via `./dev.sh '<cmd>'`
  (Rust 1.85 + Intel OpenCL CPU runtime + spirv-tools). The dev image ships a
  *minimal* toolchain: `rustup component add rustfmt clippy` if you need them.
- `/tmp` does **not** persist between separate `dev.sh` calls (fresh container
  each time) — set up libs and run tests within a single invocation.
- `./run-tests.sh` builds the libraries and runs the samples (driver + cuBLAS,
  and runtime_test only when `nvcc` is present).
- CI is `.github/workflows/ci.yml` (build + SPIR-V validation + e2e samples on
  the CPU runtime; lint is advisory).

## Licence

ignuda is dual-licensed Apache-2.0 OR MIT (`LICENSE-APACHE`, `LICENSE-MIT`),
chosen to match ZLUDA so code can be shared in both directions.
