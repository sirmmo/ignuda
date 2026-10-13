# Releasing ignuda

Releases are built and published by `.github/workflows/release.yml`. You do not
build release artifacts by hand — pushing a version tag does it.

## Cut a release

1. Make sure `main` is green (the CI workflow passes) and is what you want to
   ship.
2. Tag it and push the tag:

   ```bash
   git checkout main && git pull
   git tag -a v0.1.0 -m "ignuda 0.1.0"
   git push origin v0.1.0
   ```

3. The `Release` workflow then, on the tagged commit:
   - builds `libcuda` / `libcudart` / `libcublas` for Linux (release) and
     cross-builds the DLLs for `x86_64-pc-windows-gnu`,
   - compiles and `spirv-val`-validates the sample SPIR-V,
   - packages two archives and a `SHA256SUMS`,
   - publishes a **GitHub Release** for the tag with those files attached.

Releases are marked **pre-release** while ignuda is an early prototype; flip
`prerelease` in the workflow when a release is production-ready.

## What ships

| Archive | Contents |
| --- | --- |
| `ignuda-<version>-linux-x86_64.tar.gz` | `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12` (+ unversioned `.so` symlinks), licences, `NOTICE`, `README.md` |
| `ignuda-<version>-windows-x86_64.zip` | `nvcuda.dll`, `cudart64_12.dll`, `cublas64_12.dll`, licences, `NOTICE`, `README.md` |
| `SHA256SUMS` | checksums of both archives |

The files are named with the **sonames / DLL names applications expect**, so a
user drops them onto `LD_LIBRARY_PATH` (Linux) or next to the executable
(Windows) in place of NVIDIA's. See the README for how to point an application
at them and to select the device (`IGNUDA_DEVICE_TYPE`).

No GPU and no NVIDIA software are involved in building a release — the same as
CI. Only `x86_64` Linux has actually been run end to end so far (on Intel's
OpenCL CPU runtime); the Windows build is compiled but untested.

## Versioning

Pre-1.0, treat every release as potentially breaking. The crate versions in
`crates/*/Cargo.toml` and the git tag should agree; bump them together in the
commit you tag.

## Dry run

`Release` also runs on **workflow_dispatch** (Actions → Release → Run workflow).
That builds and uploads the archives as workflow artifacts but publishes **no**
GitHub Release — useful to check the packaging before tagging.
