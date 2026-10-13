#!/usr/bin/env python3
"""Remove top-level `impl` blocks that reference ROCm/HIP sys crates from the
vendored cuda_types modules, so the crate has no AMD backend dependency.
Operates only on files passed as arguments. Reports what it removed."""
import re
import sys

ROCM = ("hip_runtime_sys", "rocblas_sys", "hipblaslt_sys", "miopen_sys",
        "rocsparse_sys", "hipfft_sys", "rocm_smi_sys", "hip_runtime", "::hip")

def strip(path):
    src = open(path).read()
    out = []
    i = 0
    n = len(src)
    removed = 0
    lines = src.splitlines(keepends=True)
    # Work line-based: find a line starting (col 0) with "impl ", capture until
    # brace depth returns to zero, drop the block if it mentions a ROCm crate.
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if re.match(r"^impl[\s<]", line):
            # capture the block
            depth = 0
            started = False
            block = []
            j = idx
            while j < len(lines):
                block.append(lines[j])
                depth += lines[j].count("{") - lines[j].count("}")
                if "{" in lines[j]:
                    started = True
                if started and depth <= 0:
                    break
                j += 1
            block_text = "".join(block)
            if any(tok in block_text for tok in ROCM):
                removed += 1
                # drop any attribute / doc lines that immediately preceded this
                # block (e.g. a `#[cfg(unix)]` gating a ROCm-only impl)
                while out and (out[-1].lstrip().startswith("#[")
                               or out[-1].lstrip().startswith("///")
                               or out[-1].strip() == ""):
                    out.pop()
                # skip a trailing blank line too
                idx = j + 1
                if idx < len(lines) and lines[idx].strip() == "":
                    idx += 1
                continue
            else:
                out.extend(block)
                idx = j + 1
                continue
        out.append(line)
        idx += 1
    open(path, "w").write("".join(out))
    return removed

total = 0
for p in sys.argv[1:]:
    r = strip(p)
    total += r
    print(f"{p}: removed {r} ROCm impl block(s)")
print(f"TOTAL removed: {total}")
