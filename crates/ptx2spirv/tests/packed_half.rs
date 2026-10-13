//! Packed-half (`.f16x2` / `.bf16x2`) support in the PTX→SPIR-V compiler.
//!
//! ggml/llama.cpp's `half2` hot paths (`__hadd2`, `__hmul2`, `__hfma2`, …)
//! lower to packed-half PTX. These kernels must compile; numeric correctness
//! on the OpenCL CPU runtime is checked separately (see docs/OLLAMA.md).

use ptx2spirv::{compile_ptx, Options};

const KERNEL: &str = r#"
.version 7.0
.target sm_52
.address_size 64
.visible .entry htest( .param .u64 p )
{
    .reg .b32 %r<8>;
    .reg .f32 %f<4>;
    .reg .u64 %rd<2>;
    ld.param.u64 %rd1, [p];
    mov.f32 %f1, 0f3F800000;
    mov.f32 %f2, 0f40000000;
    cvt.rn.f16x2.f32 %r1, %f2, %f1;
    cvt.rn.f16x2.f32 %r2, %f1, %f2;
    add.rn.f16x2 %r3, %r1, %r2;
    mul.rn.f16x2 %r4, %r1, %r2;
    fma.rn.f16x2 %r5, %r1, %r2, %r3;
    min.f16x2 %r6, %r4, %r5;
    max.f16x2 %r7, %r4, %r5;
    neg.f16x2 %r6, %r6;
    st.global.b32 [%rd1], %r7;
    st.global.b32 [%rd1+4], %r6;
    cvt.rn.bf16x2.f32 %r1, %f2, %f1;
    add.rn.bf16x2 %r2, %r1, %r1;
    mul.rn.bf16x2 %r3, %r1, %r2;
    st.global.b32 [%rd1+8], %r3;
    ret;
}
"#;

#[test]
fn packed_half_kernel_compiles() {
    let out = compile_ptx(KERNEL, &Options::default()).expect("packed-half kernel should compile");
    assert!(!out.spirv.is_empty(), "no SPIR-V emitted");
    assert_eq!(out.kernels.len(), 1);
    assert_eq!(out.kernels[0].name, "htest");
    // SPIR-V magic number — the module is at least well-formed enough to start.
    assert_eq!(out.spirv[0], 0x0723_0203);
}

/// Unpacking a packed-half value via `cvt` (as opposed to `mov.b32`) is not a
/// real PTX pattern and must fail clearly rather than miscompile.
#[test]
fn cvt_from_packed_half_errors() {
    let src = r#"
.version 7.0
.target sm_52
.address_size 64
.visible .entry k( .param .u64 p )
{
    .reg .b32 %r<2>;
    .reg .f32 %f<2>;
    .reg .u64 %rd<2>;
    ld.param.u64 %rd1, [p];
    ld.global.b32 %r1, [%rd1];
    cvt.f32.f16x2 %f1, %r1;
    st.global.f32 [%rd1], %f1;
    ret;
}
"#;
    let err = compile_ptx(src, &Options::default()).unwrap_err();
    assert!(err.contains("packed-half"), "unexpected error: {err}");
}
