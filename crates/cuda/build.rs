fn main() {
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target == "linux" {
        // Applications link against libcuda.so.1.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libcuda.so.1");
    }
}
