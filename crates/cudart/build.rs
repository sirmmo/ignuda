fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "linux" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libcudart.so.12");
    }
}
