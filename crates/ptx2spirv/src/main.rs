//! ptx2spv: compile a PTX file to SPIR-V.  Usage: ptx2spv input.ptx [output.spv]
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} input.ptx [output.spv]", args[0]);
        exit(2);
    }
    let src = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("{}: {e}", args[1]);
        exit(1)
    });
    let out = ptx2spirv::compile_ptx(&src, &ptx2spirv::Options::default()).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        exit(1)
    });
    for k in &out.kernels {
        let ps: Vec<String> = k.params.iter().map(|p| format!("{}:{}", p.name, p.size)).collect();
        eprintln!("kernel {} ({}) dyn_shared_arg={:?} static_shared={}", k.name, ps.join(", "), k.dyn_shared_arg, k.static_shared);
    }
    for g in &out.globals {
        eprintln!("global {} size {}", g.name, g.size);
    }
    let path = args.get(2).cloned().unwrap_or_else(|| args[1].trim_end_matches(".ptx").to_string() + ".spv");
    std::fs::write(&path, out.spirv_bytes()).unwrap();
}
