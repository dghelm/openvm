#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::{env, path::PathBuf};

#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

#[cfg(feature = "rocm")]
use openvm_hip_builder::{hip_available, HipBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let builder: CudaBuilder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("cuda/include")
            .watch("cuda")
            .library_name("tracegen_gpu_primitives")
            .files_from_glob("cuda/src/**/*.cu");

        builder.emit_link_directives();
        builder.build();

        // Export include dir for downstream crates:
        let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("cuda")
            .join("include");
        println!("cargo:include={}", include_path.display()); // -> DEP_CIRCUIT_PRIMITIVES_CUDA_INCLUDE
    }

    #[cfg(feature = "rocm")]
    {
        if !hip_available() {
            return; // Skip HIP compilation
        }

        // HIP builder reuses the same CUDA kernel sources - hipcc compiles .cu files natively
        // We need to include from hip-common (for HIP-specific headers) and cuda-common
        // (for shared headers like launcher.cuh, fp.h, baby_bear.hpp)
        let builder: HipBuilder = HipBuilder::new()
            .include_from_dep("DEP_HIP_COMMON_INCLUDE")
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Shared headers (launcher.cuh, fp.h, etc.)
            .include("cuda/include") // Reuse CUDA headers (compatible with __HIPCC__ guards)
            .watch("cuda")
            .library_name("tracegen_hip_primitives")
            .files_from_glob("cuda/src/**/*.cu"); // HipBuilder accepts .cu files

        builder.emit_link_directives();
        builder.build();

        // Export include dir for downstream crates (same headers work for both):
        let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("cuda")
            .join("include");
        println!("cargo:include={}", include_path.display()); // -> DEP_CIRCUIT_PRIMITIVES_CUDA_INCLUDE
    }
}
