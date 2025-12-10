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
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .watch("cuda")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../../crates/vm/cuda")
            .library_name("tracegen_gpu_rv32im")
            .files_from_glob("cuda/src/**/*.cu");

        builder.emit_link_directives();
        builder.build();
    }

    #[cfg(feature = "rocm")]
    {
        if !hip_available() {
            return; // Skip HIP compilation
        }

        // HIP builder reuses the same CUDA kernel sources - hipcc compiles .cu files natively
        let builder: HipBuilder = HipBuilder::new()
            .include_from_dep("DEP_HIP_COMMON_INCLUDE")
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Shared headers (launcher.cuh, fp.h, etc.)
            .include("cuda/include")
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .watch("cuda")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../../crates/vm/cuda")
            .library_name("tracegen_hip_rv32im")
            .files_from_glob("cuda/src/**/*.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
