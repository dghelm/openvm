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

        let builder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("../../rv32-adapters/cuda/include")
            .watch("cuda/src/")
            .watch("../../../crates/circuits/primitives/cuda/include")
            .watch("../../../crates/vm/cuda/include")
            .watch("../../rv32-adapters/cuda/include")
            .library_name("tracegen_gpu_algebra")
            .file("cuda/src/is_eq.cu");

        builder.emit_link_directives();
        builder.build();
    }

    #[cfg(feature = "rocm")]
    {
        if !hip_available() {
            return; // Skip HIP compilation
        }

        let builder = HipBuilder::new()
            .include_from_dep("DEP_HIP_COMMON_INCLUDE")
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Shared headers
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("../../rv32-adapters/cuda/include")
            .watch("cuda/src/")
            .watch("../../../crates/circuits/primitives/cuda/include")
            .watch("../../../crates/vm/cuda/include")
            .watch("../../rv32-adapters/cuda/include")
            .library_name("tracegen_hip_algebra")
            .file("cuda/src/is_eq.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
