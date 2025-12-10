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
            .include("../circuits/primitives/cuda/include")
            .include("../circuits/poseidon2-air/cuda/include")
            .include("cuda/include");
        builder.emit_link_directives();

        builder
            .clone()
            .library_name("tracegen_gpu_system")
            .files_from_glob("cuda/src/system/**/*.cu")
            .build();

        #[cfg(any(test, feature = "test-utils"))]
        {
            builder
                .clone()
                .library_name("tracegen_gpu_testing")
                .files_from_glob("cuda/src/testing/**/*.cu")
                .build();
        }
    }

    #[cfg(feature = "rocm")]
    {
        if !hip_available() {
            return; // Skip HIP compilation
        }

        let builder = HipBuilder::new()
            .include_from_dep("DEP_HIP_COMMON_INCLUDE")
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Shared headers
            .include("../circuits/primitives/cuda/include")
            .include("../circuits/poseidon2-air/cuda/include")
            .include("cuda/include");
        builder.emit_link_directives();

        builder
            .clone()
            .library_name("tracegen_hip_system")
            .files_from_glob("cuda/src/system/**/*.cu")
            .build();

        #[cfg(any(test, feature = "test-utils"))]
        {
            builder
                .clone()
                .library_name("tracegen_hip_testing")
                .files_from_glob("cuda/src/testing/**/*.cu")
                .build();
        }
    }
}
