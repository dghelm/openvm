use std::{mem::size_of, sync::Arc};

use openvm_circuit::{system::program::ProgramExecutionCols, utils::next_power_of_two_or_zero};
use openvm_hip_backend::{
    base::DeviceMatrix, hip_device::HipDevice, prover_backend::HipBackend, types::F,
};
use openvm_hip_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_instructions::{
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SystemOpcode,
};
use openvm_stark_backend::{
    prover::{
        hal::{MatrixDimensions, TraceCommitter},
        types::{AirProvingContext, CommittedTraceData},
    },
    Chip,
};
use p3_field::FieldAlgebra;

use crate::hip_abi::program;

pub struct ProgramChipHIP {
    pub cached: Option<CommittedTraceData<HipBackend>>,
}

impl ProgramChipHIP {
    pub fn new() -> Self {
        Self { cached: None }
    }

    pub fn generate_cached_trace(program: Program<F>) -> DeviceMatrix<F> {
        let instructions = program
            .enumerate_by_pc()
            .into_iter()
            .map(|(pc, instruction, _)| {
                [
                    F::from_canonical_u32(pc),
                    instruction.opcode.to_field(),
                    instruction.a,
                    instruction.b,
                    instruction.c,
                    instruction.d,
                    instruction.e,
                    instruction.f,
                    instruction.g,
                ]
            })
            .collect::<Vec<_>>();

        let num_records = instructions.len();
        let height = next_power_of_two_or_zero(num_records);
        let records = instructions
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .to_device()
            .unwrap();

        let trace = DeviceMatrix::<F>::with_capacity(height, size_of::<ProgramExecutionCols<u8>>());
        unsafe {
            program::cached_tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &records,
                program.pc_base,
                DEFAULT_PC_STEP,
                SystemOpcode::TERMINATE.global_opcode().as_usize(),
            )
            .expect("Failed to generate cached trace");
        }
        trace
    }

    pub fn get_committed_trace(
        trace: DeviceMatrix<F>,
        device: &HipDevice,
    ) -> CommittedTraceData<HipBackend> {
        let (root, pcs_data) = device.commit(&[trace.clone()]);
        CommittedTraceData {
            commitment: root,
            trace,
            data: pcs_data,
        }
    }
}

impl Default for ProgramChipHIP {
    fn default() -> Self {
        Self::new()
    }
}

impl Chip<Vec<u32>, HipBackend> for ProgramChipHIP {
    fn generate_proving_ctx(&self, filtered_exec_freqs: Vec<u32>) -> AirProvingContext<HipBackend> {
        let cached = self.cached.clone().expect("Cached program must be loaded");
        let height = cached.trace.height();
        let filtered_len = filtered_exec_freqs.len();
        assert!(
            filtered_len <= height,
            "filtered_exec_freqs len={} > cached trace height={}",
            filtered_len,
            height
        );
        let mut buffer: DeviceBuffer<F> = DeviceBuffer::with_capacity(height);

        filtered_exec_freqs
            .into_iter()
            .map(F::from_canonical_u32)
            .collect::<Vec<_>>()
            .copy_to(&mut buffer)
            .unwrap();
        // Making sure to zero-out the untouched part of the buffer.
        if filtered_len < height {
            buffer.fill_zero_suffix(filtered_len).unwrap();
        }

        let trace = DeviceMatrix::new(Arc::new(buffer), height, 1);

        AirProvingContext {
            cached_mains: vec![cached],
            common_main: Some(trace),
            public_values: vec![],
        }
    }
}
