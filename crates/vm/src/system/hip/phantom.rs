use std::mem::size_of;

use derive_new::new;
use openvm_circuit::{
    arch::DenseRecordArena,
    system::phantom::{PhantomCols, PhantomRecord},
    utils::next_power_of_two_or_zero,
};
use openvm_hip_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::HipBackend, types::F,
};
use openvm_hip_common::copy::MemCopyH2D;
use openvm_stark_backend::{
    prover::{hal::MatrixDimensions, types::AirProvingContext},
    Chip,
};

use crate::hip_abi::phantom;

#[derive(new)]
pub struct PhantomChipHIP;

impl PhantomChipHIP {
    pub fn trace_height(arena: &DenseRecordArena) -> usize {
        let record_size = size_of::<PhantomRecord>();
        let records_len = arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    pub fn trace_width() -> usize {
        PhantomCols::<F>::width()
    }
}

impl Chip<DenseRecordArena, HipBackend> for PhantomChipHIP {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        let num_records = Self::trace_height(&arena);
        if num_records == 0 {
            return get_empty_air_proving_ctx();
        }
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, Self::trace_width());
        unsafe {
            phantom::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &arena.allocated().to_device().unwrap(),
            )
            .expect("Failed to generate trace");
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
