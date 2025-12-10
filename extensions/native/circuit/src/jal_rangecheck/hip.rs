use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipHIP;
use openvm_hip_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::HipBackend, types::F,
};
use openvm_hip_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::{JalRangeCheckCols, JalRangeCheckRecord};
use crate::hip_abi::native_jal_rangecheck_hip;

#[derive(new)]
pub struct JalRangeCheckHip {
    pub range_checker: Arc<VariableRangeCheckerChipHIP>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, HipBackend> for JalRangeCheckHip {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        const RECORD_SIZE: usize = size_of::<JalRangeCheckRecord<F>>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }
        assert_eq!(records.len() % RECORD_SIZE, 0);

        let width = JalRangeCheckCols::<F>::width();

        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, width);

        let d_records = records.to_device().unwrap();

        unsafe {
            native_jal_rangecheck_hip::tracegen(
                trace.buffer(),
                padded_height,
                width,
                &d_records,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}
