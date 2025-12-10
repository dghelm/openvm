use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipHIP;
use openvm_hip_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::HipBackend, types::F,
};
use openvm_hip_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::{NativeLoadStoreCoreCols, NativeLoadStoreCoreRecord};
use crate::{
    adapters::{NativeLoadStoreAdapterCols, NativeLoadStoreAdapterRecord},
    hip_abi::native_loadstore_hip,
};

#[derive(new)]
pub struct NativeLoadStoreChipHip<const NUM_CELLS: usize> {
    pub range_checker: Arc<VariableRangeCheckerChipHIP>,
    pub timestamp_max_bits: usize,
}

impl<const NUM_CELLS: usize> Chip<DenseRecordArena, HipBackend>
    for NativeLoadStoreChipHip<NUM_CELLS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        const fn record_size<const N: usize>() -> usize {
            size_of::<(
                NativeLoadStoreAdapterRecord<F, N>,
                NativeLoadStoreCoreRecord<F, N>,
            )>()
        }

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }

        let record_size = record_size::<NUM_CELLS>();
        assert_eq!(records.len() % record_size, 0);

        let height = records.len() / record_size;
        let padded_height = next_power_of_two_or_zero(height);
        let trace_width = NativeLoadStoreAdapterCols::<F, NUM_CELLS>::width()
            + NativeLoadStoreCoreCols::<F, NUM_CELLS>::width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        let d_records = records.to_device().unwrap();

        unsafe {
            native_loadstore_hip::tracegen(
                trace.buffer(),
                padded_height,
                trace_width,
                &d_records,
                &self.range_checker.count,
                NUM_CELLS as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}
