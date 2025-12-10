use std::{ptr::null_mut, sync::Arc};

use openvm_circuit::{
    arch::{CustomBorrow, DenseRecordArena, SizedRecord},
    system::memory::adapter::{
        records::{AccessLayout, AccessRecordMut},
        AccessAdapterCols,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipHIP;
use openvm_hip_backend::{base::DeviceMatrix, prelude::F, prover_backend::HipBackend};
use openvm_hip_common::copy::MemCopyH2D;
use openvm_stark_backend::prover::types::AirProvingContext;

use crate::hip_abi::{access_adapters::tracegen, OffsetInfo, NUM_ADAPTERS};

pub struct AccessAdapterInventoryHIP {
    max_access_adapter_n: usize,
    timestamp_max_bits: usize,
    range_checker: Arc<VariableRangeCheckerChipHIP>,
    #[cfg(feature = "metrics")]
    pub(super) unpadded_heights: Vec<usize>,
}

impl AccessAdapterInventoryHIP {
    pub(crate) fn generate_traces_from_records(
        &mut self,
        records: &mut [u8],
    ) -> Vec<Option<DeviceMatrix<F>>> {
        let max_access_adapter_n = &self.max_access_adapter_n;
        let timestamp_max_bits = self.timestamp_max_bits;
        let range_checker = &self.range_checker;

        assert!(max_access_adapter_n.is_power_of_two());
        let cnt_adapters = max_access_adapter_n.ilog2() as usize;
        if records.is_empty() {
            return vec![None; cnt_adapters];
        }

        let mut offsets = Vec::new();
        let mut offset = 0;
        let mut row_ids = [0; NUM_ADAPTERS];

        while offset < records.len() {
            offsets.push(OffsetInfo {
                record_offset: offset as u32,
                adapter_rows: row_ids,
            });
            let layout: AccessLayout = unsafe { records[offset..].extract_layout() };
            let record: AccessRecordMut<'_> = records[offset..].custom_borrow(layout.clone());
            offset += <AccessRecordMut<'_> as SizedRecord<AccessLayout>>::size(&layout);
            let bs = record.header.block_size;
            let lbs = record.header.lowest_block_size;
            for logn in lbs.ilog2()..bs.ilog2() {
                row_ids[logn as usize] += bs >> (1 + logn);
            }
        }

        let d_records = records.to_device().unwrap();
        let d_record_offsets = offsets.to_device().unwrap();
        let widths: [_; NUM_ADAPTERS] = std::array::from_fn(|i| match i {
            0 => size_of::<AccessAdapterCols<u8, 2>>(),
            1 => size_of::<AccessAdapterCols<u8, 4>>(),
            2 => size_of::<AccessAdapterCols<u8, 8>>(),
            3 => size_of::<AccessAdapterCols<u8, 16>>(),
            4 => size_of::<AccessAdapterCols<u8, 32>>(),
            _ => panic!(),
        });
        let unpadded_heights = row_ids
            .iter()
            .take(cnt_adapters)
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        let traces = unpadded_heights
            .iter()
            .enumerate()
            .map(|(i, &h)| match h {
                0 => None,
                h => Some(DeviceMatrix::<F>::with_capacity(
                    next_power_of_two_or_zero(h),
                    widths[i],
                )),
            })
            .collect::<Vec<_>>();
        let trace_ptrs = traces
            .iter()
            .map(|trace| {
                trace
                    .as_ref()
                    .map_or_else(null_mut, |t| t.buffer().as_mut_raw_ptr())
            })
            .collect::<Vec<_>>();
        let d_trace_ptrs = trace_ptrs.to_device().unwrap();
        let d_unpadded_heights = unpadded_heights.to_device().unwrap();
        let d_widths = widths.to_device().unwrap();

        unsafe {
            tracegen(
                &d_trace_ptrs,
                &d_unpadded_heights,
                &d_widths,
                offsets.len(),
                &d_records,
                &d_record_offsets,
                &range_checker.count,
                timestamp_max_bits,
            )
            .unwrap();
        }
        #[cfg(feature = "metrics")]
        {
            self.unpadded_heights = unpadded_heights;
        }

        traces
    }

    pub fn new(
        range_checker: Arc<VariableRangeCheckerChipHIP>,
        max_access_adapter_n: usize,
        timestamp_max_bits: usize,
    ) -> Self {
        Self {
            range_checker,
            max_access_adapter_n,
            timestamp_max_bits,
            #[cfg(feature = "metrics")]
            unpadded_heights: Vec::new(),
        }
    }

    // @dev: mutable borrow is only to update `self.unpadded_heights` for metrics
    pub fn generate_air_proving_ctxs(
        &mut self,
        mut arena: DenseRecordArena,
    ) -> Vec<AirProvingContext<HipBackend>> {
        let records = arena.allocated_mut();
        self.generate_traces_from_records(records)
            .into_iter()
            .map(|trace| AirProvingContext {
                cached_mains: vec![],
                common_main: trace,
                public_values: vec![],
            })
            .collect()
    }
}
