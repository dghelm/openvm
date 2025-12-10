use std::sync::Arc;

use openvm_circuit::{
    system::memory::{
        persistent::PersistentBoundaryCols, volatile::VolatileBoundaryCols,
        TimestampedEquipartition, TimestampedValues,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipHIP;
use openvm_hip_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prelude::F, prover_backend::HipBackend,
};
use openvm_hip_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator},
    prover::{hal::MatrixDimensions, types::AirProvingContext},
    Chip,
};

use super::{merkle_tree::TIMESTAMPED_BLOCK_WIDTH, poseidon2::SharedBuffer};
use crate::hip_abi::boundary::{persistent_boundary_tracegen, volatile_boundary_tracegen};

pub struct PersistentBoundary {
    pub poseidon2_buffer: SharedBuffer<F>,
    /// A `Vec` of pointers to the copied guest memory on device.
    /// This struct cannot own the device memory, hence we take extra care not to use memory we
    /// don't own. TODO: use `Arc<DeviceBuffer>` instead?
    pub initial_leaves: Vec<*const std::ffi::c_void>,
    pub touched_blocks: Option<DeviceBuffer<u32>>,
}

pub struct VolatileBoundary {
    pub range_checker: Arc<VariableRangeCheckerChipHIP>,
    pub as_max_bits: usize,
    pub ptr_max_bits: usize,
    pub records: Option<Vec<u32>>,
}

pub enum BoundaryFields {
    Persistent(PersistentBoundary),
    Volatile(VolatileBoundary),
}

pub struct BoundaryChipHIP {
    pub fields: BoundaryFields,
    pub num_records: Option<usize>,
    pub trace_width: Option<usize>,
}

impl BoundaryChipHIP {
    pub fn persistent(poseidon2_buffer: SharedBuffer<F>) -> Self {
        Self {
            fields: BoundaryFields::Persistent(PersistentBoundary {
                poseidon2_buffer,
                initial_leaves: Vec::new(),
                touched_blocks: None,
            }),
            num_records: None,
            trace_width: None,
        }
    }

    pub fn volatile(
        range_checker: Arc<VariableRangeCheckerChipHIP>,
        as_max_bits: usize,
        ptr_max_bits: usize,
    ) -> Self {
        Self {
            fields: BoundaryFields::Volatile(VolatileBoundary {
                range_checker,
                as_max_bits,
                ptr_max_bits,
                records: None,
            }),
            num_records: None,
            trace_width: None,
        }
    }

    // Records in the buffer are series of u32s. A single record consts
    // of [as, ptr, timestamp, values[0], ..., values[CHUNK - 1]].
    pub fn finalize_records_volatile<const CHUNK: usize>(
        &mut self,
        final_memory: TimestampedEquipartition<F, CHUNK>,
    ) {
        match &mut self.fields {
            BoundaryFields::Persistent(_) => panic!("call `finalize_records_persistent`"),
            BoundaryFields::Volatile(fields) => {
                self.num_records = Some(final_memory.len());
                self.trace_width = Some(VolatileBoundaryCols::<F>::width());
                let records: Vec<_> = final_memory
                    .par_iter()
                    .flat_map(|&((addr_space, ptr), ts_values)| {
                        let TimestampedValues { timestamp, values } = ts_values;
                        let mut record = vec![addr_space, ptr, timestamp];
                        record.extend_from_slice(&values.map(|x| x.as_canonical_u32()));
                        record
                    })
                    .collect();
                fields.records = Some(records);
            }
        }
    }

    pub fn finalize_records_persistent<const CHUNK: usize>(
        &mut self,
        touched_blocks: DeviceBuffer<u32>,
    ) {
        match &mut self.fields {
            BoundaryFields::Volatile(_) => panic!("call `finalize_records_volatile`"),
            BoundaryFields::Persistent(fields) => {
                self.num_records = Some(touched_blocks.len() / TIMESTAMPED_BLOCK_WIDTH);
                self.trace_width = Some(PersistentBoundaryCols::<F, CHUNK>::width());
                fields.touched_blocks = Some(touched_blocks);
            }
        }
    }

    pub fn trace_width(&self) -> usize {
        self.trace_width.expect("Finalize records to get width")
    }
}

impl<RA> Chip<RA, HipBackend> for BoundaryChipHIP {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<HipBackend> {
        let num_records = self.num_records.unwrap();
        if num_records == 0 {
            return get_empty_air_proving_ctx();
        }
        let unpadded_height = match &self.fields {
            BoundaryFields::Persistent(_) => 2 * num_records,
            BoundaryFields::Volatile(_) => num_records,
        };
        let trace_height = next_power_of_two_or_zero(unpadded_height);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        match &self.fields {
            BoundaryFields::Persistent(boundary) => {
                let mem_ptrs = boundary.initial_leaves.to_device().unwrap();
                unsafe {
                    persistent_boundary_tracegen(
                        trace.buffer(),
                        trace.height(),
                        trace.width(),
                        &mem_ptrs,
                        boundary.touched_blocks.as_ref().unwrap(),
                        num_records,
                        &boundary.poseidon2_buffer.buffer,
                        &boundary.poseidon2_buffer.idx,
                    )
                    .expect("Failed to generate persistent boundary trace");
                }
            }
            BoundaryFields::Volatile(boundary) => unsafe {
                let records = boundary
                    .records
                    .as_ref()
                    .expect("Records must be finalized before generating trace");
                let records = records.to_device().unwrap();
                volatile_boundary_tracegen(
                    trace.buffer(),
                    trace.height(),
                    trace.width(),
                    &records,
                    num_records,
                    &boundary.range_checker.count,
                    boundary.as_max_bits,
                    boundary.ptr_max_bits,
                )
                .expect("Failed to generate volatile boundary trace");
            },
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
