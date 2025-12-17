//! Hybrid HIP prover extension for native chips.
//!
//! This module implements the hybrid CPU→GPU approach for native chips, which has been proven
//! to produce correct proofs. The native GPU trace generation kernels (in cuda/src/*.cu) are
//! disabled by default because they produce invalid proofs with OodEvaluationMismatch errors.
//!
//! To re-enable native GPU trace generation for debugging/optimization, use the `hip-native-chips`
//! feature flag.

use std::mem::size_of;

use openvm_circuit::{
    arch::{
        Arena, ChipInventory, ChipInventoryError, DenseRecordArena, EmptyAdapterCoreLayout,
        MatrixRecordArena, MultiRowLayout, MultiRowMetadata, SizedRecord, VmProverExtension,
    },
    system::{hip::extensions::get_inventory_range_checker, memory::SharedMemoryHelper},
};
use openvm_hip_backend::{
    chip::{cpu_proving_ctx_to_hip, get_empty_air_proving_ctx},
    engine::HipBabyBearPoseidon2Engine,
    prover_backend::HipBackend,
    types::F,
};
use openvm_native_compiler::BLOCK_LOAD_STORE_SIZE;
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::{p3_air::BaseAir, prover::types::AirProvingContext, Chip};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};
use std::borrow::Borrow;

use crate::{
    adapters::{
        AluNativeAdapterCols, AluNativeAdapterExecutor, AluNativeAdapterFiller,
        AluNativeAdapterRecord, BranchNativeAdapterCols, BranchNativeAdapterExecutor,
        BranchNativeAdapterFiller, BranchNativeAdapterRecord, ConvertAdapterCols,
        ConvertAdapterExecutor, ConvertAdapterFiller, ConvertAdapterRecord,
        NativeLoadStoreAdapterCols, NativeLoadStoreAdapterExecutor, NativeLoadStoreAdapterFiller,
        NativeLoadStoreAdapterRecord, NativeVectorizedAdapterCols, NativeVectorizedAdapterExecutor,
        NativeVectorizedAdapterFiller, NativeVectorizedAdapterRecord,
    },
    branch_eq::{
        NativeBranchEqAir, NativeBranchEqChip, NativeBranchEqualCoreRecord, NativeBranchEqualFiller,
    },
    castf::{CastFAir, CastFChip, CastFCoreCols, CastFCoreFiller, CastFCoreRecord},
    field_arithmetic::{
        FieldArithmeticAir, FieldArithmeticChip, FieldArithmeticCoreCols,
        FieldArithmeticCoreFiller, FieldArithmeticRecord,
    },
    field_extension::{
        FieldExtensionAir, FieldExtensionChip, FieldExtensionCoreCols, FieldExtensionCoreFiller,
        FieldExtensionRecord, EXT_DEG,
    },
    fri::{
        FriReducedOpeningAir, FriReducedOpeningChip, FriReducedOpeningFiller,
        FriReducedOpeningHeaderRecord, FriReducedOpeningMetadata, FriReducedOpeningRecordMut,
        OVERALL_WIDTH,
    },
    jal_rangecheck::{
        JalRangeCheckAir, JalRangeCheckCols, JalRangeCheckFiller, JalRangeCheckRecord,
        NativeJalRangeCheckChip,
    },
    loadstore::{
        NativeLoadStoreAir, NativeLoadStoreChip, NativeLoadStoreCoreCols,
        NativeLoadStoreCoreFiller, NativeLoadStoreCoreRecord,
    },
    poseidon2::{
        air::NativePoseidon2Air, chip::NativePoseidon2Filler, columns::NativePoseidon2Cols,
        NativePoseidon2Chip,
    },
    CastFExtension, Native,
};

// ============ Hybrid Chip Wrappers ============
// These wrappers use CPU trace generation + GPU transfer via cpu_proving_ctx_to_hip().
// This approach is proven to work correctly, unlike the native GPU trace generation kernels.

/// Hybrid wrapper for FieldArithmeticChip - uses CPU trace gen + GPU transfer
#[derive(derive_new::new)]
pub struct HybridFieldArithmeticChipHip {
    cpu: FieldArithmeticChip<F>,
}

impl Chip<DenseRecordArena, HipBackend> for HybridFieldArithmeticChipHip {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        type Layout = EmptyAdapterCoreLayout<F, AluNativeAdapterExecutor>;

        let record_size = size_of::<(AluNativeAdapterRecord<F>, FieldArithmeticRecord<F>)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let width = AluNativeAdapterCols::<F>::width() + FieldArithmeticCoreCols::<F>::width();

        let mut seeker = arena.get_record_seeker::<(
            &mut AluNativeAdapterRecord<F>,
            &mut FieldArithmeticRecord<F>,
        ), Layout>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, Layout::new());

        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_hip(ctx)
    }
}

/// Hybrid wrapper for FieldExtensionChip - uses CPU trace gen + GPU transfer
#[derive(derive_new::new)]
pub struct HybridFieldExtensionChipHip {
    cpu: FieldExtensionChip<F>,
}

impl Chip<DenseRecordArena, HipBackend> for HybridFieldExtensionChipHip {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        type Layout = EmptyAdapterCoreLayout<F, NativeVectorizedAdapterExecutor<EXT_DEG>>;

        let record_size = size_of::<(
            NativeVectorizedAdapterRecord<F, EXT_DEG>,
            FieldExtensionRecord<F>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let width = NativeVectorizedAdapterCols::<F, EXT_DEG>::width()
            + FieldExtensionCoreCols::<F>::width();

        let mut seeker = arena.get_record_seeker::<(
            &mut NativeVectorizedAdapterRecord<F, EXT_DEG>,
            &mut FieldExtensionRecord<F>,
        ), Layout>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, Layout::new());

        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_hip(ctx)
    }
}

/// Hybrid wrapper for NativeLoadStoreChip - uses CPU trace gen + GPU transfer
#[derive(derive_new::new)]
pub struct HybridNativeLoadStoreChipHip<const NUM_CELLS: usize> {
    cpu: NativeLoadStoreChip<F, NUM_CELLS>,
}

impl<const NUM_CELLS: usize> Chip<DenseRecordArena, HipBackend>
    for HybridNativeLoadStoreChipHip<NUM_CELLS>
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        type Layout<const N: usize> = EmptyAdapterCoreLayout<F, NativeLoadStoreAdapterExecutor<N>>;

        let record_size = size_of::<(
            NativeLoadStoreAdapterRecord<F, NUM_CELLS>,
            NativeLoadStoreCoreRecord<F, NUM_CELLS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let width = NativeLoadStoreAdapterCols::<F, NUM_CELLS>::width()
            + NativeLoadStoreCoreCols::<F, NUM_CELLS>::width();

        let mut seeker = arena.get_record_seeker::<(
            &mut NativeLoadStoreAdapterRecord<F, NUM_CELLS>,
            &mut NativeLoadStoreCoreRecord<F, NUM_CELLS>,
        ), Layout<NUM_CELLS>>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, Layout::<NUM_CELLS>::new());

        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_hip(ctx)
    }
}

/// Hybrid wrapper for NativeBranchEqChip - uses CPU trace gen + GPU transfer
#[derive(derive_new::new)]
pub struct HybridNativeBranchEqChipHip {
    cpu: NativeBranchEqChip<F>,
}

impl Chip<DenseRecordArena, HipBackend> for HybridNativeBranchEqChipHip {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        type Layout = EmptyAdapterCoreLayout<F, BranchNativeAdapterExecutor>;

        let record_size =
            size_of::<(BranchNativeAdapterRecord<F>, NativeBranchEqualCoreRecord<F>)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let width = BranchNativeAdapterCols::<F>::width()
            + openvm_rv32im_circuit::BranchEqualCoreCols::<F, 1>::width();

        let mut seeker = arena.get_record_seeker::<(
            &mut BranchNativeAdapterRecord<F>,
            &mut NativeBranchEqualCoreRecord<F>,
        ), Layout>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, Layout::new());

        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_hip(ctx)
    }
}

/// Hybrid wrapper for NativeJalRangeCheckChip - uses CPU trace gen + GPU transfer
#[derive(derive_new::new)]
pub struct HybridJalRangeCheckHip {
    cpu: NativeJalRangeCheckChip<F>,
}

impl Chip<DenseRecordArena, HipBackend> for HybridJalRangeCheckHip {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        use openvm_circuit::arch::EmptyMultiRowLayout;

        let record_size = size_of::<JalRangeCheckRecord<F>>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let width = JalRangeCheckCols::<F>::width();

        // JalRangeCheck uses EmptyMultiRowLayout (each record = 1 row)
        let mut seeker =
            arena.get_record_seeker::<&mut JalRangeCheckRecord<F>, EmptyMultiRowLayout>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena);

        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_hip(ctx)
    }
}

/// Hybrid wrapper for FriReducedOpeningChip - uses CPU trace gen + GPU transfer
#[derive(derive_new::new)]
pub struct HybridFriReducedOpeningChipHip {
    cpu: FriReducedOpeningChip<F>,
}

impl Chip<DenseRecordArena, HipBackend> for HybridFriReducedOpeningChipHip {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }

        // FRI uses MultiRowLayout - count total rows needed
        let mut total_rows = 0usize;
        let mut offset = 0usize;
        while offset < records.len() {
            // Read the header to get length (Borrow impl comes from AlignedBytesBorrow derive)
            let header: &FriReducedOpeningHeaderRecord = records[offset..].borrow();
            let metadata = FriReducedOpeningMetadata {
                length: header.length as usize,
                is_init: header.is_init,
            };
            total_rows += MultiRowMetadata::get_num_rows(&metadata);

            // Calculate record size
            let layout = MultiRowLayout::new(metadata);
            let record_size = <FriReducedOpeningRecordMut<'_, F> as SizedRecord<
                MultiRowLayout<FriReducedOpeningMetadata>,
            >>::size(&layout);
            let record_alignment = <FriReducedOpeningRecordMut<'_, F> as SizedRecord<
                MultiRowLayout<FriReducedOpeningMetadata>,
            >>::alignment(&layout);
            offset += record_size.next_multiple_of(record_alignment);
        }

        let height = total_rows.next_power_of_two();
        let width = OVERALL_WIDTH;

        let mut seeker = arena.get_record_seeker::<
            FriReducedOpeningRecordMut<'_, F>,
            MultiRowLayout<FriReducedOpeningMetadata>,
        >();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena);

        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_hip(ctx)
    }
}

/// Hybrid wrapper for NativePoseidon2Chip - uses CPU trace gen + GPU transfer
#[derive(derive_new::new)]
pub struct HybridNativePoseidon2ChipHip<const SBOX_REGISTERS: usize> {
    cpu: NativePoseidon2Chip<F, SBOX_REGISTERS>,
}

impl<const SBOX_REGISTERS: usize> Chip<DenseRecordArena, HipBackend>
    for HybridNativePoseidon2ChipHip<SBOX_REGISTERS>
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        use crate::poseidon2::chip::NativePoseidon2Metadata;
        use crate::poseidon2::chip::NativePoseidon2RecordMut;

        type Layout = MultiRowLayout<NativePoseidon2Metadata>;

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }

        let width = NativePoseidon2Cols::<F, SBOX_REGISTERS>::width();

        // Poseidon2 records are stored as trace rows - each row is one record with num_rows=1
        // The total size is num_rows * sizeof(NativePoseidon2Cols)
        let record_size = width * size_of::<F>();
        debug_assert_eq!(records.len() % record_size, 0);
        let num_rows = records.len() / record_size;
        let height = num_rows.next_power_of_two();

        // Use seeker pattern to properly transfer records
        let mut seeker =
            arena.get_record_seeker::<NativePoseidon2RecordMut<'_, F, SBOX_REGISTERS>, Layout>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena);

        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_hip(ctx)
    }
}

/// Hybrid wrapper for CastFChip - uses CPU trace gen + GPU transfer
#[derive(derive_new::new)]
pub struct HybridCastFChipHip {
    cpu: CastFChip<F>,
}

impl Chip<DenseRecordArena, HipBackend> for HybridCastFChipHip {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<HipBackend> {
        type Layout = EmptyAdapterCoreLayout<F, ConvertAdapterExecutor<1, 4>>;

        let record_size = size_of::<(ConvertAdapterRecord<F, 1, 4>, CastFCoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<HipBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let width = ConvertAdapterCols::<F, 1, 4>::width() + CastFCoreCols::<F>::width();

        let mut seeker = arena.get_record_seeker::<(
            &mut ConvertAdapterRecord<F, 1, 4>,
            &mut CastFCoreRecord,
        ), Layout>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, Layout::new());

        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_hip(ctx)
    }
}

// ============ Prover Extension ============

pub struct NativeHipProverExt;

// This implementation uses hybrid CPU→GPU chips for correctness.
// The native GPU trace generation kernels are disabled because they produce invalid proofs.
impl VmProverExtension<HipBabyBearPoseidon2Engine, DenseRecordArena, Native>
    for NativeHipProverExt
{
    fn extend_prover(
        &self,
        _: &Native,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, HipBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let range_checker_hip = get_inventory_range_checker(inventory);
        let range_checker = range_checker_hip.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<NativeLoadStoreAir<1>>()?;
        let load_store = HybridNativeLoadStoreChipHip::<1>::new(NativeLoadStoreChip::new(
            NativeLoadStoreCoreFiller::new(NativeLoadStoreAdapterFiller),
            mem_helper.clone(),
        ));
        inventory.add_executor_chip(load_store);

        inventory.next_air::<NativeLoadStoreAir<BLOCK_LOAD_STORE_SIZE>>()?;
        let block_load_store =
            HybridNativeLoadStoreChipHip::<BLOCK_LOAD_STORE_SIZE>::new(NativeLoadStoreChip::new(
                NativeLoadStoreCoreFiller::new(NativeLoadStoreAdapterFiller),
                mem_helper.clone(),
            ));
        inventory.add_executor_chip(block_load_store);

        inventory.next_air::<NativeBranchEqAir>()?;
        let branch_eq = HybridNativeBranchEqChipHip::new(NativeBranchEqChip::new(
            NativeBranchEqualFiller::new(BranchNativeAdapterFiller),
            mem_helper.clone(),
        ));
        inventory.add_executor_chip(branch_eq);

        inventory.next_air::<JalRangeCheckAir>()?;
        let jal_rangecheck = HybridJalRangeCheckHip::new(NativeJalRangeCheckChip::new(
            JalRangeCheckFiller::new(range_checker.clone()),
            mem_helper.clone(),
        ));
        inventory.add_executor_chip(jal_rangecheck);

        inventory.next_air::<FieldArithmeticAir>()?;
        let field_arithmetic = HybridFieldArithmeticChipHip::new(FieldArithmeticChip::new(
            FieldArithmeticCoreFiller::new(AluNativeAdapterFiller),
            mem_helper.clone(),
        ));
        inventory.add_executor_chip(field_arithmetic);

        inventory.next_air::<FieldExtensionAir>()?;
        let field_extension = HybridFieldExtensionChipHip::new(FieldExtensionChip::new(
            FieldExtensionCoreFiller::new(NativeVectorizedAdapterFiller::<EXT_DEG>::new()),
            mem_helper.clone(),
        ));
        inventory.add_executor_chip(field_extension);

        inventory.next_air::<FriReducedOpeningAir>()?;
        let fri_reduced_opening = HybridFriReducedOpeningChipHip::new(FriReducedOpeningChip::new(
            FriReducedOpeningFiller::new(),
            mem_helper.clone(),
        ));
        inventory.add_executor_chip(fri_reduced_opening);

        inventory.next_air::<NativePoseidon2Air<BabyBear, 1>>()?;
        let poseidon2 = HybridNativePoseidon2ChipHip::<1>::new(NativePoseidon2Chip::new(
            NativePoseidon2Filler::new(Poseidon2Config::default()),
            mem_helper.clone(),
        ));
        inventory.add_executor_chip(poseidon2);

        Ok(())
    }
}

impl VmProverExtension<HipBabyBearPoseidon2Engine, DenseRecordArena, CastFExtension>
    for NativeHipProverExt
{
    fn extend_prover(
        &self,
        _: &CastFExtension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, HipBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let range_checker_hip = get_inventory_range_checker(inventory);
        let range_checker = range_checker_hip.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        inventory.next_air::<CastFAir>()?;
        let castf = HybridCastFChipHip::new(CastFChip::new(
            CastFCoreFiller::new(ConvertAdapterFiller::new(), range_checker),
            mem_helper.clone(),
        ));
        inventory.add_executor_chip(castf);

        Ok(())
    }
}
