use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::hip::extensions::get_inventory_range_checker,
};
use openvm_hip_backend::engine::HipBabyBearPoseidon2Engine;
use openvm_native_compiler::BLOCK_LOAD_STORE_SIZE;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};

use crate::{
    branch_eq::{NativeBranchEqAir, NativeBranchEqChipHip},
    castf::{CastFAir, CastFChipHip},
    field_arithmetic::{FieldArithmeticAir, FieldArithmeticChipHip},
    field_extension::{FieldExtensionAir, FieldExtensionChipHip},
    fri::{FriReducedOpeningAir, FriReducedOpeningChipHip},
    jal_rangecheck::{JalRangeCheckAir, JalRangeCheckHip},
    loadstore::{NativeLoadStoreAir, NativeLoadStoreChipHip},
    poseidon2::{air::NativePoseidon2Air, NativePoseidon2ChipHip},
    CastFExtension, HipBackend, Native,
};

pub struct NativeHipProverExt;
// This implementation is specific to HipBackend because the lookup chips
// (VariableRangeCheckerChipHIP, BitwiseOperationLookupChipHIP) are specific to HipBackend.
impl VmProverExtension<HipBabyBearPoseidon2Engine, DenseRecordArena, Native>
    for NativeHipProverExt
{
    fn extend_prover(
        &self,
        _: &Native,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, HipBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let range_checker = get_inventory_range_checker(inventory);

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<NativeLoadStoreAir<1>>()?;
        let load_store =
            NativeLoadStoreChipHip::<1>::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(load_store);

        inventory.next_air::<NativeLoadStoreAir<BLOCK_LOAD_STORE_SIZE>>()?;
        let block_load_store = NativeLoadStoreChipHip::<BLOCK_LOAD_STORE_SIZE>::new(
            range_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(block_load_store);

        inventory.next_air::<NativeBranchEqAir>()?;
        let branch_eq = NativeBranchEqChipHip::new(range_checker.clone(), timestamp_max_bits);

        inventory.add_executor_chip(branch_eq);

        inventory.next_air::<JalRangeCheckAir>()?;
        let jal_rangecheck = JalRangeCheckHip::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jal_rangecheck);

        inventory.next_air::<FieldArithmeticAir>()?;
        let field_arithmetic =
            FieldArithmeticChipHip::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(field_arithmetic);

        inventory.next_air::<FieldExtensionAir>()?;
        let field_extension = FieldExtensionChipHip::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(field_extension);

        inventory.next_air::<FriReducedOpeningAir>()?;
        let fri_reduced_opening =
            FriReducedOpeningChipHip::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(fri_reduced_opening);

        inventory.next_air::<NativePoseidon2Air<BabyBear, 1>>()?;
        let poseidon2 = NativePoseidon2ChipHip::<1>::new(range_checker.clone(), timestamp_max_bits);
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
        let range_checker = get_inventory_range_checker(inventory);

        inventory.next_air::<CastFAir>()?;
        let castf = CastFChipHip::new(range_checker, timestamp_max_bits);
        inventory.add_executor_chip(castf);

        Ok(())
    }
}
