use std::sync::Arc;

use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::hip::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_circuit_primitives::range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerChipHIP};
use openvm_hip_backend::{engine::HipBabyBearPoseidon2Engine, prover_backend::HipBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    Rv32AuipcAir, Rv32AuipcChipHip, Rv32BaseAluAir, Rv32BaseAluChipHip, Rv32BranchEqualAir,
    Rv32BranchEqualChipHip, Rv32BranchLessThanAir, Rv32BranchLessThanChipHip, Rv32DivRemAir,
    Rv32DivRemChipHip, Rv32HintStoreAir, Rv32HintStoreChipHip, Rv32I, Rv32Io, Rv32JalLuiAir,
    Rv32JalLuiChipHip, Rv32JalrAir, Rv32JalrChipHip, Rv32LessThanAir, Rv32LessThanChipHip,
    Rv32LoadSignExtendAir, Rv32LoadSignExtendChipHip, Rv32LoadStoreAir, Rv32LoadStoreChipHip,
    Rv32M, Rv32MulHAir, Rv32MulHChipHip, Rv32MultiplicationAir, Rv32MultiplicationChipHip,
    Rv32ShiftAir, Rv32ShiftChipHip,
};

pub struct Rv32ImHipProverExt;

// This implementation is specific to HipBackend because the lookup chips
// (VariableRangeCheckerChipHIP, BitwiseOperationLookupChipHIP) are specific to HipBackend.
impl VmProverExtension<HipBabyBearPoseidon2Engine, DenseRecordArena, Rv32I> for Rv32ImHipProverExt {
    fn extend_prover(
        &self,
        _: &Rv32I,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, HipBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32BaseAluAir>()?;
        let base_alu = Rv32BaseAluChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<Rv32LessThanAir>()?;
        let lt = Rv32LessThanChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv32ShiftAir>()?;
        let shift = Rv32ShiftChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift);

        inventory.next_air::<Rv32LoadStoreAir>()?;
        let load_store_chip =
            Rv32LoadStoreChipHip::new(range_checker.clone(), pointer_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(load_store_chip);

        inventory.next_air::<Rv32LoadSignExtendAir>()?;
        let load_sign_extend = Rv32LoadSignExtendChipHip::new(
            range_checker.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<Rv32BranchEqualAir>()?;
        let beq = Rv32BranchEqualChipHip::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv32BranchLessThanAir>()?;
        let blt = Rv32BranchLessThanChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv32JalLuiAir>()?;
        let jal_lui = Rv32JalLuiChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv32JalrAir>()?;
        let jalr = Rv32JalrChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv32AuipcAir>()?;
        let auipc = Rv32AuipcChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(auipc);

        Ok(())
    }
}

// This implementation is specific to HipBackend because the lookup chips
// (VariableRangeCheckerChipHIP, BitwiseOperationLookupChipHIP) are specific to HipBackend.
impl VmProverExtension<HipBabyBearPoseidon2Engine, DenseRecordArena, Rv32M> for Rv32ImHipProverExt {
    fn extend_prover(
        &self,
        extension: &Rv32M,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, HipBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipHIP<2>>>()
                .find(|c| {
                    c.sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                inventory.next_air::<RangeTupleCheckerAir<2>>()?;
                let chip = Arc::new(RangeTupleCheckerChipHIP::new(
                    extension.range_tuple_checker_sizes,
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32MultiplicationAir>()?;
        let mult = Rv32MultiplicationChipHip::new(
            range_checker.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv32MulHAir>()?;
        let mul_h = Rv32MulHChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv32DivRemAir>()?;
        let div_rem = Rv32DivRemChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(div_rem);

        Ok(())
    }
}

// This implementation is specific to HipBackend because the lookup chips
// (VariableRangeCheckerChipHIP, BitwiseOperationLookupChipHIP) are specific to HipBackend.
impl VmProverExtension<HipBabyBearPoseidon2Engine, DenseRecordArena, Rv32Io>
    for Rv32ImHipProverExt
{
    fn extend_prover(
        &self,
        _: &Rv32Io,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, HipBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        inventory.next_air::<Rv32HintStoreAir>()?;
        let hint_store = Rv32HintStoreChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}
