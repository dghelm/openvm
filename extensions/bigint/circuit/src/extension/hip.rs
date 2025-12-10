use openvm_circuit::{
    arch::{AirInventory, DenseRecordArena, VmBuilder, VmChipComplex},
    system::hip::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemHipBuilder,
        },
        SystemChipInventoryHIP,
    },
};
use openvm_circuit_primitives::range_tuple::RangeTupleCheckerChipHIP;
use openvm_hip_backend::{engine::HipBabyBearPoseidon2Engine, prover_backend::HipBackend};
use openvm_rv32im_circuit::Rv32ImHipProverExt;
use openvm_stark_backend::engine::StarkEngine;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use super::*;
use crate::hip::{
    BaseAlu256ChipHip, BranchEqual256ChipHip, BranchLessThan256ChipHip, LessThan256ChipHip,
    Multiplication256ChipHip, Shift256ChipHip,
};

pub struct Int256HipProverExt;

// This implementation is specific to HipBackend because the lookup chips
// (VariableRangeCheckerChipHIP, BitwiseOperationLookupChipHIP) are specific to HipBackend.
impl VmProverExtension<HipBabyBearPoseidon2Engine, DenseRecordArena, Int256>
    for Int256HipProverExt
{
    fn extend_prover(
        &self,
        extension: &Int256,
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

        inventory.next_air::<Rv32BaseAlu256Air>()?;
        let base_alu = BaseAlu256ChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<Rv32LessThan256Air>()?;
        let lt = LessThan256ChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv32BranchEqual256Air>()?;
        let beq = BranchEqual256ChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv32BranchLessThan256Air>()?;
        let blt = BranchLessThan256ChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv32Multiplication256Air>()?;
        let mult = Multiplication256ChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv32Shift256Air>()?;
        let shift = Shift256ChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift);

        Ok(())
    }
}

#[derive(Clone)]
pub struct Int256Rv32HipBuilder;

type E = HipBabyBearPoseidon2Engine;

impl VmBuilder<E> for Int256Rv32HipBuilder {
    type VmConfig = Int256Rv32Config;
    type SystemChipInventory = SystemChipInventoryHIP;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Int256Rv32Config,
        circuit: AirInventory<<E as StarkEngine>::SC>,
    ) -> Result<
        VmChipComplex<
            <E as StarkEngine>::SC,
            Self::RecordArena,
            <E as StarkEngine>::PB,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemHipBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImHipProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImHipProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImHipProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Int256HipProverExt,
            &config.bigint,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
