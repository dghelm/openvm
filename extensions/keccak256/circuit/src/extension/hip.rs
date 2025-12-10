use openvm_circuit::{
    arch::{AirInventory, DenseRecordArena, VmBuilder, VmChipComplex},
    system::hip::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemHipBuilder,
        },
        SystemChipInventoryHIP,
    },
};
use openvm_hip_backend::{engine::HipBabyBearPoseidon2Engine, prover_backend::HipBackend};
use openvm_rv32im_circuit::Rv32ImHipProverExt;
use openvm_stark_sdk::{config::baby_bear_poseidon2::BabyBearPoseidon2Config, engine::StarkEngine};

use super::*;
use crate::{air::KeccakVmAir, hip::Keccak256ChipHip};

pub struct Keccak256HipProverExt;

// This implementation is specific to HipBackend because the lookup chips
// (VariableRangeCheckerChipHIP, BitwiseOperationLookupChipHIP) are specific to HipBackend.
impl VmProverExtension<HipBabyBearPoseidon2Engine, DenseRecordArena, Keccak256>
    for Keccak256HipProverExt
{
    fn extend_prover(
        &self,
        _: &Keccak256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, HipBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<KeccakVmAir>()?;
        let keccak = Keccak256ChipHip::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits as u32,
            timestamp_max_bits as u32,
        );
        inventory.add_executor_chip(keccak);

        Ok(())
    }
}

pub struct Keccak256Rv32HipBuilder;

type E = HipBabyBearPoseidon2Engine;

impl VmBuilder<E> for Keccak256Rv32HipBuilder {
    type VmConfig = Keccak256Rv32Config;
    type SystemChipInventory = SystemChipInventoryHIP;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Keccak256Rv32Config,
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
            &Keccak256HipProverExt,
            &config.keccak,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
