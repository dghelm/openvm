use openvm_circuit_primitives::Chip;
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::p3_matrix::Matrix;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};

use crate::poseidon2::DeferralPoseidon2Chip;

#[test]
fn deferral_poseidon2_empty_trace_is_omitted() {
    let chip = DeferralPoseidon2Chip::<BabyBear>::new(Default::default());

    let ctx = <DeferralPoseidon2Chip<BabyBear> as Chip<
        (),
        CpuBackend<BabyBearPoseidon2Config>,
    >>::generate_proving_ctx(&chip, ());

    assert_eq!(
        ctx.common_main.height(),
        0,
        "unused deferral poseidon2 chip should return an empty trace",
    );
}
