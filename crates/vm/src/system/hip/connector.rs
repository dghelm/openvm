use openvm_circuit::system::connector::VmConnectorChip;
use openvm_hip_backend::{chip::HybridChip, prelude::F};

pub type VmConnectorChipHIP = HybridChip<(), VmConnectorChip<F>>;
