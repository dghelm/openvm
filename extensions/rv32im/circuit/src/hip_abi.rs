#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_hip_backend::{chip::UInt2, prelude::F};
use openvm_hip_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::HipError,
};

pub mod auipc_hip {
    use super::*;

    extern "C" {
        fn _auipc_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        HipError::from_result_i32(_auipc_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod hintstore_hip {
    use super::*;

    // This is the info needed by each row to do parallel tracegen
    // Matches the layout in hintstore/cuda.rs
    #[repr(C)]
    pub struct OffsetInfo {
        pub record_offset: u32,
        pub local_idx: u32,
    }

    extern "C" {
        pub fn _hintstore_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const u8,
            rows_used: usize,
            d_record_offsets: *const OffsetInfo,
            pointer_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        rows_used: usize,
        d_record_offsets: &DeviceBuffer<OffsetInfo>,
        pointer_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        HipError::from_result_i32(_hintstore_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.as_ptr(),
            rows_used,
            d_record_offsets.as_ptr(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            timestamp_max_bits,
        ))
    }
}

pub mod jalr_hip {
    use super::*;

    extern "C" {
        fn _jalr_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        assert!(height.is_power_of_two() || height == 0);
        HipError::from_result_i32(_jalr_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod less_than_hip {
    use super::*;

    extern "C" {
        fn _rv32_less_than_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        HipError::from_result_i32(_rv32_less_than_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod mul_hip {

    use super::*;

    extern "C" {
        fn _mul_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_range_tuple: *mut u32,
            range_tuple_sizes: UInt2,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        let width = d_trace.len() / height;
        HipError::from_result_i32(_mul_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range.as_mut_ptr() as *mut u32,
            range_bins,
            d_range_tuple.as_ptr() as *mut u32,
            range_tuple_sizes,
            timestamp_max_bits,
        ))
    }
}

pub mod divrem_hip {
    use super::*;

    extern "C" {
        pub fn _rv32_div_rem_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        HipError::from_result_i32(_rv32_div_rem_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            d_range_tuple_checker.as_mut_ptr() as *mut u32,
            range_tuple_checker_sizes,
            timestamp_max_bits,
        ))
    }
}

pub mod shift_hip {
    use super::*;

    extern "C" {
        fn _rv32_shift_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        HipError::from_result_i32(_rv32_shift_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod alu_hip {
    use super::*;
    extern "C" {
        fn _alu_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        let width = d_trace.len() / height;
        HipError::from_result_i32(_alu_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            range_bins,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            timestamp_max_bits,
        ))
    }
}

pub mod loadstore_hip {
    use super::*;

    extern "C" {
        pub fn _rv32_load_store_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        HipError::from_result_i32(_rv32_load_store_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod load_sign_extend_hip {
    use super::*;

    extern "C" {
        pub fn _rv32_load_sign_extend_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        HipError::from_result_i32(_rv32_load_sign_extend_tracegen(
            d_trace.as_mut_ptr(),
            height,
            width,
            d_records.view(),
            pointer_max_bits,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod jal_lui_hip {
    use super::*;

    extern "C" {
        fn _jal_lui_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        assert!(height.is_power_of_two() || height == 0);
        HipError::from_result_i32(_jal_lui_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod beq_hip {
    use super::*;

    extern "C" {
        fn _beq_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        assert!(height.is_power_of_two() || height == 0);
        HipError::from_result_i32(_beq_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod branch_lt_hip {
    use super::*;

    extern "C" {
        fn _blt_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            rc_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        assert!(height.is_power_of_two() || height == 0);
        HipError::from_result_i32(_blt_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
            timestamp_max_bits,
        ))
    }
}

pub mod mulh_hip {
    use super::*;

    extern "C" {
        fn _mulh_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
    ) -> Result<(), HipError> {
        assert!(height.is_power_of_two() || height == 0);
        HipError::from_result_i32(_mulh_tracegen(
            d_trace.as_mut_ptr(),
            height,
            d_trace.len() / height,
            d_records.view(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits as u32,
            d_range_tuple_checker.as_mut_ptr() as *mut u32,
            range_tuple_checker_sizes,
            timestamp_max_bits,
        ))
    }
}
