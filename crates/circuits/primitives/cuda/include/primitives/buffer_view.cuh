#pragma once

#include <cstdint>
#include <cassert>

// HIP/CUDA host+device inline compatibility
#ifndef HOST_DEVICE_INLINE
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define HOST_DEVICE_INLINE __device__ __host__ __attribute__((always_inline)) inline
#else
#define HOST_DEVICE_INLINE __device__ __host__ __forceinline__
#endif
#endif

template <typename T>
struct DeviceBufferConstView {
    T const* ptr;
    size_t size;

    HOST_DEVICE_INLINE T const* begin() const {
        return ptr;
    }

    HOST_DEVICE_INLINE T const* end() const {
        return ptr + len();
    }

    HOST_DEVICE_INLINE T const& operator [](size_t idx) const {
        assert(idx < len());
        return ptr[idx];
    }

    HOST_DEVICE_INLINE size_t len() const {
        return size / sizeof(T);
    }
};

struct DeviceRawBufferConstView {
    uintptr_t ptr;
    size_t size;

    template <typename T> HOST_DEVICE_INLINE DeviceBufferConstView<T> as_typed() const {
        assert(size % sizeof(T) == 0);
        return {
            reinterpret_cast<T const*>(ptr),
            size
        };
    }
};
