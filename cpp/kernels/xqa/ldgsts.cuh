/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "cuda_hint.cuh"
#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif
#include "barriers.cuh"

namespace ldgsts
{
// @fixme: prefetch makes it slower on sm_86. Try on other platforms.
template <uint32_t size>
__device__ inline void copyAsync(
    void* dst, void const* src, uint32_t srcSize = size) // srcSize == 0 means filling with zeros.
{
    static_assert(size == 4 || size == 8 || size == 16);
#if __CUDA_ARCH__ >= 800
    if constexpr (size == 16)
    {
        // asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::
        // "l"(__cvta_generic_to_shared(dst)), "l"(src), "r"(srcSize));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"l"(__cvta_generic_to_shared(dst)), "l"(src),
            "r"(srcSize));
    }
    else
    {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"l"(__cvta_generic_to_shared(dst)), "l"(src),
            "n"(size), "r"(srcSize));
    }
#else
    // Fallback path for pre-SM80: synchronous copy into shared memory.
    unsigned char* d = reinterpret_cast<unsigned char*>(dst);
    unsigned char const* s = reinterpret_cast<unsigned char const*>(src);
#pragma unroll
    for (uint32_t i = 0; i < size; ++i)
    {
        d[i] = (i < srcSize) ? s[i] : static_cast<unsigned char>(0);
    }
#endif
}

__device__ inline void commitGroup()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

// wait until only targetNbInFlightGroups groups are still in-flight.
template <uint32_t targetNbInFlightGroups>
__device__ inline void waitGroup()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(targetNbInFlightGroups));
#endif
}

// noInc = false:  increase expected arrive count, in additional to increasing arrive count
// noInc = true: increases arrive count but does not modify expected arrive count
__device__ inline void barArrive(CtaBarrier& bar, bool noInc = false)
{
#if __CUDA_ARCH__ >= 800
    if (noInc)
    {
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"l"(__cvta_generic_to_shared(&bar)));
    }
    else
    {
        asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];\n" ::"l"(__cvta_generic_to_shared(&bar)));
    }
#else
    (void) noInc;
    bar.arrive();
#endif
}

} // namespace ldgsts
