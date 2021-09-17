/*******************************************************************************
 *  FILENAME:      atomic.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Friday May 14th 2021
 *
 *  LAST MODIFIED: Thursday, May 20th 2021, 5:11:28 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *
 *  REFERENCE:     https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *                 https://stackoverflow.com/questions/5447570/cuda-atomic-operations-on-unsigned-chars
 *                 https://github.com/mindspore-ai/akg/blob/master/src/akg_reduce/operators/reduce_operators.cuh
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_GENERIC_ATOMIC_H__
#define __SMARTMORE_CUDAOP_GENERIC_ATOMIC_H__

#include <cudaop/generic/halfoperators.h>

namespace smartmore {
namespace cudaop {
static inline __device__ float atomicMin(float *addr, float value) {
    float old = *addr, assumed;
    if (old <= value) return old;
    do {
        assumed = old;
        old = atomicCAS((unsigned int *)addr, __float_as_int(assumed), __float_as_int(value));
    } while (old != assumed);

    return old;
}

static inline __device__ float atomicMax(float *addr, float value) {
    float old = *addr, assumed;
    if (old >= value) return old;
    do {
        assumed = old;
        old = atomicCAS((unsigned int *)addr, __float_as_int(assumed), __float_as_int(value));
    } while (old != assumed);

    return old;
}

#if __CUDA_ARCH__ >= 700
__device__ half atomicMin(half *addr, half val) {
    unsigned short int *const addr_as_usi = (unsigned short int *)addr;
    unsigned short int old = *addr_as_usi, assumed;

    if (__ushort_as_half(old) < val) return __ushort_as_half(old);

    do {
        assumed = old;
        if (__ushort_as_half(assumed) < val) break;
        old = atomicCAS(addr_as_usi, assumed, __half_as_ushort(val));
    } while (assumed != old);

    return __ushort_as_half(old);
}

__device__ half atomicMax(half *addr, half val) {
    unsigned short int *const addr_as_usi = (unsigned short int *)addr;
    unsigned short int old = *addr_as_usi, assumed;

    if (__ushort_as_half(old) > val) return __ushort_as_half(old);

    do {
        assumed = old;
        if (__ushort_as_half(assumed) > val) break;
        old = atomicCAS(addr_as_usi, assumed, __half_as_ushort(val));
    } while (assumed != old);

    return __ushort_as_half(old);
}
#endif

__device__ static inline unsigned char atomicMin(unsigned char *address, unsigned char val) {
    size_t long_address_modulo = (size_t)address & 3;
    auto *base_address = (unsigned int *)((unsigned char *)address - long_address_modulo);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[long_address_modulo];
    unsigned int long_old, long_assumed, long_val, replacement;
    long_old = *base_address;
    if (long_old <= val) return long_old;

    do {
        long_assumed = long_old;
        long_val = val;
        replacement = __byte_perm(long_old, long_val, selector);
        long_old = atomicCAS(base_address, long_assumed, replacement);
    } while (long_old != long_assumed);
    return __byte_perm(long_old, 0, long_address_modulo);
}

__device__ static inline unsigned char atomicMax(unsigned char *address, unsigned char val) {
    size_t long_address_modulo = (size_t)address & 3;
    auto *base_address = (unsigned int *)((unsigned char *)address - long_address_modulo);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[long_address_modulo];
    unsigned int long_old, long_assumed, long_val, replacement;
    long_old = *base_address;
    if (long_old >= val) return long_old;

    do {
        long_assumed = long_old;
        long_val = val;
        replacement = __byte_perm(long_old, long_val, selector);
        long_old = atomicCAS(base_address, long_assumed, replacement);
    } while (long_old != long_assumed);
    return __byte_perm(long_old, 0, long_address_modulo);
}
}  // namespace cudaop
}  // namespace smartmore

#endif