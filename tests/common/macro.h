#include <cuda_runtime.h>

#include <stdexcept>

#define CUDA_CHECK(status)                                                                                           \
    do {                                                                                                             \
        auto ret = (status);                                                                                         \
        if (ret != 0) {                                                                                              \
            throw std::runtime_error("cuda failure: " + std::to_string(ret) + " (" + cudaGetErrorString(ret) + ")" + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));                            \
        }                                                                                                            \
    } while (0)

#define CUDA_CHECK_AND_FREE(device_ptr) \
    do {                                \
        if (device_ptr) {               \
            cudaFree(device_ptr);       \
        }                               \
    } while (0)
