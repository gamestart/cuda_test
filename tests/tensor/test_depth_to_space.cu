#include <array>
#include <iostream>
#include <ostream>
#define CATCH_CONFIG_MAIN
#include <cuda_runtime_api.h>
#include <cudaop/cudaop.h>
#include <sys/types.h>

#include <catch2/catch.hpp>
#include <chrono>

TEST_CASE("DCR Mode", "[DepthToSpace]") {
    // shape: {1, 8, 2, 1}
    float arr[16] = {1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16};
    float expected[16] = {1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16};
    void *src = nullptr;
    void *dst = nullptr;
    cudaMalloc(&src, sizeof(float) * 16);
    cudaMalloc(&dst, sizeof(float) * 16);
    cudaMemcpy(src, arr, sizeof(float) * 16, cudaMemcpyHostToDevice);
    smartmore::cudaop::DepthToSpace<float, true, 2, 1, 8, 2, 1>(static_cast<float *>(src), static_cast<float *>(dst));
    cudaMemcpy(arr, dst, sizeof(float) * 16, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) {
        REQUIRE(arr[i] == expected[i]);
    }
}

TEST_CASE("CRD Mode", "[DepthToSpace]") {
    // shape: {1, 8, 2, 1}
    float arr[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float expected[16] = {1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16};
    void *src = nullptr;
    void *dst = nullptr;
    cudaMalloc(&src, sizeof(float) * 16);
    cudaMalloc(&dst, sizeof(float) * 16);
    cudaMemcpy(src, arr, sizeof(float) * 16, cudaMemcpyHostToDevice);
    smartmore::cudaop::DepthToSpace<float, false, 2, 1, 8, 2, 1>(static_cast<float *>(src), static_cast<float *>(dst));
    cudaMemcpy(arr, dst, sizeof(float) * 16, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) {
        REQUIRE(arr[i] == expected[i]);
    }
}

TEST_CASE("Profiling DCR", "[DepthToSpace]") {
    // shape: {1, 128, 270, 480}
    void *src = nullptr;
    void *dst = nullptr;
    cudaMalloc(&src, sizeof(float) * 1 * 128 * 270 * 480);
    cudaMalloc(&dst, sizeof(float) * 1 * 128 * 270 * 480);
    for (int i = 0; i < 2; i++)
        smartmore::cudaop::DepthToSpace<float, true, 4, 1, 128, 270, 480>(static_cast<float *>(src),
                                                                          static_cast<float *>(dst));
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; i++)
        smartmore::cudaop::DepthToSpace<float, true, 4, 1, 128, 270, 480>(static_cast<float *>(src),
                                                                          static_cast<float *>(dst));
    auto end = std::chrono::high_resolution_clock::now();
    typedef std::chrono::duration<double, std::ratio<1, 1000>> milliSecond;
    milliSecond duration_ms = std::chrono::duration_cast<milliSecond>(end - start);
    auto ms = duration_ms.count() / 200;
    std::string time_str;
    if (ms < 1.0) {
        time_str = std::to_string(ms * 1000.0) + "μs";
    }
    if (ms > 1000.0) {
        time_str = std::to_string(ms / 1000.0) + "s";
    }
    time_str = std::to_string(ms) + "ms";
    std::cout << "DCR Duration avg over 200 iterations: " << time_str << std::endl;
    cudaFree(src);
    cudaFree(dst);
}

TEST_CASE("Profiling CRD", "[DepthToSpace]") {
    // shape: {1, 128, 270, 480}
    void *src = nullptr;
    void *dst = nullptr;
    cudaMalloc(&src, sizeof(float) * 1 * 128 * 270 * 480);
    cudaMalloc(&dst, sizeof(float) * 1 * 128 * 270 * 480);
    for (int i = 0; i < 2; i++)
        smartmore::cudaop::DepthToSpace<float, false, 4, 1, 128, 270, 480>(static_cast<float *>(src),
                                                                           static_cast<float *>(dst));
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; i++)
        smartmore::cudaop::DepthToSpace<float, false, 4, 1, 128, 270, 480>(static_cast<float *>(src),
                                                                           static_cast<float *>(dst));
    auto end = std::chrono::high_resolution_clock::now();
    typedef std::chrono::duration<double, std::ratio<1, 1000>> milliSecond;
    milliSecond duration_ms = std::chrono::duration_cast<milliSecond>(end - start);
    auto ms = duration_ms.count() / 200;
    std::string time_str;
    if (ms < 1.0) {
        time_str = std::to_string(ms * 1000.0) + "μs";
    }
    if (ms > 1000.0) {
        time_str = std::to_string(ms / 1000.0) + "s";
    }
    time_str = std::to_string(ms) + "ms";
    std::cout << "CRD Duration avg over 200 iterations: " << time_str << std::endl;
    cudaFree(src);
    cudaFree(dst);
}
