/******************************************************************************
 * FILENAME:      test_concat.cu
 *
 * AUTHORS:       Yutong Huang
 *
 * LAST MODIFIED: Wed 26 May 2021 10:01:01 AM CST
 *
 * CONTACT:       yutong.huang@smartmore.com
 ******************************************************************************/

#define CATCH_CONFIG_MAIN
#include <cudaop/cudaop.h>

#include <catch2/catch.hpp>

#include "macro.h"

namespace {
constexpr float epsilon = 0.001f;
}

TEST_CASE("concat", "concat") {
    std::vector<float> lhs{0.01097937, 0.7205631,  0.47166034, 0.61041445, 0.73491615, 0.24883934,
                           0.27167893, 0.19414721, 0.56568044, 0.03571895, 0.82428875, 0.95509511,
                           0.67675268, 0.49172655, 0.75043525, 0.07917435, 0.82603228, 0.09668733,
                           0.11233177, 0.95286767, 0.34732047, 0.0754146,  0.73657873, 0.85475406};

    std::vector<float> rhs{
        0.6812822,  0.6119252,  0.75371998, 0.40783232, 0.82288113, 0.30909813, 0.06448972, 0.82820103, 0.60603602,
        0.05561141, 0.0744732,  0.47944858, 0.05131854, 0.51749473, 0.58961016, 0.17841821, 0.81719229, 0.89304113,
        0.31837209, 0.84057261, 0.65806954, 0.55442602, 0.42908242, 0.68752797, 0.5409535,  0.77161711, 0.14461309,
        0.21628952, 0.79668969, 0.04142308, 0.1981673,  0.92437707, 0.89660727, 0.83684494, 0.49191388, 0.20781362};

    std::vector<float> expected{
        0.01097937, 0.7205631,  0.47166034, 0.61041445, 0.73491615, 0.24883934, 0.27167893, 0.19414721, 0.56568044,
        0.03571895, 0.82428875, 0.95509511, 0.67675268, 0.49172655, 0.75043525, 0.07917435, 0.82603228, 0.09668733,
        0.11233177, 0.95286767, 0.34732047, 0.0754146,  0.73657873, 0.85475406, 0.6812822,  0.6119252,  0.75371998,
        0.40783232, 0.82288113, 0.30909813, 0.06448972, 0.82820103, 0.60603602, 0.05561141, 0.0744732,  0.47944858,
        0.05131854, 0.51749473, 0.58961016, 0.17841821, 0.81719229, 0.89304113, 0.31837209, 0.84057261, 0.65806954,
        0.55442602, 0.42908242, 0.68752797, 0.5409535,  0.77161711, 0.14461309, 0.21628952, 0.79668969, 0.04142308,
        0.1981673,  0.92437707, 0.89660727, 0.83684494, 0.49191388, 0.20781362};

    void *a = nullptr;
    void *b = nullptr;
    void *out = nullptr;

    CUDA_CHECK(cudaMalloc(&a, sizeof(float) * lhs.size()));
    CUDA_CHECK(cudaMalloc(&b, sizeof(float) * rhs.size()));
    CUDA_CHECK(cudaMalloc(&out, sizeof(float) * expected.size()));

    CUDA_CHECK(cudaMemcpy(a, lhs.data(), sizeof(float) * lhs.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, rhs.data(), sizeof(float) * rhs.size(), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    Concat<float>((float *)a, (float *)b, (float *)out, 1, {1, 2, 3, 4}, {1, 3, 3, 4});

    std::vector<float> got(expected.size());

    CUDA_CHECK(cudaMemcpy(got.data(), out, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));

    float diff_sum = 0.0, max_diff = 0.0;
    for (int i = 0; i < expected.size(); i++) {
        float diff = std::abs(got[i] - expected[i]);
        diff_sum += diff;
        max_diff = diff > max_diff ? diff : max_diff;
    }

    REQUIRE(max_diff <= epsilon);
}
