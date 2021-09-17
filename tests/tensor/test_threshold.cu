/*******************************************************************************
 *  FILENAME:      test_threshold.cu
 *
 *  AUTHORS:       Hou Yue    START DATE: Monday August 9th 2021
 *
 *  LAST MODIFIED: Monday, August 9th 2021, 2:26:20 pm
 *
 *  CONTACT:       yue.hou@smartmore.com
 *******************************************************************************/

#define CATCH_CONFIG_MAIN

#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <stdio.h>
#include <utils.h>

#include <catch2/catch.hpp>

template <typename DataType, smartmore::cudaop::ThreshType thresh_type>
inline float GetMaxDiff(std::vector<DataType> input_data, std::vector<DataType> actual, std::vector<DataType> expect,
                        double thresh, double maxval, int len) {
    float maxDiff = 0;
    switch (thresh_type) {
        case smartmore::cudaop::ThreshType::kThresh_Binary:
            for (int i = 0; i < len; i++) {
                if (input_data[i] > thresh)
                    expect[i] = maxval;
                else
                    expect[i] = 0;
                maxDiff = fabs(actual[i] - expect[i]) > maxDiff ? fabs(actual[i] - expect[i]) : maxDiff;
            }
            break;
        case smartmore::cudaop::ThreshType::kThresh_Binary_INV:
            for (int i = 0; i < len; i++) {
                if (input_data[i] > thresh)
                    expect[i] = 0;
                else
                    expect[i] = maxval;
                maxDiff = fabs(actual[i] - expect[i]) > maxDiff ? fabs(actual[i] - expect[i]) : maxDiff;
            }
            break;
        case smartmore::cudaop::ThreshType::kThresh_Trunc:
            for (int i = 0; i < len; i++) {
                if (input_data[i] > thresh)
                    expect[i] = thresh;
                else
                    expect[i] = input_data[i];
                maxDiff = fabs(actual[i] - expect[i]) > maxDiff ? fabs(actual[i] - expect[i]) : maxDiff;
            }
            break;
        case smartmore::cudaop::ThreshType::kThresh_ToZero:
            for (int i = 0; i < len; i++) {
                if (input_data[i] > thresh)
                    expect[i] = input_data[i];
                else
                    expect[i] = 0;
                maxDiff = fabs(actual[i] - expect[i]) > maxDiff ? fabs(actual[i] - expect[i]) : maxDiff;
            }
            break;
        case smartmore::cudaop::ThreshType::kThresh_ToZero_INV:
            for (int i = 0; i < len; i++) {
                if (input_data[i] > thresh)
                    expect[i] = 0;
                else
                    expect[i] = input_data[i];
                maxDiff = fabs(actual[i] - expect[i]) > maxDiff ? fabs(actual[i] - expect[i]) : maxDiff;
            }
            break;
        default:
            break;
    }

    return maxDiff;
}

TEST_CASE("Threshold INT8", "[threshold]") {
    const unsigned int len = 100;
    std::vector<unsigned char> input_data(len);
    smartmore::RandomInt8Vector(input_data);

    void *src = nullptr, *dst = nullptr;
    std::vector<unsigned char> expect(len), actual(len);
    CUDA_CHECK(cudaMalloc(&src, len * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&dst, len * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(src, input_data.data(), input_data.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    double thresh = 5.0f;
    double maxval = 10.0f;

    float maxDiff = 0;
    SECTION("Threshold type is kThresh_Binary") {
        Threshold<DataType::kInt8, ThreshType::kThresh_Binary>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        maxDiff =
            GetMaxDiff<unsigned char, ThreshType::kThresh_Binary>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_Binary_INV") {
        Threshold<DataType::kInt8, ThreshType::kThresh_Binary_INV>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        maxDiff =
            GetMaxDiff<unsigned char, ThreshType::kThresh_Binary_INV>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_Trunc") {
        Threshold<DataType::kInt8, ThreshType::kThresh_Trunc>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<unsigned char, ThreshType::kThresh_Trunc>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_ToZero") {
        Threshold<DataType::kInt8, ThreshType::kThresh_ToZero>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        maxDiff =
            GetMaxDiff<unsigned char, ThreshType::kThresh_ToZero>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_ToZero_INV") {
        Threshold<DataType::kInt8, ThreshType::kThresh_ToZero_INV>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        maxDiff =
            GetMaxDiff<unsigned char, ThreshType::kThresh_ToZero_INV>(input_data, actual, expect, thresh, maxval, len);
    }

    REQUIRE(maxDiff < 1);
    CUDA_CHECK_AND_FREE(src);
    CUDA_CHECK_AND_FREE(dst);
}

TEST_CASE("Threshold kFloat32", "[threshold]") {
    const unsigned int len = 100;
    std::vector<float> input_data(len);
    smartmore::RandomFloatVector(input_data);

    void *src = nullptr, *dst = nullptr;
    std::vector<float> expect(len), actual(len);
    CUDA_CHECK(cudaMalloc(&src, len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, len * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    double thresh = 0.6f;
    double maxval = 0.8f;

    float maxDiff = 0;
    SECTION("Threshold type is kThresh_Binary") {
        Threshold<DataType::kFloat32, ThreshType::kThresh_Binary>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_Binary>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_Binary_INV") {
        Threshold<DataType::kFloat32, ThreshType::kThresh_Binary_INV>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_Binary_INV>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_Trunc") {
        Threshold<DataType::kFloat32, ThreshType::kThresh_Trunc>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_Trunc>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_ToZero") {
        Threshold<DataType::kFloat32, ThreshType::kThresh_ToZero>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_ToZero>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_ToZero_INV") {
        Threshold<DataType::kFloat32, ThreshType::kThresh_ToZero_INV>(src, dst, len, thresh, maxval);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_ToZero_INV>(input_data, actual, expect, thresh, maxval, len);
    }

    REQUIRE(maxDiff < 0.0001);
    CUDA_CHECK_AND_FREE(src);
    CUDA_CHECK_AND_FREE(dst);
}

#if __CUDA_ARCH__ >= 700
TEST_CASE("Threshold kHalf", "[threshold]") {
    const unsigned int len = 100;
    std::vector<float> input_data(len);
    smartmore::RandomFloatVector(input_data);

    void *src_float = nullptr, *dst_float = nullptr;
    void *src_half = nullptr, *dst_half = nullptr;
    std::vector<float> expect(len), actual(len);

    CUDA_CHECK(cudaMalloc(&src_float, len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&src_half, len * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&dst_float, len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_half, len * sizeof(float) / 2));
    CUDA_CHECK(cudaMemcpy(src_float, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src_float, src_half, input_data.size());
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(dst_float, dst_half, input_data.size());
    double thresh = 0.6f;
    double maxval = 0.8f;

    float maxDiff = 0;
    SECTION("Threshold type is kThresh_Binary") {
        Threshold<DataType::kHalf, ThreshType::kThresh_Binary>(src_half, dst_half, len, thresh, maxval);

        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_float, input_data.size());
        CUDA_CHECK(cudaMemcpy(&actual[0], dst_float, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_Binary>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_Binary_INV") {
        Threshold<DataType::kHalf, ThreshType::kThresh_Binary_INV>(src_half, dst_half, len, thresh, maxval);

        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_float, input_data.size());
        CUDA_CHECK(cudaMemcpy(&actual[0], dst_float, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_Binary_INV>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_Trunc") {
        Threshold<DataType::kHalf, ThreshType::kThresh_Trunc>(src_half, dst_half, len, thresh, maxval);

        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_float, input_data.size());
        CUDA_CHECK(cudaMemcpy(&actual[0], dst_float, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_Trunc>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_ToZero") {
        Threshold<DataType::kHalf, ThreshType::kThresh_ToZero>(src_half, dst_half, len, thresh, maxval);

        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_float, input_data.size());
        CUDA_CHECK(cudaMemcpy(&actual[0], dst_float, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_ToZero>(input_data, actual, expect, thresh, maxval, len);
    }
    SECTION("Threshold type is kThresh_ToZero_INV") {
        Threshold<DataType::kHalf, ThreshType::kThresh_ToZero_INV>(src_half, dst_half, len, thresh, maxval);

        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_float, input_data.size());
        CUDA_CHECK(cudaMemcpy(&actual[0], dst_float, len * sizeof(float), cudaMemcpyDeviceToHost));

        maxDiff = GetMaxDiff<float, ThreshType::kThresh_ToZero_INV>(input_data, actual, expect, thresh, maxval, len);
    }

    REQUIRE(maxDiff < 0.01);
    CUDA_CHECK_AND_FREE(src_float);
    CUDA_CHECK_AND_FREE(src_half);
    CUDA_CHECK_AND_FREE(dst_float);
    CUDA_CHECK_AND_FREE(dst_half);
}
#endif
