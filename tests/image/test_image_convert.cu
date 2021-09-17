/*******************************************************************************
 *  FILENAME:      test_image_convert.cpp
 *
 *  AUTHORS:       Lin Qinghuang    START DATE: Monday March 22nd 2021
 *
 *  LAST MODIFIED: Monday, September 6th 2021, 11:53:13 am
 *
 *  CONTACT:       qinghuang.lin@smartmore.com
 *******************************************************************************/

#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("ImageConvert YUV_NV12_To_BGR_HWC int8", "[image_convert]") {
    // NV12 to RGB_HWC Int8ToInt8
    void *src_device, *dst_device;
    // 4 * 6 image, use random number
    std::vector<unsigned char> input_nv12{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 4, width = 6;

    cv::Mat src_nv12_raw(1.5 * height, width, CV_8UC1, input_nv12.data());

    CUDA_CHECK(cudaMalloc(&src_device, 1.5 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_device, 3 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(src_device, src_nv12_raw.data, 1.5 * height * width * sizeof(char), cudaMemcpyHostToDevice));
    // convert with cuda
    smartmore::cudaop::ImageConvert<smartmore::cudaop::ImageType::kYUV_NV12, smartmore::cudaop::DataType::kInt8,
                                    smartmore::cudaop::ImageType::kBGR_HWC, smartmore::cudaop::DataType::kInt8,
                                    smartmore::cudaop::YUVFormula::kBT601>(src_device, dst_device, height, width);
    cv::Mat result_cuda(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_device, 3 * height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // convert with opencv
    cv::Mat result_opencv(height, width, CV_8UC3);
    cv::cvtColor(src_nv12_raw, result_opencv, cv::ColorConversionCodes::COLOR_YUV2BGR_NV12);

    // compare cuda with opencv
    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_cuda.data[i]);
        int expect = static_cast<int>(result_opencv.data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_device);
    CUDA_CHECK_AND_FREE(dst_device);
}

TEST_CASE("ImageConvert YUV_NV12_To_BGR_HWC half", "[image_convert]") {
    // NV12 to RGB_HWC Int8ToInt8
    void *src_float, *src_half, *dst_half, *dst_float;
    // 4 * 6 image, use random number
    int height = 4, width = 6;
    std::vector<float> input_nv12_float(height * width * 3 / 2);
    smartmore::RandomFloatVector(input_nv12_float);
    cv::Mat src_nv12_raw(1.5 * height, width, CV_32FC1, input_nv12_float.data());

    CUDA_CHECK(cudaMalloc(&src_half, 1.5 * height * width * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&src_float, 1.5 * height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_half, 3 * height * width * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&dst_float, 3 * height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_float, src_nv12_raw.data, 1.5 * height * width * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    // convert with cuda
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src_float, src_half, 1.5 * height * width);
    ImageConvert<ImageType::kYUV_NV12, DataType::kHalf, ImageType::kBGR_HWC, DataType::kHalf, YUVFormula::kBT601>(
        src_half, dst_half, height, width);
    DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_float, 3 * height * width);

    cv::Mat actual(height, width, CV_32FC3);
    CUDA_CHECK(cudaMemcpy(actual.data, dst_float, 3 * height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // convert with opencv
    cv::Mat input_8uc3, expect_8uc3, expect(height, width, CV_32FC3);
    src_nv12_raw.convertTo(input_8uc3, CV_8UC3, 255.0);
    cv::cvtColor(input_8uc3, expect_8uc3, cv::ColorConversionCodes::COLOR_YUV2BGR_NV12);
    expect_8uc3.convertTo(expect, CV_32FC3, 1.0 / 255.0);

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);
    REQUIRE(max_diff < 0.1);

    CUDA_CHECK_AND_FREE(src_float);
    CUDA_CHECK_AND_FREE(src_half);
    CUDA_CHECK_AND_FREE(dst_float);
    CUDA_CHECK_AND_FREE(dst_half);
}

TEST_CASE("ImageConvert YUV_NV12 To YUV_I420", "[image_convert]") {
    void *src_nv12 = nullptr, *dst_i420 = nullptr;
    const int width = 6, height = 4;

    std::vector<unsigned char> nv12{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                    130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                    220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    std::vector<unsigned char> expect{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                      130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                      220, 132, 212, 33,  244, 139, 213, 80,  216, 242, 6,   170};

    CUDA_CHECK(cudaMalloc(&src_nv12, sizeof(unsigned char) * nv12.size()));
    CUDA_CHECK(cudaMalloc(&dst_i420, sizeof(unsigned char) * nv12.size()));
    CUDA_CHECK(cudaMemcpy(src_nv12, nv12.data(), sizeof(unsigned char) * nv12.size(), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    ImageConvert<ImageType::kYUV_NV12, DataType::kInt8, ImageType::kYUV_I420, DataType::kInt8>(src_nv12, dst_i420,
                                                                                               height, width);

    std::vector<unsigned char> actual(nv12.size());
    CUDA_CHECK(cudaMemcpy(&actual[0], dst_i420, sizeof(unsigned char) * nv12.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(fabs(actual[i] - expect[i]) < 1);
    }
}

TEST_CASE("ImageConvert YUV_422p To BGR_HWC", "[image_convert]") {
    int height = 1000, width = 2000;

    std::vector<unsigned char> input_y(height * width);
    smartmore::RandomInt8Vector(input_y);
    std::vector<unsigned char> input_u(height * width / 2);
    smartmore::RandomInt8Vector(input_u);
    std::vector<unsigned char> input_v(height * width / 2);
    smartmore::RandomInt8Vector(input_v);

    cv::Mat src_y(height, width, CV_8UC1, input_y.data());
    cv::Mat src_u(height, width / 2, CV_8UC1, input_u.data());
    cv::Mat src_v(height, width / 2, CV_8UC1, input_v.data());

    cv::Mat yuv_uyvy(height, width, CV_8UC2);
    // merge yuv_422p to yuv_uyvy manually
    for (int i = 0; i < height * width * 2; i += 4) {
        yuv_uyvy.data[i] = src_u.data[i / 4];          // u
        yuv_uyvy.data[i + 1] = src_y.data[i / 2];      // y
        yuv_uyvy.data[i + 2] = src_v.data[i / 4];      // v
        yuv_uyvy.data[i + 3] = src_y.data[i / 2 + 1];  // y
    }
    cv::Mat bgr_hwc(height, width, CV_8UC3);
    cv::cvtColor(yuv_uyvy, bgr_hwc, cv::ColorConversionCodes::COLOR_YUV2BGR_UYVY);

    // cvt yuv422p to bgr_hwc by cuda
    void *yuv422p_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&yuv422p_dev, height * width * 2 * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(yuv422p_dev, input_y.data(), height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<unsigned char *>(yuv422p_dev) + height * width, input_u.data(),
                          height * width / 2 * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<unsigned char *>(yuv422p_dev) + height * width * 3 / 2, input_v.data(),
                          height * width / 2 * sizeof(char), cudaMemcpyHostToDevice));

    void *bgr_hwc_dev;
    CUDA_CHECK(cudaMalloc(&bgr_hwc_dev, height * width * 3 * sizeof(char)));
    smartmore::cudaop::ImageConvert<smartmore::cudaop::ImageType::kYuv422p, smartmore::cudaop::DataType::kInt8,
                                    smartmore::cudaop::ImageType::kBGR_HWC, smartmore::cudaop::DataType::kInt8,
                                    smartmore::cudaop::YUVFormula::kBT601>(yuv422p_dev, bgr_hwc_dev, height, width);

    unsigned char *result_cuda = (unsigned char *)malloc(height * width * 3 * sizeof(unsigned char));
    CUDA_CHECK(cudaMemcpy(result_cuda, bgr_hwc_dev, height * width * 3 * sizeof(char), cudaMemcpyDeviceToHost));

    int step = height * width * 3;

    int cnt = 0;
    float max_diff = 0.0;
    float sum_diff = 0.0;
    for (int i = 0; i < step; i++) {
        float cur_diff = fabs((int)result_cuda[i] - (int)bgr_hwc.data[i]);
        sum_diff += cur_diff;
        max_diff = fmax(max_diff, cur_diff);
        if (cur_diff > 1.0) {
            cnt++;
        }
    }
    std::cout << "error larger than 1, cnt " << cnt << std::endl;
    std::cout << "YUV_422p To BGR_HWC, error rate: " << (float)cnt / step * 100 << "%" << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean_diff: " << sum_diff / step << std::endl;

    CUDA_CHECK_AND_FREE(yuv422p_dev);
    CUDA_CHECK_AND_FREE(bgr_hwc_dev);

    free(result_cuda);
}

TEST_CASE("ImageConvert YUV_UYVY To YUV_422p", "[image_convert]") {
    int height = 2000, width = 2000;

    std::vector<unsigned char> input_uyvy(height * width * 2);
    smartmore::RandomInt8Vector(input_uyvy);
    cv::Mat yuv_uyvy(height, width, CV_8UC2, input_uyvy.data());
    cv::Mat yuv422p(height, width, CV_8UC2, input_uyvy.data());

    // cvt uyvy to yuv422p by cuda
    void *uyvy_dev;
    void *yuv422p_dev;
    void *split_dev;
    CUDA_CHECK(cudaMalloc(&uyvy_dev, height * width * 2 * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&yuv422p_dev, height * width * 2 * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&split_dev, height * width * 2 * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(uyvy_dev, input_uyvy.data(), height * width * 2 * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kYUV_UYVY, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(
        uyvy_dev,
        {split_dev, static_cast<unsigned char *>(split_dev) + height * width,
         static_cast<unsigned char *>(split_dev) + height * width * 3 / 2},
        height, width);

    smartmore::cudaop::ImageConvert<smartmore::cudaop::ImageType::kYUV_UYVY, smartmore::cudaop::DataType::kInt8,
                                    smartmore::cudaop::ImageType::kYuv422p, smartmore::cudaop::DataType::kInt8>(
        uyvy_dev, yuv422p_dev, height, width);

    unsigned char *result_cuda = (unsigned char *)malloc(height * width * 2 * sizeof(unsigned char));
    CUDA_CHECK(cudaMemcpy(result_cuda, yuv422p_dev, height * width * 2 * sizeof(char), cudaMemcpyDeviceToHost));

    unsigned char *split_cuda = (unsigned char *)malloc(height * width * 2 * sizeof(unsigned char));
    CUDA_CHECK(cudaMemcpy(split_cuda, split_dev, height * width * 2 * sizeof(char), cudaMemcpyDeviceToHost));

    int step = height * width * 2;
    for (int i = 0; i < step; i++) {
        REQUIRE((int)split_cuda[i] == (int)result_cuda[i]);
    }

    CUDA_CHECK_AND_FREE(uyvy_dev);
    CUDA_CHECK_AND_FREE(yuv422p_dev);
    CUDA_CHECK_AND_FREE(split_dev);

    free(result_cuda);
    free(split_cuda);
}