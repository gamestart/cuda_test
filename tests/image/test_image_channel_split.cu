/*******************************************************************************
 *  FILENAME:      test_image_channel_s.cu.cpp
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Monday May 24th 2021
 *
 *  LAST MODIFIED: Wednesday, July 14th 2021, 12:25 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("ImageChannelSplit BGR_HWC", "[image_channel_split]") {
    // BGR_HWC channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_b_dev = nullptr;
    void *dst_g_dev = nullptr;
    void *dst_r_dev = nullptr;
    // 3 * 4 * 3 HWC image, use random number
    std::vector<unsigned char> input_bgr{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                         130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                         220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 3, width = 4;
    // HWC
    cv::Mat src_bgr_raw(height, width, CV_8UC3, input_bgr.data());

    CUDA_CHECK(cudaMalloc(&src_dev, 3 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_bgr_raw.data, 3 * height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kBGR_HWC, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_r_dev, dst_g_dev, dst_b_dev},
                                                                             height, width);

    cv::Mat result_b_cuda(height, width, CV_8UC1);
    cv::Mat result_g_cuda(height, width, CV_8UC1);
    cv::Mat result_r_cuda(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_b_cuda.data, dst_b_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_g_cuda.data, dst_g_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_r_cuda.data, dst_r_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // convert with opencv
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(src_bgr_raw, bgrChannels);

    // compare cuda with opencv
    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_b_cuda.data[i]);
        int expect = static_cast<int>(bgrChannels[0].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_g_cuda.data[i]);
        expect = static_cast<int>(bgrChannels[1].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_r_cuda.data[i]);
        expect = static_cast<int>(bgrChannels[2].data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_b_dev);
    CUDA_CHECK_AND_FREE(dst_g_dev);
    CUDA_CHECK_AND_FREE(dst_r_dev);
}

TEST_CASE("ImageChannelSplit BGR_CHW", "[image_channel_split]") {
    // BGR_CHW channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_b_dev = nullptr;
    void *dst_g_dev = nullptr;
    void *dst_r_dev = nullptr;
    // 3 * 4 * 3 CHW image, use random number
    std::vector<unsigned char> input_bgr{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                         130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                         220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 3, width = 4;

    cv::Mat src_bgr_raw(height, width, CV_8UC3, input_bgr.data());
    // 3 channels represent as BGR
    cv::Mat src_b_raw(height, width, CV_8UC1, input_bgr.data());
    cv::Mat src_g_raw(height, width, CV_8UC1, input_bgr.data() + height * width);
    cv::Mat src_r_raw(height, width, CV_8UC1, input_bgr.data() + height * width * 2);

    CUDA_CHECK(cudaMalloc(&src_dev, 3 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_bgr_raw.data, 3 * height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kBGR_CHW, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_r_dev, dst_g_dev, dst_b_dev},
                                                                             height, width);

    cv::Mat result_b_cuda(height, width, CV_8UC1);
    cv::Mat result_g_cuda(height, width, CV_8UC1);
    cv::Mat result_r_cuda(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_b_cuda.data, dst_b_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_g_cuda.data, dst_g_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_r_cuda.data, dst_r_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // compare cuda with opencv
    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_b_cuda.data[i]);
        int expect = static_cast<int>(src_b_raw.data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_g_cuda.data[i]);
        expect = static_cast<int>(src_g_raw.data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_r_cuda.data[i]);
        expect = static_cast<int>(src_r_raw.data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_b_dev);
    CUDA_CHECK_AND_FREE(dst_g_dev);
    CUDA_CHECK_AND_FREE(dst_r_dev);
}

TEST_CASE("ImageChannelSplit BGRA_HWC", "[image_channel_split]") {
    // BGRA_HWC channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_b_dev = nullptr;
    void *dst_g_dev = nullptr;
    void *dst_r_dev = nullptr;
    // 3 * 4 * 4 HWC image, use random number
    std::vector<unsigned char> input_bgra{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170,
                                          92,  36,  72,  50,  44,  23,  196, 79,  199, 220, 225, 255};
    int height = 3, width = 4;

    cv::Mat src_bgra_raw(height, width, CV_8UC(4), input_bgra.data());

    CUDA_CHECK(cudaMalloc(&src_dev, 4 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_bgra_raw.data, 4 * height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kBGRA_HWC, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_r_dev, dst_g_dev, dst_b_dev},
                                                                             height, width);

    cv::Mat result_b_cuda(height, width, CV_8UC1);
    cv::Mat result_g_cuda(height, width, CV_8UC1);
    cv::Mat result_r_cuda(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_b_cuda.data, dst_b_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_g_cuda.data, dst_g_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_r_cuda.data, dst_r_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // convert with opencv
    std::vector<cv::Mat> bgraChannels(4);
    cv::split(src_bgra_raw, bgraChannels);

    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_b_cuda.data[i]);
        int expect = static_cast<int>(bgraChannels[0].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_g_cuda.data[i]);
        expect = static_cast<int>(bgraChannels[1].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_r_cuda.data[i]);
        expect = static_cast<int>(bgraChannels[2].data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_b_dev);
    CUDA_CHECK_AND_FREE(dst_g_dev);
    CUDA_CHECK_AND_FREE(dst_r_dev);
}

TEST_CASE("ImageChannelSplit BGRA_CHW", "[image_channel_split]") {
    // BGRA_CHW channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_b_dev = nullptr;
    void *dst_g_dev = nullptr;
    void *dst_r_dev = nullptr;
    // 3 * 4 * 4 CHW image, use random number
    std::vector<unsigned char> input_bgra{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170,
                                          92,  36,  72,  50,  44,  23,  196, 79,  199, 220, 225, 255};
    int height = 3, width = 4;

    cv::Mat src_bgra_raw(height, width, CV_8UC(4), input_bgra.data());

    CUDA_CHECK(cudaMalloc(&src_dev, 4 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_bgra_raw.data, 4 * height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kBGRA_CHW, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_r_dev, dst_g_dev, dst_b_dev},
                                                                             height, width);

    cv::Mat result_b_cuda(height, width, CV_8UC1);
    cv::Mat result_g_cuda(height, width, CV_8UC1);
    cv::Mat result_r_cuda(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_b_cuda.data, dst_b_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_g_cuda.data, dst_g_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_r_cuda.data, dst_r_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // direct split
    std::vector<cv::Mat> bgraChannels(4);
    bgraChannels[0] = cv::Mat(height, width, CV_8UC1, input_bgra.data());
    bgraChannels[1] = cv::Mat(height, width, CV_8UC1, input_bgra.data() + height * width);
    bgraChannels[2] = cv::Mat(height, width, CV_8UC1, input_bgra.data() + height * width * 2);

    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_b_cuda.data[i]);
        int expect = static_cast<int>(bgraChannels[0].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_g_cuda.data[i]);
        expect = static_cast<int>(bgraChannels[1].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_r_cuda.data[i]);
        expect = static_cast<int>(bgraChannels[2].data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_b_dev);
    CUDA_CHECK_AND_FREE(dst_g_dev);
    CUDA_CHECK_AND_FREE(dst_r_dev);
}

TEST_CASE("ImageChannelSplit RGB_HWC", "[image_channel_split]") {
    // RGB_HWC channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_r_dev = nullptr;
    void *dst_g_dev = nullptr;
    void *dst_b_dev = nullptr;
    // 3 * 4 * 3 HWC image, use random number
    std::vector<unsigned char> input_rgb{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                         130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                         220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 3, width = 4;
    // HWC
    cv::Mat src_rgb_raw(height, width, CV_8UC3, input_rgb.data());

    CUDA_CHECK(cudaMalloc(&src_dev, 3 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_rgb_raw.data, 3 * height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kRGB_HWC, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_r_dev, dst_g_dev, dst_b_dev},
                                                                             height, width);

    cv::Mat result_r_cuda(height, width, CV_8UC1);
    cv::Mat result_g_cuda(height, width, CV_8UC1);
    cv::Mat result_b_cuda(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_r_cuda.data, dst_r_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_g_cuda.data, dst_g_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_b_cuda.data, dst_b_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // convert with opencv
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(src_rgb_raw, rgbChannels);

    // compare cuda with opencv
    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_r_cuda.data[i]);
        int expect = static_cast<int>(rgbChannels[0].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_g_cuda.data[i]);
        expect = static_cast<int>(rgbChannels[1].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_b_cuda.data[i]);
        expect = static_cast<int>(rgbChannels[2].data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_r_dev);
    CUDA_CHECK_AND_FREE(dst_g_dev);
    CUDA_CHECK_AND_FREE(dst_b_dev);
}

TEST_CASE("ImageChannelSplit RGB_CHW", "[image_channel_split]") {
    // RGB_CHW channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_r_dev = nullptr;
    void *dst_g_dev = nullptr;
    void *dst_b_dev = nullptr;
    // 3 * 4 * 3 CHW image, use random number
    std::vector<unsigned char> input_rgb{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                         130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                         220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 3, width = 4;
    // CHW
    cv::Mat src_rgb_raw(height, width, CV_8UC3, input_rgb.data());
    // 3 channels represent as RGB
    cv::Mat src_r_raw(height, width, CV_8UC1, input_rgb.data());
    cv::Mat src_g_raw(height, width, CV_8UC1, input_rgb.data() + height * width);
    cv::Mat src_b_raw(height, width, CV_8UC1, input_rgb.data() + height * width * 2);

    CUDA_CHECK(cudaMalloc(&src_dev, 3 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_r_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_b_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_rgb_raw.data, 3 * height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kRGB_CHW, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_r_dev, dst_g_dev, dst_b_dev},
                                                                             height, width);

    cv::Mat result_r_cuda(height, width, CV_8UC1);
    cv::Mat result_g_cuda(height, width, CV_8UC1);
    cv::Mat result_b_cuda(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_r_cuda.data, dst_r_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_g_cuda.data, dst_g_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_b_cuda.data, dst_b_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // compare cuda with opencv
    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_r_cuda.data[i]);
        int expect = static_cast<int>(src_r_raw.data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_g_cuda.data[i]);
        expect = static_cast<int>(src_g_raw.data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_b_cuda.data[i]);
        expect = static_cast<int>(src_b_raw.data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_r_dev);
    CUDA_CHECK_AND_FREE(dst_g_dev);
    CUDA_CHECK_AND_FREE(dst_b_dev);
}

TEST_CASE("ImageChannelSplit RGBA_HWC", "[image_channel_split]") {
    // RGBA_HWC channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_b_dev = nullptr;
    void *dst_g_dev = nullptr;
    void *dst_r_dev = nullptr;
    // 3 * 4 * 4 HWC image, use random number
    std::vector<unsigned char> input_rgba{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170,
                                          92,  36,  72,  50,  44,  23,  196, 79,  199, 220, 225, 255};
    int height = 3, width = 4;

    cv::Mat src_rgba_raw(height, width, CV_8UC(4), input_rgba.data());

    CUDA_CHECK(cudaMalloc(&src_dev, 4 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_r_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_b_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_rgba_raw.data, 4 * height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kRGBA_HWC, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_r_dev, dst_g_dev, dst_b_dev},
                                                                             height, width);

    cv::Mat result_r_cuda(height, width, CV_8UC1);
    cv::Mat result_g_cuda(height, width, CV_8UC1);
    cv::Mat result_b_cuda(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_r_cuda.data, dst_r_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_g_cuda.data, dst_g_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_b_cuda.data, dst_b_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // convert with opencv
    std::vector<cv::Mat> rgbaChannels(4);
    cv::split(src_rgba_raw, rgbaChannels);

    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_r_cuda.data[i]);
        int expect = static_cast<int>(rgbaChannels[0].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_g_cuda.data[i]);
        expect = static_cast<int>(rgbaChannels[1].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_b_cuda.data[i]);
        expect = static_cast<int>(rgbaChannels[2].data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_r_dev);
    CUDA_CHECK_AND_FREE(dst_g_dev);
    CUDA_CHECK_AND_FREE(dst_b_dev);
}

TEST_CASE("ImageChannelSplit RGBA_CHW", "[image_channel_split]") {
    // RGBA_CHW channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_b_dev = nullptr;
    void *dst_g_dev = nullptr;
    void *dst_r_dev = nullptr;
    // 3 * 4 * 4 CHW image, use random number
    std::vector<unsigned char> input_rgba{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170,
                                          92,  36,  72,  50,  44,  23,  196, 79,  199, 220, 225, 255};
    int height = 3, width = 4;

    cv::Mat src_rgba_raw(height, width, CV_8UC(4), input_rgba.data());

    CUDA_CHECK(cudaMalloc(&src_dev, 4 * height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_r_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_b_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_rgba_raw.data, 4 * height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kRGBA_CHW, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_r_dev, dst_g_dev, dst_b_dev},
                                                                             height, width);

    cv::Mat result_r_cuda(height, width, CV_8UC1);
    cv::Mat result_g_cuda(height, width, CV_8UC1);
    cv::Mat result_b_cuda(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_r_cuda.data, dst_r_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_g_cuda.data, dst_g_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_b_cuda.data, dst_b_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // direct split
    std::vector<cv::Mat> rgbaChannels(4);
    rgbaChannels[0] = cv::Mat(height, width, CV_8UC1, input_rgba.data());
    rgbaChannels[1] = cv::Mat(height, width, CV_8UC1, input_rgba.data() + height * width);
    rgbaChannels[2] = cv::Mat(height, width, CV_8UC1, input_rgba.data() + height * width * 2);

    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_r_cuda.data[i]);
        int expect = static_cast<int>(rgbaChannels[0].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_g_cuda.data[i]);
        expect = static_cast<int>(rgbaChannels[1].data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_b_cuda.data[i]);
        expect = static_cast<int>(rgbaChannels[2].data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_b_dev);
    CUDA_CHECK_AND_FREE(dst_g_dev);
    CUDA_CHECK_AND_FREE(dst_r_dev);
}

TEST_CASE("ImageChannelSplit YUV_NV12", "[image_channel_split]") {
    // YUV_NV12 channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_y_dev = nullptr;
    void *dst_u_dev = nullptr;
    void *dst_v_dev = nullptr;
    // 4 * 6 HWC image, use random number
    std::vector<unsigned char> input_nv12{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 4, width = 6;
    // HWC
    cv::Mat src_nv12_raw(height * 1.5, width, CV_8UC1, input_nv12.data());

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(char) * 1.5));
    CUDA_CHECK(cudaMalloc(&dst_y_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_u_dev, height * width * sizeof(char) * 0.25));
    CUDA_CHECK(cudaMalloc(&dst_v_dev, height * width * sizeof(char) * 0.25));

    CUDA_CHECK(cudaMemcpy(src_dev, src_nv12_raw.data, height * width * sizeof(char) * 1.5, cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kYUV_NV12, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_y_dev, dst_u_dev, dst_v_dev},
                                                                             height, width);

    cv::Mat result_y_cuda(height, width, CV_8UC1);
    cv::Mat result_u_cuda(height / 2, width / 2, CV_8UC1);
    cv::Mat result_v_cuda(height / 2, width / 2, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_y_cuda.data, dst_y_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_u_cuda.data, dst_u_dev, height * width * sizeof(char) * 0.25, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_v_cuda.data, dst_v_dev, height * width * sizeof(char) * 0.25, cudaMemcpyDeviceToHost));

    // convert with opencv
    cv::Mat nv12_y(height, width, CV_8UC1);
    cv::Mat nv12_u(height / 2, width / 2, CV_8UC1);
    cv::Mat nv12_v(height / 2, width / 2, CV_8UC1);

    for (int i = 0; i < height * width; i++) {
        nv12_y.data[i] = src_nv12_raw.data[i];
    }
    for (int i = 0; i < height * width * 0.25; i++) {
        nv12_u.data[i] = src_nv12_raw.data[i * 2 + height * width];
        nv12_v.data[i] = src_nv12_raw.data[i * 2 + height * width + 1];
    }

    // compare cuda with opencv
    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_y_cuda.data[i]);
        int expect = static_cast<int>(nv12_y.data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }
    for (int i = 0; i < height * width * 0.25; i++) {
        int result = static_cast<int>(result_u_cuda.data[i]);
        int expect = static_cast<int>(nv12_u.data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_v_cuda.data[i]);
        expect = static_cast<int>(nv12_v.data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_y_dev);
    CUDA_CHECK_AND_FREE(dst_u_dev);
    CUDA_CHECK_AND_FREE(dst_v_dev);
}

TEST_CASE("ImageChannelSplit YUV_UYVY", "[image_channel_split]") {
    // YUV_UYVY channel split, Int8ToInt8
    void *src_dev = nullptr;
    void *dst_y_dev = nullptr;
    void *dst_u_dev = nullptr;
    void *dst_v_dev = nullptr;
    // 3 * 4 HWC image, use random number
    std::vector<unsigned char> input_uyvy{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187};
    int height = 3, width = 4;
    // HWC
    cv::Mat src_uyvy_raw(height * 2, width, CV_8UC1, input_uyvy.data());

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(char) * 2));
    CUDA_CHECK(cudaMalloc(&dst_y_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_u_dev, height * width * sizeof(char) * 0.5));
    CUDA_CHECK(cudaMalloc(&dst_v_dev, height * width * sizeof(char) * 0.5));

    CUDA_CHECK(cudaMemcpy(src_dev, src_uyvy_raw.data, height * width * sizeof(char) * 2, cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelSplit<smartmore::cudaop::ImageType::kYUV_UYVY, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(src_dev, {dst_y_dev, dst_u_dev, dst_v_dev},
                                                                             height, width);

    cv::Mat result_y_cuda(height, width, CV_8UC1);
    cv::Mat result_u_cuda(height / 2, width, CV_8UC1);
    cv::Mat result_v_cuda(height / 2, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_y_cuda.data, dst_y_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_u_cuda.data, dst_u_dev, height * width * sizeof(char) * 0.5, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_v_cuda.data, dst_v_dev, height * width * sizeof(char) * 0.5, cudaMemcpyDeviceToHost));

    // convert with opencv
    cv::Mat uyvy_y(height, width, CV_8UC1);
    cv::Mat uyvy_u(height / 2, width, CV_8UC1);
    cv::Mat uyvy_v(height / 2, width, CV_8UC1);

    for (int i = 0; i < height * width * 2; i += 4) {
        uyvy_y.data[i / 2] = src_uyvy_raw.data[i + 1];
        uyvy_y.data[i / 2 + 1] = src_uyvy_raw.data[i + 3];
        uyvy_u.data[i / 4] = src_uyvy_raw.data[i];
        uyvy_v.data[i / 4] = src_uyvy_raw.data[i + 2];
    }

    // compare cuda with opencv
    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_y_cuda.data[i]);
        int expect = static_cast<int>(uyvy_y.data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }
    for (int i = 0; i < height * width * 0.5; i++) {
        int result = static_cast<int>(result_u_cuda.data[i]);
        int expect = static_cast<int>(uyvy_u.data[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_v_cuda.data[i]);
        expect = static_cast<int>(uyvy_v.data[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_y_dev);
    CUDA_CHECK_AND_FREE(dst_u_dev);
    CUDA_CHECK_AND_FREE(dst_v_dev);
}

TEST_CASE("ImageChannelSplit to YUV_I420", "[image_channel_split]") {
    void *src = nullptr;
    void *dst_y = nullptr, *dst_u = nullptr, *dst_v = nullptr;
    const int height = 4, width = 6;

    std::vector<unsigned char> i420{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                    130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                    220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    CUDA_CHECK(cudaMalloc(&src, sizeof(unsigned char) * height * width * 3 / 2));
    CUDA_CHECK(cudaMalloc(&dst_y, sizeof(unsigned char) * height * width));
    CUDA_CHECK(cudaMalloc(&dst_u, sizeof(unsigned char) * height * width / 4));
    CUDA_CHECK(cudaMalloc(&dst_v, sizeof(unsigned char) * height * width / 4));

    CUDA_CHECK(cudaMemcpy(src, i420.data(), sizeof(unsigned char) * height * width * 3 / 2, cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    ImageChannelSplit<ImageType::kYUV_I420, DataType::kInt8, DataType::kInt8>(src, {dst_y, dst_u, dst_v}, height,
                                                                              width);

    std::vector<unsigned char> actual(height * width * 3 / 2);
    CUDA_CHECK(cudaMemcpy(&actual[0], dst_y, sizeof(unsigned char) * height * width, cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(&actual[height * width], dst_u, sizeof(unsigned char) * height * width / 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&actual[height * width / 4 * 5], dst_v, sizeof(unsigned char) * height * width / 4,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(fabs(actual[i] - i420[i]) < 1);
    }

    CUDA_CHECK_AND_FREE(src);
    CUDA_CHECK_AND_FREE(dst_y);
    CUDA_CHECK_AND_FREE(dst_u);
    CUDA_CHECK_AND_FREE(dst_v);
}

TEST_CASE("ImageChannelSplit RGB_HWC to half", "[image_channel_split]") {
    void *src_rgb_float = nullptr, *src_rgb_half = nullptr;
    void *dst_r_half = nullptr, *dst_g_half = nullptr, *dst_b_half = nullptr;
    void *dst_r_float = nullptr, *dst_g_float = nullptr, *dst_b_float = nullptr;

    int height = 3, width = 4;
    std::vector<float> input_rgb(height * width * 3);
    std::vector<float> actual(height * width * 3);
    smartmore::RandomFloatVector(input_rgb);

    CUDA_CHECK(cudaMalloc(&src_rgb_float, height * width * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&src_rgb_half, height * width * 3 * sizeof(float) / 2));

    CUDA_CHECK(cudaMalloc(&dst_r_half, height * width * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&dst_g_half, height * width * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&dst_b_half, height * width * sizeof(float) / 2));

    CUDA_CHECK(cudaMalloc(&dst_r_float, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_g_float, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_b_float, height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_rgb_float, input_rgb.data(), height * width * 3 * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src_rgb_float, src_rgb_half, height * width * 3);

    {
        ImageChannelSplit<ImageType::kRGB_CHW, DataType::kHalf, DataType::kHalf>(
            src_rgb_half, {dst_r_half, dst_g_half, dst_b_half}, height, width);

        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_r_half, dst_r_float, sizeof(float) * height * width);
        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_g_half, dst_g_float, sizeof(float) * height * width);
        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_b_half, dst_b_float, sizeof(float) * height * width);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst_r_float, sizeof(float) * height * width, cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(&actual[height * width], dst_g_float, sizeof(float) * height * width, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&actual[2 * height * width], dst_b_float, sizeof(float) * height * width,
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < input_rgb.size(); i++) {
            REQUIRE(fabs(input_rgb[i] - actual[i]) < 0.001);
        }
    }

    {
        ImageChannelSplit<ImageType::kRGB_CHW, DataType::kHalf, DataType::kFloat32>(
            src_rgb_half, {dst_r_float, dst_g_float, dst_b_float}, height, width);

        CUDA_CHECK(cudaMemcpy(&actual[0], dst_r_float, sizeof(float) * height * width, cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(&actual[height * width], dst_g_float, sizeof(float) * height * width, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&actual[2 * height * width], dst_b_float, sizeof(float) * height * width,
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < input_rgb.size(); i++) {
            REQUIRE(fabs(input_rgb[i] - actual[i]) < 0.001);
        }
    }
}
