/*******************************************************************************
 *  FILENAME:      test_image_channel_merge.cu
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

TEST_CASE("ImageChannelMerge to BGR_CHW", "[image_channel_merge]") {
    // channal merge to BGR_CHW, Int8ToInt8
    void *src_r_dev;
    void *src_g_dev;
    void *src_b_dev;
    void *dst_dev;
    // 3 * 4 image, use random number
    std::vector<unsigned char> input_bgr{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                         130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                         220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 3, width = 4;

    cv::Mat src_b(height, width, CV_8UC1, input_bgr.data());
    cv::Mat src_g(height, width, CV_8UC1, input_bgr.data() + height * width);
    cv::Mat src_r(height, width, CV_8UC1, input_bgr.data() + height * width * 2);

    CUDA_CHECK(cudaMalloc(&src_r_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_dev, 3 * height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_r_dev, src_r.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_g_dev, src_g.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_dev, src_b.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kBGR_CHW, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_r_dev, src_g_dev, src_b_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, 3 * height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // compare cuda with opencv
    for (int i = 0; i < height * width * 3; i++) {
        int result = static_cast<int>(result_cuda.data[i]);
        int expect = static_cast<int>(input_bgr[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_r_dev);
    CUDA_CHECK_AND_FREE(src_g_dev);
    CUDA_CHECK_AND_FREE(src_b_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("ImageChannelMerge to BGR_HWC", "[image_channel_merge]") {
    // channal merge to BGR_HWC, Int8ToInt8
    void *src_r_dev;
    void *src_g_dev;
    void *src_b_dev;
    void *dst_dev;
    // 3 * 4 image, use random number
    std::vector<unsigned char> input_bgr{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                         130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                         220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 3, width = 4;

    // assume data in vector in order of CHW
    cv::Mat src_b(height, width, CV_8UC1, input_bgr.data());
    cv::Mat src_g(height, width, CV_8UC1, input_bgr.data() + height * width);
    cv::Mat src_r(height, width, CV_8UC1, input_bgr.data() + height * width * 2);

    CUDA_CHECK(cudaMalloc(&src_r_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_dev, 3 * height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_r_dev, src_r.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_g_dev, src_g.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_dev, src_b.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kBGR_HWC, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_r_dev, src_g_dev, src_b_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, 3 * height * width * sizeof(char), cudaMemcpyDeviceToHost));

    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_cuda.data[i * 3]);
        int expect = static_cast<int>(input_bgr[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_cuda.data[i * 3 + 1]);
        expect = static_cast<int>(input_bgr[i + height * width]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_cuda.data[i * 3 + 2]);
        expect = static_cast<int>(input_bgr[i + height * width * 2]);

        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_r_dev);
    CUDA_CHECK_AND_FREE(src_g_dev);
    CUDA_CHECK_AND_FREE(src_b_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("ImageChannelMerge to BGRA_CHW", "[image_channel_merge]") {
    // BGRA_CHW channel merge, Int8ToInt8
    void *dst_dev = nullptr;
    void *src_b_dev = nullptr;
    void *src_g_dev = nullptr;
    void *src_r_dev = nullptr;
    // 3 * 4 * 4 CHW image, line 1~3 use random number, line 4 set to 255 for channel alpha
    std::vector<unsigned char> input_bgra{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170,
                                          255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    int height = 3, width = 4;

    cv::Mat src_b(height, width, CV_8UC1, input_bgra.data());
    cv::Mat src_g(height, width, CV_8UC1, input_bgra.data() + height * width);
    cv::Mat src_r(height, width, CV_8UC1, input_bgra.data() + height * width * 2);

    // trans 3 channels, 4th channel be set to 255
    CUDA_CHECK(cudaMalloc(&src_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_r_dev, src_r.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_g_dev, src_g.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_dev, src_b.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dst_dev, 4 * height * width * sizeof(char)));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kBGRA_CHW, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_r_dev, src_g_dev, src_b_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height, width, CV_8UC(4));

    // cuda image merge, merge to 3 channels image
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, height * width * sizeof(char) * 4, cudaMemcpyDeviceToHost));

    for (int i = 0; i < height * width * 4; i++) {
        int result = static_cast<int>(result_cuda.data[i]);
        int expect = static_cast<int>(input_bgra[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(dst_dev);
    CUDA_CHECK_AND_FREE(src_b_dev);
    CUDA_CHECK_AND_FREE(src_g_dev);
    CUDA_CHECK_AND_FREE(src_r_dev);
}

TEST_CASE("ImageChannelMerge to BGRA_HWC", "[image_channel_merge]") {
    // BGRA_HWC channel merge, Int8ToInt8
    void *dst_dev = nullptr;
    void *src_b_dev = nullptr;
    void *src_g_dev = nullptr;
    void *src_r_dev = nullptr;
    // 3 * 4 * 4 HWC image, line 1~3 use random number, line 4 set to 255 for channel alpha
    std::vector<unsigned char> input_bgra{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170,
                                          255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    int height = 3, width = 4;

    cv::Mat src_b(height, width, CV_8UC1, input_bgra.data());
    cv::Mat src_g(height, width, CV_8UC1, input_bgra.data() + height * width);
    cv::Mat src_r(height, width, CV_8UC1, input_bgra.data() + height * width * 2);

    // trans 3 channels, 4th channel be set to 255
    CUDA_CHECK(cudaMalloc(&src_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_r_dev, src_r.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_g_dev, src_g.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_dev, src_b.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dst_dev, 4 * height * width * sizeof(char)));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kBGRA_HWC, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_r_dev, src_g_dev, src_b_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height, width, CV_8UC(4));

    // cuda image merge, merge to 3 channels image
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, height * width * sizeof(char) * 4, cudaMemcpyDeviceToHost));

    for (int i = 0; i < height * width; i++) {
        // compare channel b
        int result = static_cast<int>(result_cuda.data[i * 4]);
        int expect = static_cast<int>(input_bgra[i]);
        REQUIRE(fabs(result - expect) < 2);
        // compare channel g
        result = static_cast<int>(result_cuda.data[i * 4 + 1]);
        expect = static_cast<int>(input_bgra[i + height * width]);
        REQUIRE(fabs(result - expect) < 2);
        // compare channel r
        result = static_cast<int>(result_cuda.data[i * 4 + 2]);
        expect = static_cast<int>(input_bgra[i + height * width * 2]);
        REQUIRE(fabs(result - expect) < 2);
        // compare channel a
        result = static_cast<int>(result_cuda.data[i * 4 + 3]);
        expect = static_cast<int>(input_bgra[i + height * width * 3]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(dst_dev);
    CUDA_CHECK_AND_FREE(src_b_dev);
    CUDA_CHECK_AND_FREE(src_g_dev);
    CUDA_CHECK_AND_FREE(src_r_dev);
}

TEST_CASE("ImageChannelMerge to RGB_CHW", "[image_channel_merge]") {
    // channal merge to RGB_CHW, Int8ToInt8
    void *src_r_dev;
    void *src_g_dev;
    void *src_b_dev;
    void *dst_dev;
    // 3 * 4 image, use random number
    std::vector<unsigned char> input_bgr{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                         130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                         220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 3, width = 4;

    cv::Mat src_r(height, width, CV_8UC1, input_bgr.data());
    cv::Mat src_g(height, width, CV_8UC1, input_bgr.data() + height * width);
    cv::Mat src_b(height, width, CV_8UC1, input_bgr.data() + height * width * 2);

    CUDA_CHECK(cudaMalloc(&src_r_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_dev, 3 * height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_r_dev, src_r.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_g_dev, src_g.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_dev, src_b.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kRGB_CHW, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_r_dev, src_g_dev, src_b_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, 3 * height * width * sizeof(char), cudaMemcpyDeviceToHost));

    // compare cuda with opencv
    for (int i = 0; i < height * width * 3; i++) {
        int result = static_cast<int>(result_cuda.data[i]);
        int expect = static_cast<int>(input_bgr[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_r_dev);
    CUDA_CHECK_AND_FREE(src_g_dev);
    CUDA_CHECK_AND_FREE(src_b_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("ImageChannelMerge to RGB_HWC", "[image_channel_merge]") {
    // channal merge to RGB_HWC, Int8ToInt8
    void *src_r_dev;
    void *src_g_dev;
    void *src_b_dev;
    void *dst_dev;
    // 3 * 4 image, use random number
    std::vector<unsigned char> input_bgr{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                         130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                         220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 3, width = 4;

    // assume data in vector in order of CHW
    cv::Mat src_r(height, width, CV_8UC1, input_bgr.data());
    cv::Mat src_g(height, width, CV_8UC1, input_bgr.data() + height * width);
    cv::Mat src_b(height, width, CV_8UC1, input_bgr.data() + height * width * 2);

    CUDA_CHECK(cudaMalloc(&src_r_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_dev, 3 * height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_r_dev, src_r.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_g_dev, src_g.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_dev, src_b.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kRGB_HWC, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_r_dev, src_g_dev, src_b_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, 3 * height * width * sizeof(char), cudaMemcpyDeviceToHost));

    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_cuda.data[i * 3]);
        int expect = static_cast<int>(input_bgr[i]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_cuda.data[i * 3 + 1]);
        expect = static_cast<int>(input_bgr[i + height * width]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_cuda.data[i * 3 + 2]);
        expect = static_cast<int>(input_bgr[i + height * width * 2]);

        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(src_r_dev);
    CUDA_CHECK_AND_FREE(src_g_dev);
    CUDA_CHECK_AND_FREE(src_b_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("ImageChannelMerge to RGBA_CHW", "[image_channel_merge]") {
    // RGBA_CHW channel merge, Int8ToInt8
    void *dst_dev = nullptr;
    void *src_b_dev = nullptr;
    void *src_g_dev = nullptr;
    void *src_r_dev = nullptr;
    // 3 * 4 * 4 CHW image, line 1~3 use random number, line 4 set to 255 for channel alpha
    std::vector<unsigned char> input_bgra{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170,
                                          255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    int height = 3, width = 4;

    cv::Mat src_r(height, width, CV_8UC1, input_bgra.data());
    cv::Mat src_g(height, width, CV_8UC1, input_bgra.data() + height * width);
    cv::Mat src_b(height, width, CV_8UC1, input_bgra.data() + height * width * 2);

    // trans 3 channels, 4th channel be set to 255
    CUDA_CHECK(cudaMalloc(&src_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_r_dev, src_r.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_g_dev, src_g.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_dev, src_b.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dst_dev, 4 * height * width * sizeof(char)));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kRGBA_CHW, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_r_dev, src_g_dev, src_b_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height, width, CV_8UC(4));

    // cuda image merge, merge to 3 channels image
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, height * width * sizeof(char) * 4, cudaMemcpyDeviceToHost));

    for (int i = 0; i < height * width * 4; i++) {
        int result = static_cast<int>(result_cuda.data[i]);
        int expect = static_cast<int>(input_bgra[i]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(dst_dev);
    CUDA_CHECK_AND_FREE(src_b_dev);
    CUDA_CHECK_AND_FREE(src_g_dev);
    CUDA_CHECK_AND_FREE(src_r_dev);
}

TEST_CASE("ImageChannelMerge to RGBA_HWC", "[image_channel_merge]") {
    // RGBA_HWC channel merge, Int8ToInt8
    void *dst_dev = nullptr;
    void *src_b_dev = nullptr;
    void *src_g_dev = nullptr;
    void *src_r_dev = nullptr;
    // 3 * 4 * 4 HWC image, line 1~3 use random number, line 4 set to 255 for channel alpha
    std::vector<unsigned char> input_bgra{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170,
                                          255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    int height = 3, width = 4;

    cv::Mat src_r(height, width, CV_8UC1, input_bgra.data());
    cv::Mat src_g(height, width, CV_8UC1, input_bgra.data() + height * width);
    cv::Mat src_b(height, width, CV_8UC1, input_bgra.data() + height * width * 2);

    // trans 3 channels, 4th channel be set to 255
    CUDA_CHECK(cudaMalloc(&src_b_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_g_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_r_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_r_dev, src_r.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_g_dev, src_g.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_dev, src_b.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dst_dev, 4 * height * width * sizeof(char)));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kRGBA_HWC, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_r_dev, src_g_dev, src_b_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height, width, CV_8UC(4));

    // cuda image merge, merge to 3 channels image
    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, height * width * sizeof(char) * 4, cudaMemcpyDeviceToHost));

    for (int i = 0; i < height * width; i++) {
        // compare channel b
        int result = static_cast<int>(result_cuda.data[i * 4]);
        int expect = static_cast<int>(input_bgra[i]);
        REQUIRE(fabs(result - expect) < 2);
        // compare channel g
        result = static_cast<int>(result_cuda.data[i * 4 + 1]);
        expect = static_cast<int>(input_bgra[i + height * width]);
        REQUIRE(fabs(result - expect) < 2);
        // compare channel r
        result = static_cast<int>(result_cuda.data[i * 4 + 2]);
        expect = static_cast<int>(input_bgra[i + height * width * 2]);
        REQUIRE(fabs(result - expect) < 2);
        // compare channel a
        result = static_cast<int>(result_cuda.data[i * 4 + 3]);
        expect = static_cast<int>(input_bgra[i + height * width * 3]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(dst_dev);
    CUDA_CHECK_AND_FREE(src_b_dev);
    CUDA_CHECK_AND_FREE(src_g_dev);
    CUDA_CHECK_AND_FREE(src_r_dev);
}

TEST_CASE("ImageChannelMerge to YUV_NV12", "[image_channel_merge]") {
    // YUV_NV12 channel merge, Int8ToInt8
    void *dst_dev = nullptr;
    void *src_y_dev = nullptr;
    void *src_u_dev = nullptr;
    void *src_v_dev = nullptr;
    // 4 * 6 HWC image, use random number
    std::vector<unsigned char> input_nv12{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 4, width = 6;
    // HWC
    cv::Mat src_y(height, width, CV_8UC1, input_nv12.data());
    cv::Mat src_u(height * 0.25, width, CV_8UC1, input_nv12.data() + height * width);
    cv::Mat src_v(height * 0.25, width, CV_8UC1, input_nv12.data() + height * width * 5 / 4);

    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(char) * 1.5));
    CUDA_CHECK(cudaMalloc(&src_y_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_u_dev, height * width * sizeof(char) * 0.25));
    CUDA_CHECK(cudaMalloc(&src_v_dev, height * width * sizeof(char) * 0.25));

    CUDA_CHECK(cudaMemcpy(src_y_dev, src_y.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_u_dev, src_u.data, height * width * sizeof(char) * 0.25, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_v_dev, src_v.data, height * width * sizeof(char) * 0.25, cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kYUV_NV12, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_y_dev, src_u_dev, src_v_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height * 1.5, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, height * width * sizeof(char) * 1.5, cudaMemcpyDeviceToHost));

    // compare channel y
    for (int i = 0; i < height * width; i++) {
        int result = static_cast<int>(result_cuda.data[i]);
        int expect = static_cast<int>(input_nv12[i]);
        REQUIRE(fabs(result - expect) < 2);
    }
    // compare channel u
    for (int i = 0; i < height * width * 0.5; i += 2) {
        int result = static_cast<int>(result_cuda.data[i + height * width]);
        int expect = static_cast<int>(input_nv12[i / 2 + height * width]);
        REQUIRE(fabs(result - expect) < 2);
    }
    // compare channel v
    for (int i = 0; i < height * width * 0.5; i += 2) {
        int result = static_cast<int>(result_cuda.data[i + 1 + height * width]);
        int expect = static_cast<int>(input_nv12[i / 2 + height * width * 5 / 4]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(dst_dev);
    CUDA_CHECK_AND_FREE(src_y_dev);
    CUDA_CHECK_AND_FREE(src_u_dev);
    CUDA_CHECK_AND_FREE(src_v_dev);
}

TEST_CASE("ImageChannelMerge to YUV_UYVY", "[image_channel_merge]") {
    // YUV_UYVY channel merge, Int8ToInt8
    void *dst_dev = nullptr;
    void *src_y_dev = nullptr;
    void *src_u_dev = nullptr;
    void *src_v_dev = nullptr;
    // 4 * 6 HWC image, use random number
    std::vector<unsigned char> input_uyvy{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                          130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                          220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    int height = 4, width = 6;
    // HWC
    cv::Mat src_y(height, width, CV_8UC1, input_uyvy.data());
    cv::Mat src_u(height * 0.5, width, CV_8UC1, input_uyvy.data() + height * width);
    cv::Mat src_v(height * 0.5, width, CV_8UC1, input_uyvy.data() + height * width * 3 / 2);

    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(char) * 2));
    CUDA_CHECK(cudaMalloc(&src_y_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&src_u_dev, height * width * sizeof(char) * 0.5));
    CUDA_CHECK(cudaMalloc(&src_v_dev, height * width * sizeof(char) * 0.5));

    CUDA_CHECK(cudaMemcpy(src_y_dev, src_y.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_u_dev, src_u.data, height * width * sizeof(char) * 0.5, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_v_dev, src_v.data, height * width * sizeof(char) * 0.5, cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kYUV_UYVY, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>({src_y_dev, src_u_dev, src_v_dev}, dst_dev,
                                                                             height, width);

    cv::Mat result_cuda(height * 2, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(result_cuda.data, dst_dev, height * width * sizeof(char) * 2, cudaMemcpyDeviceToHost));

    // compare channel y, u & v
    for (int i = 0; i < height * width * 2; i += 4) {
        int result = static_cast<int>(result_cuda.data[i]);
        int expect = static_cast<int>(src_u.data[i / 4]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_cuda.data[i + 2]);
        expect = static_cast<int>(src_v.data[i / 4]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_cuda.data[i + 1]);
        expect = static_cast<int>(src_y.data[i / 2]);
        REQUIRE(fabs(result - expect) < 2);

        result = static_cast<int>(result_cuda.data[i + 3]);
        expect = static_cast<int>(src_y.data[i / 2 + 1]);
        REQUIRE(fabs(result - expect) < 2);
    }

    CUDA_CHECK_AND_FREE(dst_dev);
    CUDA_CHECK_AND_FREE(src_y_dev);
    CUDA_CHECK_AND_FREE(src_u_dev);
    CUDA_CHECK_AND_FREE(src_v_dev);
}

TEST_CASE("ImageChannelMerge to YUV_I420", "[image_channel_merge]") {
    void *src_y = nullptr, *src_u = nullptr, *src_v = nullptr;
    void *dst = nullptr;
    const int height = 4, width = 6;

    std::vector<unsigned char> i420{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                    130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                    220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    CUDA_CHECK(cudaMalloc(&src_y, sizeof(unsigned char) * height * width));
    CUDA_CHECK(cudaMalloc(&src_u, sizeof(unsigned char) * height * width / 4));
    CUDA_CHECK(cudaMalloc(&src_v, sizeof(unsigned char) * height * width / 4));
    CUDA_CHECK(cudaMalloc(&dst, sizeof(unsigned char) * height * width * 3 / 2));

    CUDA_CHECK(cudaMemcpy(src_y, &i420[0], sizeof(unsigned char) * height * width, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(src_u, &i420[height * width], sizeof(unsigned char) * height * width / 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_v, &i420[height * width / 4 * 5], sizeof(unsigned char) * height * width / 4,
                          cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    ImageChannelMerge<ImageType::kYUV_I420, DataType::kInt8, DataType::kInt8>({src_y, src_u, src_v}, dst, height,
                                                                              width);

    std::vector<unsigned char> actual(height * width * 3 / 2);
    CUDA_CHECK(cudaMemcpy(&actual[0], dst, actual.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(fabs(actual[i] - i420[i]) < 1);
    }
    CUDA_CHECK_AND_FREE(src_y);
    CUDA_CHECK_AND_FREE(src_u);
    CUDA_CHECK_AND_FREE(src_v);
    CUDA_CHECK_AND_FREE(dst);
}

TEST_CASE("ImageChannelMerge half to RGB_CHW", "[image_channel_merge]") {
    // channal merge to BGR_CHW, HalfToHalf
    void *src_r_float = nullptr, *src_g_float = nullptr, *src_b_float = nullptr;
    void *src_r_half = nullptr, *src_g_half = nullptr, *src_b_half = nullptr;
    void *dst_half = nullptr, *dst_float = nullptr;

    int height = 3, width = 4;
    // 3 * 4 image, use random number
    std::vector<float> input_rgb(height * width * 3);
    smartmore::RandomFloatVector(input_rgb);

    CUDA_CHECK(cudaMalloc(&src_r_float, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&src_g_float, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&src_b_float, height * width * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&src_r_half, height * width * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&src_g_half, height * width * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&src_b_half, height * width * sizeof(float) / 2));

    CUDA_CHECK(cudaMalloc(&dst_half, 3 * height * width * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&dst_float, 3 * height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_r_float, &input_rgb[0], height * width * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(src_g_float, &input_rgb[height * width], height * width * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_b_float, &input_rgb[height * width * 2], height * width * sizeof(float),
                          cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src_r_float, src_r_half, height * width);
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src_g_float, src_g_half, height * width);
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src_b_float, src_b_half, height * width);

    // half to half
    {
        ImageChannelMerge<ImageType::kRGB_CHW, DataType::kHalf, DataType::kHalf>({src_r_half, src_g_half, src_b_half},
                                                                                 dst_half, height, width);

        DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_float, height * width * 3);

        std::vector<float> actual(height * width * 3);
        CUDA_CHECK(cudaMemcpy(&actual[0], dst_float, 3 * height * width * sizeof(float), cudaMemcpyDeviceToHost));

        // compare cuda with opencv
        for (int i = 0; i < height * width * 3; i++) {
            REQUIRE(fabs(actual[i] - input_rgb[i]) < 0.001);
        }
    }

    // half to float
    {
        ImageChannelMerge<ImageType::kRGB_CHW, DataType::kHalf, DataType::kFloat32>(
            {src_r_half, src_g_half, src_b_half}, dst_float, height, width);

        std::vector<float> actual(height * width * 3);
        CUDA_CHECK(cudaMemcpy(&actual[0], dst_float, 3 * height * width * sizeof(float), cudaMemcpyDeviceToHost));

        // compare cuda with opencv
        for (int i = 0; i < height * width * 3; i++) {
            REQUIRE(fabs(actual[i] - input_rgb[i]) < 0.001);
        }
    }

    CUDA_CHECK_AND_FREE(src_r_float);
    CUDA_CHECK_AND_FREE(src_g_float);
    CUDA_CHECK_AND_FREE(src_b_float);
    CUDA_CHECK_AND_FREE(src_r_half);
    CUDA_CHECK_AND_FREE(src_g_half);
    CUDA_CHECK_AND_FREE(src_b_half);
    CUDA_CHECK_AND_FREE(dst_half);
    CUDA_CHECK_AND_FREE(dst_float);
}

TEST_CASE("ImageChannelMerge YUV_422p To YUV_UYVY", "[image_channel_merge]") {
    int height = 1000, width = 1000;

    std::vector<unsigned char> input_y(height * width);
    smartmore::RandomInt8Vector(input_y);
    std::vector<unsigned char> input_u(height / 2 * width);
    smartmore::RandomInt8Vector(input_u);
    std::vector<unsigned char> input_v(height / 2 * width);
    smartmore::RandomInt8Vector(input_v);

    cv::Mat src_y(height, width, CV_8UC1, input_y.data());
    cv::Mat src_u(height / 2, width, CV_8UC1, input_u.data());
    cv::Mat src_v(height / 2, width, CV_8UC1, input_v.data());

    cv::Mat yuv_uyvy(height * 2, width, CV_8UC1);
    for (int i = 0; i < height * width * 2; i += 4) {
        yuv_uyvy.data[i] = src_u.data[i / 4];          // u
        yuv_uyvy.data[i + 1] = src_y.data[i / 2];      // y
        yuv_uyvy.data[i + 2] = src_v.data[i / 4];      // v
        yuv_uyvy.data[i + 3] = src_y.data[i / 2 + 1];  // y
    }

    // merge yuv422p to yuv_uyvy by cuda
    void *yuv422p_dev = nullptr;
    void *yuv_uyvy_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&yuv422p_dev, height * width * 2 * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&yuv_uyvy_dev, height * width * 2 * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(yuv422p_dev, input_y.data(), height * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<unsigned char *>(yuv422p_dev) + height * width, input_u.data(),
                          height / 2 * width * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<unsigned char *>(yuv422p_dev) + height * width * 3 / 2, input_v.data(),
                          height / 2 * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::ImageChannelMerge<smartmore::cudaop::ImageType::kYUV_UYVY, smartmore::cudaop::DataType::kInt8,
                                         smartmore::cudaop::DataType::kInt8>(
        {yuv422p_dev, static_cast<unsigned char *>(yuv422p_dev) + height * width,
         static_cast<unsigned char *>(yuv422p_dev) + height * width * 3 / 2},
        yuv_uyvy_dev, height, width);

    unsigned char *uyvy_cuda = (unsigned char *)malloc(height * width * 2 * sizeof(unsigned char));
    CUDA_CHECK(cudaMemcpy(uyvy_cuda, yuv_uyvy_dev, height * width * 2 * sizeof(char), cudaMemcpyDeviceToHost));

    int step = height * width * 2;
    for (int i = 0; i < step; i++) {
        REQUIRE(uyvy_cuda[i] == yuv_uyvy.data[i]);
    }

    CUDA_CHECK_AND_FREE(yuv422p_dev);
    CUDA_CHECK_AND_FREE(yuv_uyvy_dev);
    free(uyvy_cuda);
}