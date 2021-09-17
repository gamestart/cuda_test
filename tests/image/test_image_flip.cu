/*******************************************************************************
 *  FILENAME:      test_image_flip.cu
 *
 *  AUTHORS:       Hou Yue    START DATE: Friday July 30th 2021
 *
 *  LAST MODIFIED: Tuesday, August 3rd 2021, 8:37:59 pm
 *
 *  CONTACT:       yue.hou@smartmore.com
 *******************************************************************************/

#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

template <typename DataType>
inline void YUVHorFlip(void *src, void *dst, const int width, const int height) {
    int index = 0;
    // Y
    for (int i = 0; i < height; i++) {
        for (int j = width - 1; j >= 0; j--) {
            ((DataType *)dst)[index++] = ((DataType *)src)[i * width + j];
        }
    }
    // U
    DataType *uheader = ((DataType *)src) + width * height;
    for (int i = 0; i < height / 2; i++) {
        for (int j = width / 2 - 1; j >= 0; j--) {
            ((DataType *)dst)[index++] = uheader[i * width / 2 + j];
        }
    }
    // V
    DataType *vheader = uheader + width * height / 4;
    for (int i = 0; i < height / 2; i++) {
        for (int j = width / 2 - 1; j >= 0; j--) {
            ((DataType *)dst)[index++] = vheader[i * width / 2 + j];
        }
    }
}

template <typename DataType>
inline void YUVertFlip(void *src, void *dst, const int width, const int height) {
    int index = 0;
    // Y
    for (int i = height - 1; i >= 0; i--) {
        for (int j = 0; j < width; j++) {
            ((DataType *)dst)[index++] = ((DataType *)src)[i * width + j];
        }
    }
    // U
    DataType *uheader = ((DataType *)src) + width * height;
    for (int i = height / 2 - 1; i >= 0; i--) {
        for (int j = 0; j < width / 2; j++) {
            ((DataType *)dst)[index++] = uheader[i * width / 2 + j];
        }
    }
    // V
    DataType *vheader = uheader + width * height / 4;
    for (int i = height / 2 - 1; i >= 0; i--) {
        for (int j = 0; j < width / 2; j++) {
            ((DataType *)dst)[index++] = vheader[i * width / 2 + j];
        }
    }
}

template <typename DataType>
inline void YUVHorVertFlip(void *src, void *dst, const int width, const int height) {
    int index = 0;
    // Y
    for (int i = height - 1; i >= 0; i--) {
        for (int j = width - 1; j >= 0; j--) {
            ((DataType *)dst)[index++] = ((DataType *)src)[i * width + j];
        }
    }
    // U
    DataType *uheader = ((DataType *)src) + width * height;
    for (int i = height / 2 - 1; i >= 0; i--) {
        for (int j = width / 2 - 1; j >= 0; j--) {
            ((DataType *)dst)[index++] = uheader[i * width / 2 + j];
        }
    }
    // V
    DataType *vheader = uheader + width * height / 4;
    for (int i = height / 2 - 1; i >= 0; i--) {
        for (int j = width / 2 - 1; j >= 0; j--) {
            ((DataType *)dst)[index++] = vheader[i * width / 2 + j];
        }
    }
}

template <typename DataType>
inline void YUVFlip(cv::Mat src, cv::Mat dst, int width, int height, int flipcode) {
    DataType *yuvbuf = new DataType[width * height * 3 / 2];
    memcpy(yuvbuf, src.data, height * width * 3 / 2);
    DataType *dstbuf = new DataType[width * height * 3 / 2];
    switch (flipcode) {
        case 1:
            YUVHorFlip<DataType>(yuvbuf, dstbuf, width, height);
            break;
        case 0:
            YUVertFlip<DataType>(yuvbuf, dstbuf, width, height);
            break;
        case -1:
            YUVHorVertFlip<DataType>(yuvbuf, dstbuf, width, height);
            break;
        default:
            break;
    }

    memcpy(dst.data, dstbuf, width * height * 3 / 2);
    delete[] yuvbuf;
    delete[] dstbuf;
}

// kBGR_HWC
TEST_CASE("ImageFlip to kBGR_HWC", "[image_flip]") {
    const int height = 3333, width = 2000;

    void *d_src, *d_dst;

    std::vector<float> input_bgr(width * height * 3);
    smartmore::RandomFloatVector(input_bgr);

    cv::Mat src_f = cv::Mat(height, width, CV_32FC3, input_bgr.data());
    cv::Mat dst_f = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat actual(height, width, CV_32FC3);
    cv::Mat expect(height, width, CV_32FC3);

    CUDA_CHECK(cudaMalloc(&d_src, sizeof(float) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels()));
    CUDA_CHECK(cudaMemcpy(d_src, src_f.data, sizeof(float) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, dst_f.data, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels(),
                          cudaMemcpyHostToDevice));
    using namespace smartmore::cudaop;
    SECTION("Hor flip to kBGR_HWC") {
        ImageFlip<DataType::kFloat32, ImageType::kBGR_HWC, FlipType::kHor>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, expect, 1);
    }
    SECTION("Vert flip to kBGR_HWC") {
        ImageFlip<DataType::kFloat32, ImageType::kBGR_HWC, FlipType::kVert>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, expect, 0);
    }
    SECTION("Hor and vert flip to kBGR_HWC") {
        ImageFlip<DataType::kFloat32, ImageType::kBGR_HWC, FlipType::kHor_Vert>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, expect, -1);
    }

    CUDA_CHECK(cudaMemcpy(actual.data, d_dst, sizeof(float) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);
    REQUIRE(max_diff < 0.0001);
    CUDA_CHECK_AND_FREE(d_src);
    CUDA_CHECK_AND_FREE(d_dst);
}

// kBGR_CHW
TEST_CASE("ImageHorFlip to kBGR_CHW", "[image_flip]") {
    const int height = 3333, width = 2000;
    void *d_src, *d_dst;
    std::vector<float> input_bgr(width * height * 3);
    smartmore::RandomFloatVector(input_bgr);

    cv::Mat src_f = cv::Mat(height, width, CV_32FC3, input_bgr.data());
    cv::Mat dst_f = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat actual(height, width, CV_32FC3);

    cv::Mat mat, mat_flipped;
    std::vector<cv::Mat> channels;

    // cuda flip
    cv::split(src_f, channels);
    cv::Mat src_chw(src_f.rows, src_f.cols, CV_32FC3);
    memcpy(src_chw.data, channels[0].data, sizeof(float) * src_f.rows * src_f.cols);
    memcpy((float *)(src_chw.data) + src_f.cols * src_f.rows, channels[1].data,
           sizeof(float) * src_f.rows * src_f.cols);
    memcpy((float *)(src_chw.data) + 2 * src_f.cols * src_f.rows, channels[2].data,
           sizeof(float) * src_f.rows * src_f.cols);

    // cuda memory
    CUDA_CHECK(cudaMalloc(&d_src, sizeof(float) * src_chw.cols * src_chw.rows * src_chw.channels()));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels()));
    CUDA_CHECK(cudaMemcpy(d_src, src_chw.data, sizeof(float) * src_chw.cols * src_chw.rows * src_chw.channels(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, dst_f.data, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels(),
                          cudaMemcpyHostToDevice));
    using namespace smartmore::cudaop;
    SECTION("Hor flip to kBGR_CHW") {
        ImageFlip<DataType::kFloat32, ImageType::kBGR_CHW, FlipType::kHor>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, mat_flipped, 1);
    }
    SECTION("Vert flip to kBGR_CHW") {
        ImageFlip<DataType::kFloat32, ImageType::kBGR_CHW, FlipType::kVert>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, mat_flipped, 0);
    }
    SECTION("Hor and vert flip to kBGR_CHW") {
        ImageFlip<DataType::kFloat32, ImageType::kBGR_CHW, FlipType::kHor_Vert>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, mat_flipped, -1);
    }

    CUDA_CHECK(cudaMemcpy(actual.data, d_dst, sizeof(float) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));

    // opencv flip
    cv::split(mat_flipped, channels);
    cv::Mat expect(src_f.rows, src_f.cols, CV_32FC3);
    memcpy(expect.data, channels[0].data, sizeof(float) * mat_flipped.rows * mat_flipped.cols);
    memcpy((float *)(expect.data) + mat_flipped.cols * mat_flipped.rows, channels[1].data,
           sizeof(float) * mat_flipped.rows * mat_flipped.cols);
    memcpy((float *)(expect.data) + 2 * mat_flipped.cols * mat_flipped.rows, channels[2].data,
           sizeof(float) * mat_flipped.rows * mat_flipped.cols);

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);
    REQUIRE(max_diff < 0.0001);
    CUDA_CHECK_AND_FREE(d_src);
    CUDA_CHECK_AND_FREE(d_dst);
}

TEST_CASE("ImageFlip to kGRAY", "[image_flip]") {
    const int height = 3333, width = 2000;

    void *d_src, *d_dst;

    std::vector<uchar> input_gray(width * height * 3);
    smartmore::RandomInt8Vector(input_gray);

    cv::Mat src_f = cv::Mat(height, width, CV_8UC1, input_gray.data());
    cv::Mat dst_f = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat actual(height, width, CV_8UC1);
    cv::Mat expect(height, width, CV_8UC1);

    CUDA_CHECK(cudaMalloc(&d_src, sizeof(unsigned char) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeof(unsigned char) * dst_f.cols * dst_f.rows * dst_f.channels()));
    CUDA_CHECK(cudaMemcpy(d_src, src_f.data, sizeof(unsigned char) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, dst_f.data, sizeof(unsigned char) * dst_f.cols * dst_f.rows * dst_f.channels(),
                          cudaMemcpyHostToDevice));
    using namespace smartmore::cudaop;
    SECTION("Hor flip to kGRAY") {
        ImageFlip<DataType::kInt8, ImageType::kGRAY, FlipType::kHor>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, expect, 1);
    }
    SECTION("Vert flip to kGRAY") {
        ImageFlip<DataType::kInt8, ImageType::kGRAY, FlipType::kVert>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, expect, 0);
    }
    SECTION("Hor and vert flip to kGRAY") {
        ImageFlip<DataType::kInt8, ImageType::kGRAY, FlipType::kHor_Vert>(d_src, d_dst, Size{width, height});
        cv::flip(src_f, expect, -1);
    }

    CUDA_CHECK(cudaMemcpy(actual.data, d_dst, sizeof(unsigned char) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));

    int maxDiff = 0;
    for (int i = 0; i < actual.rows; i++) {
        for (int j = 0; j < actual.cols; j++) {
            int diff = abs(actual.at<unsigned char>(i, j) - expect.at<unsigned char>(i, j));
            if (diff > maxDiff) maxDiff = diff;
        }
    }
    REQUIRE(maxDiff < 1);
    CUDA_CHECK_AND_FREE(d_src);
    CUDA_CHECK_AND_FREE(d_dst);
}

// kYUV_I420
TEST_CASE("ImageFlip to kYUV_I420", "[image_flip]") {
    const int height = 434, width = 600;

    void *d_src, *d_dst;

    std::vector<uchar> input_yuv420(width * height * 3 / 2);
    smartmore::RandomInt8Vector(input_yuv420);

    cv::Mat src_f = cv::Mat(height * 3 / 2, width, CV_8UC1, input_yuv420.data());
    cv::Mat dst_f = cv::Mat::zeros(height * 3 / 2, width, CV_8UC1);
    cv::Mat actual(height * 3 / 2, width, CV_8UC1);
    cv::Mat expect(height * 3 / 2, width, CV_8UC1);

    CUDA_CHECK(cudaMalloc(&d_src, sizeof(unsigned char) * height * width * 3 / 2));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeof(unsigned char) * height * width * 3 / 2));
    CUDA_CHECK(cudaMemcpy(d_src, src_f.data, sizeof(unsigned char) * height * width * 3 / 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, dst_f.data, sizeof(unsigned char) * height * width * 3 / 2, cudaMemcpyHostToDevice));
    using namespace smartmore::cudaop;

    SECTION("Hor flip to kYUV_I420") {
        ImageFlip<DataType::kInt8, ImageType::kYUV_I420, FlipType::kHor>(d_src, d_dst, Size{width, height});
        YUVFlip<unsigned char>(src_f, expect, width, height, 1);
    }
    SECTION("Vert flip to kYUV_I420") {
        ImageFlip<DataType::kInt8, ImageType::kYUV_I420, FlipType::kVert>(d_src, d_dst, Size{width, height});
        YUVFlip<unsigned char>(src_f, expect, width, height, 0);
    }
    SECTION("Hor and vert flip to kYUV_I420") {
        ImageFlip<DataType::kInt8, ImageType::kYUV_I420, FlipType::kHor_Vert>(d_src, d_dst, Size{width, height});
        YUVFlip<unsigned char>(src_f, expect, width, height, -1);
    }
    CUDA_CHECK(cudaMemcpy(actual.data, d_dst, sizeof(unsigned char) * width * height * 3 / 2, cudaMemcpyDeviceToHost));

    int maxDiff = 0;
    for (int i = 0; i < actual.rows; i++) {
        for (int j = 0; j < actual.cols; j++) {
            int diff = abs(actual.at<unsigned char>(i, j) - expect.at<unsigned char>(i, j));
            if (diff > maxDiff) maxDiff = diff;
        }
    }
    REQUIRE(maxDiff < 1);

    CUDA_CHECK_AND_FREE(d_src);
    CUDA_CHECK_AND_FREE(d_dst);
}
