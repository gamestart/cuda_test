/*******************************************************************************
 *  FILENAME:      test_transpose.cu
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Friday September 3rd 2021
 *
 *  LAST MODIFIED: Monday, September 6th 2021, 11:34:30 am
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/tensor/transpose.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

TEST_CASE("Transpose Int 8", "[tensor_transpose]") {
    int height = 1080, width = 1920;

    std::vector<unsigned char> vec_gray(height * width);
    smartmore::RandomInt8Vector(vec_gray);

    cv::Mat src_gray(height, width, CV_8UC1, vec_gray.data());
    cv::Mat dst_mat(height, width, CV_8UC1);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_gray.data, height * width * sizeof(char), cudaMemcpyHostToDevice));

    smartmore::cudaop::Transpose<smartmore::cudaop::DataType::kInt8>(src_dev, dst_dev, height, width);

    std::vector<uchar> result_cuda(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda.data(), dst_dev, height * width * sizeof(char), cudaMemcpyDeviceToHost));

    cv::transpose(src_gray, dst_mat);

    // compare cuda with opencv
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int cur_diff = abs((int)result_cuda[i * height + j] - (int)(dst_mat.at<unsigned char>(i, j)));
            REQUIRE(cur_diff == 0);
        }
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Transpose Float 32", "[tensor_transpose]") {
    int height = 1080, width = 1920;

    std::vector<uchar> vec_gray(height * width);
    smartmore::RandomInt8Vector(vec_gray);
    cv::Mat src_mat(height, width, CV_8UC1, vec_gray.data());
    src_mat.convertTo(src_mat, CV_32FC1, 1 / 255.0);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_mat.data, height * width * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Transpose-float-CUDA: ");
        smartmore::cudaop::Transpose<smartmore::cudaop::DataType::kFloat32>(src_dev, dst_dev, height, width);
    }

    std::vector<float> result_cuda(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    cv::Mat dst_mat(height, width, CV_32FC1);

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Transpose-float-OpenCV: ");
        cv::transpose(src_mat, dst_mat);
    }

    // compare cuda with opencv
    float epsilon = 0.0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            float cur_diff = fabs(result_cuda[i * height + j] - dst_mat.at<float>(i, j));
            REQUIRE(cur_diff <= epsilon);
        }
    }

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}
