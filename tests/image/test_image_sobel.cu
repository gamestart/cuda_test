/*******************************************************************************
 *  FILENAME:      test_image_sobel.cu
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Monday September 6th 2021
 *
 *  LAST MODIFIED: Tuesday, September 7th 2021, 5:31:37 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/image/imgsobel.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

TEST_CASE("Sobel X", "[image_sobel]") {
    int height = 1080, width = 1920;

    std::vector<unsigned char> vec_gray(height * width);
    smartmore::RandomInt8Vector(vec_gray);

    cv::Mat src_gray(height, width, CV_8UC1, vec_gray.data());
    cv::Mat src_gray_f32;
    src_gray.convertTo(src_gray_f32, CV_32FC1, 1 / 255.0);
    cv::Mat dst_mat(height, width, CV_32FC1);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_gray_f32.data, height * width * sizeof(float), cudaMemcpyHostToDevice));
    // x deriv
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sobel X deriv-cuda: ");
        smartmore::cudaop::ImageSobel<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32>(
            src_dev, dst_dev, height, width, 1, 0);
    }

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sobel X deriv-opencv: ");
        cv::Sobel(src_gray_f32, dst_mat, CV_32F, 1, 0);  // x deriv
    }

    // compare cuda with opencv
    float max_diff = 0.0;
    float sum_diff = 0.0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float cur_diff = fabs(result_cuda_f32[i * width + j] - dst_mat.at<float>(i, j));
            max_diff = fmax(max_diff, cur_diff);
            sum_diff += cur_diff;
        }
    }
    std::cout << "Sobel Deriv X: " << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Sobel Y", "[image_sobel]") {
    int height = 1080, width = 1920;

    std::vector<unsigned char> vec_gray(height * width);
    smartmore::RandomInt8Vector(vec_gray);

    cv::Mat src_gray(height, width, CV_8UC1, vec_gray.data());
    cv::Mat src_gray_f32;
    src_gray.convertTo(src_gray_f32, CV_32FC1, 1 / 255.0);
    cv::Mat dst_mat(height, width, CV_32FC1);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_gray_f32.data, height * width * sizeof(float), cudaMemcpyHostToDevice));
    // x deriv
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sobel X deriv-cuda: ");
        smartmore::cudaop::ImageSobel<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32>(
            src_dev, dst_dev, height, width, 0, 1);
    }

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sobel X deriv-opencv: ");
        cv::Sobel(src_gray_f32, dst_mat, CV_32F, 0, 1);  // y deriv
    }

    // compare cuda with opencv
    float max_diff = 0.0;
    float sum_diff = 0.0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float cur_diff = fabs(result_cuda_f32[i * width + j] - dst_mat.at<float>(i, j));
            max_diff = fmax(max_diff, cur_diff);
            sum_diff += cur_diff;
        }
    }
    std::cout << "Sobel Deriv Y: " << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Scharr when ksize == -1", "[image_sobel]") {
    int height = 1080, width = 1920;

    std::vector<unsigned char> vec_gray(height * width);
    smartmore::RandomInt8Vector(vec_gray);

    cv::Mat src_gray(height, width, CV_8UC1, vec_gray.data());
    cv::Mat src_gray_f32;
    src_gray.convertTo(src_gray_f32, CV_32FC1, 1 / 255.0);
    cv::Mat dst_mat(height, width, CV_32FC1);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_gray_f32.data, height * width * sizeof(float), cudaMemcpyHostToDevice));
    // x deriv
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sobel X deriv-cuda: ");
        smartmore::cudaop::ImageSobel<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32, -1>(
            src_dev, dst_dev, height, width, 1, 0);
    }

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sobel X deriv-opencv: ");
        cv::Sobel(src_gray_f32, dst_mat, CV_32F, 1, 0, -1);  // x deriv
    }

    // compare cuda with opencv
    float max_diff = 0.0;
    float sum_diff = 0.0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float cur_diff = fabs(result_cuda_f32[i * width + j] - dst_mat.at<float>(i, j));
            max_diff = fmax(max_diff, cur_diff);
            sum_diff += cur_diff;
        }
    }
    std::cout << "Scharr Deriv X: " << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Sobel X with scale delta", "[image_sobel]") {
    double scale = (rand() % 1000) / 500.0;
    double delta = rand() % 1000;
    int height = 1080, width = 1920;

    std::vector<unsigned char> vec_gray(height * width);
    smartmore::RandomInt8Vector(vec_gray);

    cv::Mat src_gray(height, width, CV_8UC1, vec_gray.data());
    cv::Mat src_gray_f32;
    src_gray.convertTo(src_gray_f32, CV_32FC1, 1 / 255.0);
    cv::Mat dst_mat(height, width, CV_32FC1);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_gray_f32.data, height * width * sizeof(float), cudaMemcpyHostToDevice));
    // x deriv
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sobel X deriv-cuda: ");
        smartmore::cudaop::ImageSobel<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32>(
            src_dev, dst_dev, height, width, 1, 0, scale, delta);
    }

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sobel X deriv-opencv: ");
        cv::Sobel(src_gray_f32, dst_mat, CV_32F, 1, 0, 3, scale, delta);  // x deriv
    }

    // compare cuda with opencv
    float max_diff = 0.0;
    float sum_diff = 0.0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float cur_diff = fabs(result_cuda_f32[i * width + j] - dst_mat.at<float>(i, j));
            max_diff = fmax(max_diff, cur_diff);
            sum_diff += cur_diff;
        }
    }
    std::cout << "Sobel Deriv X with scale delta: " << std::endl;
    std::cout << "scale: " << scale << ", delta: " << delta << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}
