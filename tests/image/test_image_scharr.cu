/*******************************************************************************
 *  FILENAME:      test_image_scharr.cu
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Tuesday August 31st 2021
 *
 *  LAST MODIFIED: Tuesday, September 7th 2021, 5:31:12 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/image/imgscharr.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

TEST_CASE("Scharr X deriv", "[image_scharr]") {
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
        smartmore::Clock clk("Scharr X deriv-cuda: ");
        smartmore::cudaop::ImageScharr<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32>(
            src_dev, dst_dev, height, width);
    }

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Macro CV_32F --- 5

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Scharr X deriv-opencv: ");
        cv::Scharr(src_gray_f32, dst_mat, CV_32F, 1, 0);  // x deriv
    }
    cv::imwrite("../data/output/scharr_gray.png", dst_mat * 255);

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

TEST_CASE("Scharr Y deriv", "[image_scharr]") {
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
    // y deriv
    smartmore::cudaop::ImageScharr<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32>(
        src_dev, dst_dev, height, width, 0, 1);

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Macro CV_32F --- 5
    cv::Scharr(src_gray_f32, dst_mat, CV_32F, 0, 1);  // y deriv
    cv::imwrite("../data/output/scharr_gray.png", dst_mat * 255);

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
    std::cout << "Scharr Deriv Y: " << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Scharr X deriv with scale delta", "[image_scharr]") {
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
    smartmore::cudaop::ImageScharr<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32>(
        src_dev, dst_dev, height, width, 1, 0, scale, delta);

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Macro CV_32F --- 5
    cv::Scharr(src_gray_f32, dst_mat, CV_32F, 1, 0, scale, delta);  // x deriv
    cv::imwrite("../data/output/scharr_gray.png", dst_mat * 255);

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
    std::cout << "Scharr Deriv X with scale delta: " << std::endl;
    std::cout << "scale: " << scale << ", delta: " << delta << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Scharr Y deriv with scale delta", "[image_scharr]") {
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
    smartmore::cudaop::ImageScharr<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32>(
        src_dev, dst_dev, height, width, 0, 1, scale, delta);

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Macro CV_32F --- 5
    cv::Scharr(src_gray_f32, dst_mat, CV_32F, 0, 1, scale, delta);  // x deriv
    cv::imwrite("../data/output/scharr_gray.png", dst_mat * 255);

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
    std::cout << "Scharr Deriv Y with scale delta: " << std::endl;
    std::cout << "scale: " << scale << ", delta: " << delta << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Scharr X deriv with reflect-total border type", "[image_scharr]") {
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
    smartmore::cudaop::ImageScharr<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32,
                                   smartmore::cudaop::BorderType::kReflectTotal>(src_dev, dst_dev, height, width, 1, 0,
                                                                                 scale, delta);

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Macro CV_32F --- 5
    cv::Scharr(src_gray_f32, dst_mat, CV_32F, 1, 0, scale, delta, cv::BorderTypes::BORDER_REFLECT);
    // BORDER_REFLECT in cv stands for total reflect with the outer line
    cv::imwrite("../data/output/scharr_gray.png", dst_mat * 255);

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
    std::cout << "Scharr Deriv X with reflect-total border type: " << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Scharr X deriv with replicate border type", "[image_scharr]") {
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
    smartmore::cudaop::ImageScharr<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32,
                                   smartmore::cudaop::BorderType::kReplicate>(src_dev, dst_dev, height, width, 1, 0,
                                                                              scale, delta);

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Macro CV_32F --- 5
    cv::Scharr(src_gray_f32, dst_mat, CV_32F, 1, 0, scale, delta, cv::BorderTypes::BORDER_REPLICATE);
    cv::imwrite("../data/output/scharr_gray.png", dst_mat * 255);

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
    std::cout << "Scharr Deriv X with replicate border type: " << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Scharr X deriv with constant(0.0) border type", "[image_scharr]") {
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
    smartmore::cudaop::ImageScharr<smartmore::cudaop::DataType::kFloat32, smartmore::cudaop::DataType::kFloat32,
                                   smartmore::cudaop::BorderType::kConstant>(src_dev, dst_dev, height, width, 1, 0,
                                                                             scale, delta);

    std::vector<float> result_cuda_f32(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda_f32.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Macro CV_32F --- 5
    cv::Scharr(src_gray_f32, dst_mat, CV_32F, 1, 0, scale, delta, cv::BorderTypes::BORDER_CONSTANT);
    cv::imwrite("../data/output/scharr_gray.png", dst_mat * 255);

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
    std::cout << "Scharr Deriv X with constant border type: " << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Scharr X deriv Int8", "[image_scharr]") {
    int height = 1080, width = 1920;

    std::vector<unsigned char> vec_gray(height * width);
    smartmore::RandomInt8Vector(vec_gray);

    cv::Mat src_gray(height, width, CV_8UC1, vec_gray.data());
    cv::Mat dst_mat(height, width, CV_32FC1);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&src_dev, height * width * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dst_dev, height * width * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(src_dev, src_gray.data, height * width * sizeof(char), cudaMemcpyHostToDevice));
    // x deriv Int8
    smartmore::cudaop::ImageScharr<smartmore::cudaop::DataType::kInt8, smartmore::cudaop::DataType::kFloat32>(
        src_dev, dst_dev, height, width);

    std::vector<float> result_cuda(height * width);
    CUDA_CHECK(cudaMemcpy(result_cuda.data(), dst_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost));

    cv::Scharr(src_gray, dst_mat, CV_32F, 1, 0);  // x deriv
    cv::imwrite("../data/output/scharr_gray.png", dst_mat);

    // compare cuda with opencv
    float max_diff = 0;
    float sum_diff = 0.0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float cur_diff = fabs(result_cuda[i * width + j] - (dst_mat.at<float>(i, j)));
            max_diff = fmax(max_diff, cur_diff);
            sum_diff += cur_diff;
        }
    }
    std::cout << "Scharr Deriv X Int8: " << std::endl;
    std::cout << "max diff: " << max_diff << std::endl;
    std::cout << "mean diff: " << sum_diff / (height * width) << std::endl;
    std::cout << std::endl;
    REQUIRE(max_diff < 0.001);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}
