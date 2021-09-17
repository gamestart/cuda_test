/*******************************************************************************
 *  FILENAME:      test_image_minarearect.cu
 *
 *  AUTHORS:       Chen Fting    START DATE: Tuesday August 17th 2021
 *
 *  LAST MODIFIED: Thursday, August 19th 2021, 2:20:57 pm
 *
 *  CONTACT:       fting.chen@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("MinAreaRect", "[minarearect]") {
    int height = 2000, width = 2;
    std::vector<float> input_data(height * width);
    smartmore::RandomFloatVector(input_data);
    std::vector<cv::Point2f> points;
    for (int i = 0; i < height; i++) {
        cv::Point2f point;
        point.x = input_data[i * 2];
        point.y = input_data[i * 2 + 1];
        points.push_back(point);
    }
    int size = (height * (height - 1) / 2 + 511) / 512 * 11;
    cv::RotatedRect box = cv::minAreaRect(points);
    cv::Mat expect;
    cv::boxPoints(box, expect);
    std::cout << expect << std::endl;
    std::vector<float> actual(8);
    void *input_device = nullptr, *output_device = nullptr, *g_data = nullptr;
    CUDA_CHECK(cudaMalloc(&input_device, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_device, 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_device, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("MinAreaRect");
        MinAreaRect<DataType::kFloat32>(input_device, output_device, g_data, height);
    }

    CUDA_CHECK(cudaMemcpy(&actual[0], output_device, 8 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; i++) {
        std::cout << actual[i * 2] << "," << actual[i * 2 + 1] << ";" << std::endl;
    }

    // for (int i = 0; i < actual.size(); i++)
    // {
    //     REQUIRE(fabs(actual[i] - exptct[i]) <= 0.0001f);
    // }

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}