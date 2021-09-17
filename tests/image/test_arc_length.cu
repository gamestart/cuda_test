/*******************************************************************************
 *  FILENAME:      test_arc_length.cu
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Tuesday September 7th 2021
 *
 *  LAST MODIFIED: Monday, September 13th 2021, 10:42:35 am
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/image/arclength.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("Arc Length: float, closed == false", "[arcLength]") {
    int point_num = 10000;
    std::vector<float> vec_point(point_num * 2);
    smartmore::RandomFloatVector(vec_point);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&src_dev, vec_point.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_dev, sizeof(float)));  // use one elem store length of arc

    CUDA_CHECK(cudaMemcpy(src_dev, vec_point.data(), vec_point.size() * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Arc Length - CUDA: ");
        smartmore::cudaop::ArcLength<float, false>(src_dev, point_num, static_cast<float *>(dst_dev));
    }

    float arc_length_cuda = 0.0f;

    CUDA_CHECK(cudaMemcpy(&arc_length_cuda, dst_dev, sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<cv::Point2f> arc_point(point_num);
    for (int i = 0; i < point_num; i++) {
        cv::Point2f cur_point(vec_point[i * 2], vec_point[i * 2 + 1]);
        arc_point.emplace_back(cur_point);
    }

    float arc_length_opencv = 0.0f;
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Arc Length - OpenCV: ");
        arc_length_opencv = cv::arcLength(arc_point, false);
    }

    // compare cuda with opencv
    REQUIRE(fabs(arc_length_opencv - arc_length_cuda) < 0.001 * (1 + point_num / 5000));

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Arc Length: float, is_closed == true", "[arcLength]") {
    int point_num = 10000;
    std::vector<float> vec_point(point_num * 2);
    smartmore::RandomFloatVector(vec_point);

    void *src_dev = nullptr;
    void *dst_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&src_dev, vec_point.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_dev, sizeof(float)));  // use one elem store length of arc

    CUDA_CHECK(cudaMemcpy(src_dev, vec_point.data(), vec_point.size() * sizeof(float), cudaMemcpyHostToDevice));

    smartmore::cudaop::ArcLength<float, true>(src_dev, point_num, static_cast<float *>(dst_dev));

    float arc_length_cuda = 0.0f;

    CUDA_CHECK(cudaMemcpy(&arc_length_cuda, dst_dev, sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<cv::Point2f> arc_point(point_num);
    for (int i = 0; i < point_num; i++) {
        cv::Point2f cur_point(vec_point[i * 2], vec_point[i * 2 + 1]);
        arc_point.emplace_back(cur_point);
    }
    float arc_length_opencv = cv::arcLength(arc_point, true);

    // compare cuda with opencv
    REQUIRE(fabs(arc_length_opencv - arc_length_cuda) < 0.001 * (1 + point_num / 5000));

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Arc Length: int, closed == false", "[arcLength]") {
    int point_num = 10000;
    std::vector<int> vec_point(point_num * 2);

    srand((unsigned int)(time(NULL)));
    std::for_each(vec_point.begin(), vec_point.end(), [](int &i) { i = rand() % 1000; });

    void *src_dev = nullptr;
    void *dst_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&src_dev, vec_point.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dst_dev, sizeof(float)));  // use one elem store length of arc

    CUDA_CHECK(cudaMemcpy(src_dev, vec_point.data(), vec_point.size() * sizeof(int), cudaMemcpyHostToDevice));

    smartmore::cudaop::ArcLength<int, false>(src_dev, point_num, static_cast<float *>(dst_dev));

    float arc_length_cuda = 0.0f;

    CUDA_CHECK(cudaMemcpy(&arc_length_cuda, dst_dev, sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<cv::Point> arc_point(point_num);
    for (int i = 0; i < point_num; i++) {
        cv::Point cur_point(vec_point[i * 2], vec_point[i * 2 + 1]);
        arc_point.emplace_back(cur_point);
    }
    float arc_length_opencv = cv::arcLength(arc_point, false);

    // compare cuda with opencv
    REQUIRE(fabs(arc_length_opencv - arc_length_cuda) < 1 + point_num / 2000);
    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}

TEST_CASE("Arc Length: int, closed == true", "[arcLength]") {
    int point_num = 10000;
    std::vector<int> vec_point(point_num * 2);

    srand((unsigned int)(time(NULL)));
    std::for_each(vec_point.begin(), vec_point.end(), [](int &i) { i = rand() % 1000; });

    void *src_dev = nullptr;
    void *dst_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&src_dev, vec_point.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dst_dev, sizeof(float)));  // use one elem store length of arc

    CUDA_CHECK(cudaMemcpy(src_dev, vec_point.data(), vec_point.size() * sizeof(int), cudaMemcpyHostToDevice));

    smartmore::cudaop::ArcLength<int, true>(src_dev, point_num, static_cast<float *>(dst_dev));

    float arc_length_cuda = 0.0f;

    CUDA_CHECK(cudaMemcpy(&arc_length_cuda, dst_dev, sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<cv::Point> arc_point(point_num);
    for (int i = 0; i < point_num; i++) {
        cv::Point cur_point(vec_point[i * 2], vec_point[i * 2 + 1]);
        arc_point.emplace_back(cur_point);
    }
    float arc_length_opencv = cv::arcLength(arc_point, true);

    // compare cuda with opencv
    REQUIRE(fabs(arc_length_opencv - arc_length_cuda) < 1 + point_num / 2000);

    CUDA_CHECK_AND_FREE(src_dev);
    CUDA_CHECK_AND_FREE(dst_dev);
}
