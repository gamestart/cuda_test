/*******************************************************************************
 *  FILENAME:      test_image_slice.cu
 *
 *  AUTHORS:       Liang Jia    START DATE: Friday July 23rd 2021
 *
 *  LAST MODIFIED: Saturday, August 14th 2021, 3:27:08 pm
 *
 *  CONTACT:       jia.liang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace smartmore::cudaop;

TEST_CASE("ImageSlice", "[image_slice]") {
    const int src_h = 1080, src_w = 1920, src_c = 3;
    const int slice_h = 480, slice_w = 640;
    const int overlay_size = 10;
    auto slice_size = Size{slice_w, slice_h};
    int effective_width = slice_w - 2 * overlay_size;
    int effective_height = slice_h - 2 * overlay_size;
    int rows = std::ceil(src_h * 1.0f / effective_height);
    int cols = std::ceil(src_w * 1.0f / effective_width);
    int slices_count = rows * cols;
    void *src_device;
    std::vector<void *> dst_devices(slices_count);

    std::vector<float> input_data(src_h * src_w * src_c);
    smartmore::RandomFloatVector(input_data);

    cv::Mat src_f = cv::Mat(src_h, src_w, CV_32FC3, input_data.data());
    std::vector<cv::Mat> actual_mats, expet_mats;

    CUDA_CHECK(cudaMalloc(&src_device, sizeof(float) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMemcpy(src_device, src_f.data, sizeof(float) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));

    for (size_t i = 0; i < slices_count; i++) {
        CUDA_CHECK(cudaMalloc(&dst_devices[i], sizeof(float) * slice_w * slice_h * src_f.channels()));
        CUDA_CHECK(cudaMemset(dst_devices[i], 0, sizeof(float) * slice_w * slice_h * src_f.channels()));
    }

    std::vector<Rect> sliced_rects;
    ImageSlice<ImageType::kBGR_HWC, DataType::kFloat32>(src_device, Size{src_w, src_h}, slice_size, dst_devices,
                                                        sliced_rects, overlay_size);

    REQUIRE(sliced_rects.size() - slices_count == 0);
    for (size_t i = 0; i < slices_count; i++) {
        cv::Mat dst_mat = cv::Mat::zeros(slice_h, slice_w, CV_32FC3);
        CUDA_CHECK(cudaMemcpy((void *)dst_mat.data, dst_devices[i],
                              sizeof(float) * slice_w * slice_h * src_f.channels(), ::cudaMemcpyDeviceToHost));
        actual_mats.push_back(std::move(dst_mat));
    }

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            int index = r * cols + c;
            cv::Mat expet_mat = cv::Mat::zeros(slice_h, slice_w, CV_32FC3);
            cv::Rect slice_rect(effective_width * c - overlay_size, effective_height * r - overlay_size, slice_w,
                                slice_h);
            cv::Point dst_tl(0, 0);
            if (slice_rect.x < 0) {
                slice_rect.x = 0;
                slice_rect.width -= overlay_size;
                dst_tl.x += overlay_size;
            }
            if (slice_rect.y < 0) {
                slice_rect.y = 0;
                slice_rect.height -= overlay_size;
                dst_tl.y += overlay_size;
            }
            if (slice_rect.width + slice_rect.x > src_w) {
                slice_rect.width = src_w - slice_rect.x;
            }
            if (slice_rect.height + slice_rect.y > src_h) {
                slice_rect.height = src_h - slice_rect.y;
            }
            src_f(slice_rect).copyTo(expet_mat(cv::Rect(dst_tl, slice_rect.size())));

            float max_diff = smartmore::CVMatMaxDiff(actual_mats[index], expet_mat);

            REQUIRE(max_diff < 0.0001);
        }
    }

    CUDA_CHECK_AND_FREE(src_device);
    for (size_t i = 0; i < slices_count; i++) {
        CUDA_CHECK_AND_FREE(dst_devices[i]);
    }
}