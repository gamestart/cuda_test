/*******************************************************************************
 *  FILENAME:      imgslice.h
 *
 *  AUTHORS:       Liang Jia    START DATE: Friday July 23rd 2021
 *
 *  LAST MODIFIED: Thursday, August 12th 2021, 10:40:30 am
 *
 *  CONTACT:       jia.liang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGSLICE_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGSLICE_H__
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>
#include <cudaop/image/imgcopy.h>

#include <iostream>
#include <vector>

namespace smartmore {
namespace cudaop {
template <ImageType image_type, DataType input_data_type>
void CudaImageSlice(void *src, const Size &src_size, const Size &slice_size, std::vector<void *> dsts,
                    std::vector<Rect> &sliced_rects, int overlay_size) {
    // slice roi = effective roi + overlay
    CUDAOP_ASSERT_TRUE(src != nullptr);
    int effective_width = slice_size.width - 2 * overlay_size;
    int effective_height = slice_size.height - 2 * overlay_size;
    CUDAOP_ASSERT_TRUE(effective_width > 0);
    CUDAOP_ASSERT_TRUE(effective_height > 0);
    int rows = std::ceil(src_size.height * 1.0f / effective_height);
    int cols = std::ceil(src_size.width * 1.0f / effective_width);
    const int dsts_size = rows * cols;
    CUDAOP_ASSERT_TRUE(dsts.size() == dsts_size);

    sliced_rects.resize(dsts_size);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int index = r * cols + c;
            Point pt = Point{effective_width * c, effective_height * r};
            Size rect_size = Size{std::min(src_size.width - pt.x, effective_width),
                                  std::min(src_size.height - pt.y, effective_height)};
            sliced_rects[index] = Rect{pt, rect_size};

            Rect copy_rect =
                Rect{Point{pt.x - overlay_size, pt.y - overlay_size}, Size{slice_size.width, slice_size.height}};
            Point dst_tl = Point{0, 0};
            // boundary treatment
            if (copy_rect.topleft.x < 0) {
                copy_rect.topleft.x = 0;
                copy_rect.size.width -= overlay_size;
                dst_tl.x += overlay_size;
            }
            if (copy_rect.topleft.y < 0) {
                copy_rect.topleft.y = 0;
                copy_rect.size.height -= overlay_size;
                dst_tl.y += overlay_size;
            }
            if (copy_rect.size.width + copy_rect.topleft.x > src_size.width) {
                copy_rect.size.width = src_size.width - copy_rect.topleft.x;
            }
            if (copy_rect.size.height + copy_rect.topleft.y > src_size.height) {
                copy_rect.size.height = src_size.height - copy_rect.topleft.y;
            }

            ImageCopy<image_type, input_data_type>(src, src_size, copy_rect, dsts[r * cols + c], slice_size, dst_tl);
        }
    }
}

/**
 * @brief
 *
 * @tparam image_type image type
 * @tparam input_data_type data type
 * @param src input image data buffer
 * @param src_size input image size
 * @param slice_size slice size
 * @param dsts output slices
 * @param sliced_rects slice rects from src
 * @param overlay_size slice overlay size
 */
template <ImageType image_type, DataType input_data_type>
void ImageSlice(void *src, const Size &src_size, const Size &slice_size, std::vector<void *> dsts,
                std::vector<Rect> &sliced_rects, int overlay_size) {
    CudaImageSlice<image_type, input_data_type>(src, src_size, slice_size, dsts, sliced_rects, overlay_size);
    return;
}

}  // namespace cudaop
}  // namespace smartmore

#endif /* __SMARTMORE_CUDAOP_IMAGE_IMGSTITCHING_H__ */
