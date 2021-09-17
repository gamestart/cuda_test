/*******************************************************************************
 *  FILENAME:      imgstitching.h
 *
 *  AUTHORS:       Liang Jia    START DATE: Friday July 23rd 2021
 *
 *  LAST MODIFIED: Thursday, August 12th 2021, 10:40:24 am
 *
 *  CONTACT:       jia.liang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGSTITCHING_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGSTITCHING_H__
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>
#include <cudaop/image/imgcopy.h>

#include <iostream>
#include <vector>

namespace smartmore {
namespace cudaop {
template <ImageType image_type, DataType input_data_type>
void ImageStitching(const std::vector<void *> slices, void *dst, const Size &dst_size, const Size &slice_size,
                    std::vector<Rect> &sliced_rects, int overlay_size) {
    CUDAOP_ASSERT_TRUE(dst != nullptr);
    int effective_width = slice_size.width - 2 * overlay_size;
    int effective_height = slice_size.height - 2 * overlay_size;
    CUDAOP_ASSERT_TRUE(effective_width > 0);
    CUDAOP_ASSERT_TRUE(effective_height > 0);
    int rows = std::ceil(dst_size.height * 1.0f / effective_height);
    int cols = std::ceil(dst_size.width * 1.0f / effective_width);
    const int dsts_size = rows * cols;
    CUDAOP_ASSERT_TRUE(slices.size() == dsts_size);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int index = r * cols + c;
            Rect src_rect = Rect{Point{overlay_size, overlay_size}, sliced_rects[index].size};
            Point dst_tl = sliced_rects[index].topleft;
            ImageCopy<image_type, input_data_type>(slices[index], slice_size, src_rect, dst, dst_size, dst_tl);
        }
    }
}

}  // namespace cudaop
}  // namespace smartmore

#endif /* __SMARTMORE_CUDAOP_IMAGE_IMGSTITCHING_H__ */
