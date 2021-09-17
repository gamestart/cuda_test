/*******************************************************************************
 *  FILENAME:      types.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Tuesday March 9th 2021
 *
 *  LAST MODIFIED: Monday, September 13th 2021, 3:42:33 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TYPES_H__
#define __SMARTMORE_CUDAOP_TYPES_H__

namespace smartmore {
namespace cudaop {
enum class ImageType {
    kBGRA_CHW,
    kBGRA_HWC,
    kBGR_CHW,
    kBGR_HWC,

    kRGBA_CHW,
    kRGBA_HWC,
    kRGB_CHW,
    kRGB_HWC,

    kYUV_NV12,
    kYUV_UYVY,
    kYUV_I420,
    kYuv422p,

    // Single channel image
    kGRAY,
};

enum class DataType {
    kInt8,
    kFloat32,
    kHalf,
};

enum class YUVFormula {
    kBT601,
    kBT709,
    kYCrCb,
};

enum class EinsumType {
    // Input vector<int> parameters order:bnchw
    kBNHW_BNCHW_To_BCHW,
    // Input vector<int> parameters order:bchw:
    kBCHW_BCHW_To_BHW,
};

enum class ResizeScaleType {
    kStretch,
    kSelfAdapt,
};

enum class ResizeAlgoType {
    kBilinear,
    kNearest,
    kBicubic,
};

enum class ReduceType {
    kSum,
    kMax,
    kMin,
};

enum class BorderType {
    kReflect,      /* `gfedcb | abcdefgh | gfedcba` */
    kReflectTotal, /* `fedcba | abcdefgh | hgfedcb` */
    kReplicate,    /* `aaaaaa | abcdefgh | hhhhhhh` */
    kConstant,     /* `iiiiii | abcdefgh | iiiiiii` with some specified `i` */
};

enum class FlipType {
    kHor_Vert = 0,
    kVert,
    kHor,
};

enum class ThreshType {
    kThresh_Binary,
    kThresh_Binary_INV,
    kThresh_Trunc,
    kThresh_ToZero,
    kThresh_ToZero_INV,
};

enum class CopyBorderType {
    kBorder_Replicate,
    kBorder_Reflect,
    kBorder_Reflect_101,
    kBorder_Warp,
    kBorder_Constant,
};

struct Size {
    int width;
    int height;
};
struct Point {
    int x;
    int y;
};

struct Rect {
    Point topleft;
    Size size;
};
};  // namespace cudaop
}  // namespace smartmore

#endif
