# 算子介绍

## 图像格式转换

支持各种图像格式之间的转换。

```c++
template <ImageType input_image_type, DataType input_data_type,
          ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula = YUVFormula::kBT601>
void ImageConvert(void *src, void *dst, int in_h, int in_w);

template <ImageType input_image_type, DataType input_data_type,
          ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula = YUVFormula::kBT601>
void ImageConvertAsync(void *src, void *dst, int in_h, int in_w, cudaStream_t stream);
```

- `ImageType input_image_type`
  - 输入图像的颜色空间和内存排布
- `DataType input_data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `ImageType output_image_type`
  - 输出图像的颜色空间和内存排布
- `DataType output_data_type`
  - 输出图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `YUVFormula yuv_formula`
  - 指定不同yuv转换公式的枚举值
- `void *src`
  - 指向输入图像内存地址的cuda设备指针。
- `void *dst`
  - 指向预先分配好的内存空间的cuda设备指针，用于存放转换结果。
- `int in_h`
  - 输入图像的高度，单位为像素的个数
- `int in_w`
  - 输入图像的宽度，单位为像素的个数

- 仅支持YUV与RGB类型图片的转换，不支持灰度图。
  > OpenCV中没有不同采样比例(如YUV_UYVY --> YUV_I420)的YUV图像转换, 不建议使用.

## 爱因斯坦求和

对两个张量进行爱因斯坦求和运算，具体请参考[这里](https://en.wikipedia.org/wiki/Einstein_notation)

```c++
template <EinsumType einsum_type>
void Einsum(float *src1, float *src2, float *dst,
            const std::vector<size_t> &dims);

template <EinsumType einsum_type>
void EinsumAsync(float *src1, float *src2, float *dst,
                const std::vector<size_t> &dims, cudaStream_t stream);
```

- `EinsumType einsum_type`
  - 求和运算的类型，和输入张量的维度有关
- `float *src1`
  - 指向第一个输入张量的cuda设备指针
- `float *src2`
  - 指向第二个输入张量的cuda设备指针
- `float *dst`
  - 指向输出位置的cuda设备张量，需要预先分配内存
- `const std::vector<size_t> &dims`
  - 输入张量的维度

## 图像通道拆分

将多通道的图像拆分为多个单通道的图像

```c++
template <ImageType image_type, DataType input_data_type,
          DataType output_data_type>
void ImageChannelSplit(void *src, const std::vector<void *> &dsts,
                       const int in_h, const int in_w);

template <ImageType image_type, DataType input_data_type,
          DataType output_data_type>
void ImageChannelSplitAsync(void *src, const std::vector<void *> &dsts,
                            const int in_h, const int in_w, cudaStream_t stream)
```

- `ImageType image_type`
  - 表示输入图像类型的枚举值
- `DataType input_data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `DataType output_data_type`
  - 输出图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `void *src`
  - 指向输入图像的起始地址的cuda设备指针
- `const std::vector<void *> &dsts`
  - 一个包含指向输出结果的cuda设备指针的vector，数量应该与输入图像的通道数相同
- `int in_h`
  - 输入图像的高度，单位为像素的个数
- `int in_w`
  - 输入图像的宽度，单位为像素的个数

## 图像通道合并

将多个单通道的数据合并成图片

```c++
template <ImageType image_type, DataType input_data_type, DataType output_data_type>
void ImageChannelMerge(const std::vector<void *> &srcs, void *dst,
                       const int in_h, const int in_w)

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
void ImageChannelMergeAsync(const std::vector<void *> &srcs, void *dst,
                            const int in_h, const int in_w, cudaStream_t stream)
```

- `ImageType image_type`
  - 表示输入图像类型的枚举值
- `DataType input_data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `DataType output_data_type`
  - 输出图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `const std::vector<void *> &src`
  - 一个包含指向输出结果的cuda设备指针的vector，数量应该与输出图像的通道数相同
- `void *dsts`
  - 指向输出图像的起始地址的cuda设备指针
- `int in_h`
  - 输入图像的高度，单位为像素的个数
- `int in_w`
  - 输入图像的宽度，单位为像素的个数

## 图片缩放

```c++
template <ImageType src_image_type, DataType input_data_type,
          DataType output_data_type, ResizeScaleType scale_type,
          ResizeAlgoType algo_type>
void ImageResize(void *src, void *dst, int in_h, int in_w,
                 int out_h, int out_w);

template <ImageType src_image_type, DataType input_data_type,
          DataType output_data_type, ResizeScaleType scale_type,
          ResizeAlgoType algo_type>
void ImageResizeAsync(void *src, void *dst, int in_h, int in_w,
                      int out_h, int out_w, cudaStream_t stream)
```

- `ImageType src_image_type`
  - 代表图像类型的枚举值
- `DataType input_data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `DataType output_data_type`
  - 输出图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `ResizeScaleType scale_type`
  - 代表缩放类型的枚举值，可以是拉伸或自适应
- `ResizeAlgoType algo_type`
  - 代表缩放算法的枚举值，默认为双线性插值
- `void *src`
  - 指向输入图片的cuda设备指针
- `void *dst`
  - 指向输出的cuda设备指针
- `int in_h`
  - 输入图片的高度
- `int in_w`
  - 输入图片的宽度
- `int out_h`
  - 输出图片的高度
- `int out_w`
  - 输出图片的宽度

## 横向扫描效果

输出一根横向扫描的竖线，竖线左右可以输出不同图像，可用于对比超分效果。
可以支持图像类型转换。

```c++
template <ImageType input_image_type, DataType input_data_type,
          ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula = YUVFormula::kBT601>
void ImageHorizonScanSpecialEffect(
    void *src_l, void *src_r, void *dst, int in_h, int in_w, int scan_x);

template <ImageType input_image_type, DataType input_data_type,
          ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula = YUVFormula::kBT601>
void ImageHorizonScanSpecialEffectAsync(
    void *src_l, void *src_r, void *dst, int in_h, int in_w, int scan_x, cudaStream_t stream)
```

- `ImageType input_image_type`
  - 输入图像`src_l`与`src_r`的颜色空间和内存排布
- `DataType input_data_type`
  - 输入图像`src_l`与`src_r`数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `ImageType output_image_type`
  - 输出图像的颜色空间和内存排布
- `DataType output_data_type`
  - 输出图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `YUVFormula yuv_formula`
  - 指定不同yuv转换公式的枚举值
- `void *src_l`
  - 指向左侧图像的cuda设备指针
- `void *src_r`
  - 指向右侧图像的cuda设备指针
- `void *dst`
  - 指向输出图像的cuda设备指针
- `int in_h`
  - 输入图像的高度，单位为像素的个数
- `int in_w`
  - 输入图像的宽度，单位为像素的个数
- `int scan_x`
  - 竖线的横坐标，单位为像素的个数

## Concat算子

```c++
template <typename DataType>
void Concat(DataType *lhs, DataType *rhs, DataType *dst, long axis,
            const std::vector<size_t> &lhs_shape,
            const std::vector<size_t> &rhs_shape);

template <typename DataType>
void ConcatAsync(DataType *lhs, DataType *rhs, DataType *dst, long axis,
                 const std::vector<size_t> &lhs_shape,
                 const std::vector<size_t> &rhs_shape, cudaStream_t stream)
```

- `DataType`
  - 表示输入输出的数据类型
- `DataType *lhs, DataType *rhs`
  - 输入
- `DataType *dst`
  - 输出
- `long axis`
  - concat的维度
- `const std::vector<size_t> &lhs_shape`
  - `lhs`形状
- `const std::vector<size_t> &rhs_shape`
  - `rhs`形状

## DepthToSpace算子

```c++
template<typename DataType, bool is_NCHW, unsigned int block_size,
    unsigned int in_N, unsigned int in_C, unsigned int in_H, unsigned int in_W>
void DepthToSpace(DataType * src, DataType * dst)

template <typename DataType, bool is_NCHW, unsigned int block_size,
          unsigned int in_N, unsigned int in_C, unsigned int in_H, unsigned int in_W>
void DepthToSpaceAsync(DataType *src, DataType *dst, cudaStream_t stream)
```

- `DataType`
  - 表示输入输出的数据类型
- `bool is_NCHW`
  - 表示张量的内存排布顺序
  - true表示`NCHW`
  - false表示`NHWC`
- `unsigned int block_size`
  - channel坍缩后的block的边长
  - channel的大小必须是`block_size`的平方的倍数
- `unsigned int in_N, unsigned int in_C, unsigned int in_H, unsigned int in_W`
  - 输入张量的维度
- `DataType * src, DataType * dst`
  - 输入输出的buffer指针

## Reduce(并行归约)

```c++
template <DataType data_type, ReduceType reduce_type>
  void Reduce(void *input_data, int lenth, void *output_data)

template <DataType data_type, ReduceType reduce_type>
  void ReduceAsync(void *input_data, int lenth, void *output_data, cudaStream_t stream)
```

- `DataType data_type`
  - 输入数据（input_data）类型的枚举值，比如单精度浮点、半精度浮点、整型
- `ReduceType reduce_type`
  - 规约方式，目前支持求和
- `input_data`
  - 输入数组首地址指针
- `lenth`
  - 输入数组长度（元素个数）
- `output_data`
  - 输出地址，应当存有一个初始值

- 不支持输入kInt8与kSum的组合（容易导致数据溢出）
- Half类型的Reduce操作需要满足`compute capability>7.0`

## Clamp算子

```c++
template <DataType data_type>
  void Clamp(void *src, unsigned int length, float lower_bound, float upper_bound)
  
template <DataType data_type>
  void ClampAsync(void *src, unsigned int length, float lower_bound, float upper_bound, cudaStream_t stream)
```

- `DataType data_type`
  - 输入数据(src)类型的枚举值，比如单精度浮点、半精度浮点、整型
- `void *src`
  - 输入数组首地址的设备指针
- `unsigned int lenth`
  - 输入数组长度（元素个数）
- `float lower_bound`
  - clamp的下界
- `float upper_bound`
  - clamp的上界

## 类型转换算子

```c++
template <DataType input_data_type, DataType output_data_type>
void DataTypeConvert(void *input_data, void *output_data, int lenth)

template <DataType input_data_type, DataType output_data_type>
void DataTypeConvertAsync(void *input_data, void *output_data, int lenth,
                          cudaStream_t stream)
```

- `DataType input_data_type`
  - 输入数据(input_data)类型的枚举值，比如单精度浮点、半精度浮点、整型
- `DataType output_data_type`
  - 输出数据(input_data)类型的枚举值，比如单精度浮点、半精度浮点、整型

## 均值滤波

```c++
template <ImageType image_type, DataType data_type, BorderType border_type,
          int kernel_h,int kernel_w>
void MeanFilter(void *in, void *out, int in_h, int in_w)
```

- `ImageType image_type`
  - 输入图片的内存，目前支持灰度图
- `DataType data_type`
  - 输入图片的数据类型，目前支持`Int8`
- `BorderType border_type`
  - 边界处理方式，目前支持`kReflect`

## 中值滤波

```c++
template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_size>
void MedianFilter(void *src, void *dst, int in_h, int in_w)
```

- `ImageType image_type`
  - 输入图片的内存，目前支持灰度图
- `DataType data_type`
  - 输入图片的数据类型，目前支持`Int8`
- `BorderType border_type`
  - 边界处理方式，目前支持`kReplicate`
- `int kernel_size`
  - 滤波窗口的长度，必须是大于1的奇数

## 高斯滤波

```c++
template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_h, int kernel_w>
void ImageGaussianFilter(unsigned char *src, unsigned char *dst, int in_h, int in_w, float sigma_h, float sigma_w)
```

- `ImageType image_type`
  - 输入图片的内存，目前支持灰度图
- `DataType data_type`
  - 输入图片的数据类型，目前支持`Int8`
- `BorderType border_type`
  - 边界处理方式，目前支持`kReplicate`
- `int kernel_h`
  - 滤波窗口的高度，必须是大于1的奇数
- `int kernel_w`
  - 滤波窗口的宽度，必须是大于1的奇数
- `float sigma_h`
  - 纵向滤波的标准差，值越大模糊力度越大
- `float sigma_w`
  - 横向滤波的标准差，值越大模糊力度越大

## 金字塔下采样

```c++

template <ImageType image_type, DataType data_type,BorderType border_type>
void ImagePyrDown(unsigned char *src, unsigned char *dst, int in_h, int in_w)
```

- `ImageType image_type`
  - 输入图片的内存，目前支持灰度图
- `DataType data_type`
  - 输入图片的数据类型，目前支持`Int8`
- `BorderType border_type`
  - 边界处理方式，目前支持`kReplicate`

## 图像ROI拷贝

模拟OpenCV的图像拷贝操作：cv::Mat::copyTo()

```c++
template <ImageType image_type, DataType input_data_type>
void ImageCopy(void *src, const Size &src_size, const Rect &copy_rect, void *dst, const Size &dst_size, const Point &dst_rect_tl)

template <ImageType image_type, DataType input_data_type>
void ImageCopyAsync(void *src, const Size &src_size, const Rect &copy_rect, void *dst, const Size &dst_size, const Point &dst_rect_tl, cudaStream_t stream)
```

- `ImageType image_type`
  - 源图像的类型
- `DataType input_data_type`
  - 源图像的数据类型
- `void *src`
  - 源图像数据
- `const Size &src_size`
  - 源图像尺寸
- `const Rect &copy_rect`
  - 需要拷贝的区域 (ROI)
- `void *dst`
  - 目标图像数据
- `const Size &dst_size`
  - 目标图像尺寸
- `const Point &dst_rect_tl`
  - 目标区域左上点

## 减均值除方差

  ```c++
  template <DataType data_type>
  void MeanNormalization(void *src, void *dst, unsigned int length,const float mean, const float variance)

  template <DataType data_type>
  void MeanNormalizationAsync(void *src, void *dst, unsigned int length,const float mean, const float variance,cudaStream_t stream)
  ```

- `DataType data_type`
  - 输入数据类型的枚举值，目前支持Float32 以及Half类型
- `void *src`
  - 输入数组首地址指针
- `void *dst`
  - 输出地址
- `unsigned int length`
  - 输入数组长度(元素个数)
- `const float mean`
  - 输入数据的均值
- `const float variance`
  - 输入数据的方差

## 图片翻转

```c++
template <DataType data_type, ImageType image_type, FlipType flip_type>
void ImageFlip(void *src, void *dst, const Size src_size)

template <DataType data_type, ImageType image_type, FlipType flip_type>
void ImageFlipAsync(void *src, void *dst, const Size src_size, cudaStream_t stream)
```

- `ImageType image_type`
  - 代表图像类型的枚举值
- `DataType data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `FlipType flip_type`
  - 翻转类型枚举，比如水平翻转、垂直翻转、水平垂直翻转
- `void *src`
  - 指向输入图片的cuda设备指针
- `void *dst`
  - 指向输出的cuda设备指针
- `const Size src_size`
  - 输入图片的尺寸


## 图像指定大小批量切片

  超分算法中，常因为显存资源限制，需要将图像分成较小的块再送入超分算法，此接口提供一种按指定尺寸，指定overlay大小的方式批量完成图像切片操作

  ```c++
  template <ImageType image_type, DataType input_data_type>
  void ImageSlice(void *src, const Size &src_size, const Size &slice_size, 
                std::vector<void *> dsts, std::vector<Rect> &slice_rts, int overlay_size)
  ```

- `ImageType image_type`
  - 源图像的类型
- `DataType input_data_type`
  - 源图像的数据类型
- `void *src`
  - 源图像数据
- `const Size &src_size`
  - 源图像尺寸
- `const Size &slice_size`
  - 切片大小
- `std::vector<void *> dsts`
  - 切片结果图像数据序列（行优先）
- `std::vector<Rect> &slice_rts`
  - 记录在源图像中，切片的有效区域，便于处理完成后，拼接复原
- `int overlay_size`
  - 切片之间的overlay大小，单位为像素
  
## 图像切片拼接，恢复原图
此接口一般与ImageSlice成对出现，合并ImageSlice输出的结果，恢复原始图像
```c++
template <ImageType image_type, DataType input_data_type>
void ImageStitching(const std::vector<void *> slices, void *dst, const Size &dst_size,
                     const Size &slice_size, std::vector<Rect> &sliced_rects, int overlay_size)
```
- `ImageType image_type`
  - 源图像的类型
- `DataType input_data_type`
  - 源图像的数据类型
- `const std::vector<void *> slices`
  - 切片图像数据
- `void *dst`
  - 目标图像尺寸
- `const Size &dst_size`
  - 目标图像尺寸
- `const Size &slice_size`
  - 切片大小
- `std::vector<Rect> &sliced_rects`
  - 记录在源图像中，切片的有效区域，便于处理完成后，拼接复原
- `int overlay_size`
  - 切片之间的overlay大小，单位为像素

## 阈值化

```c++
template <DataType data_type, ThreshType thresh_type>
        void Threshold(void *src, void *dst, unsigned int length, double thresh, double maxval)
template <DataType data_type, ThreshType thresh_type>
        void ThresholdAsync(void *src, void *dst, unsigned int length, double thresh,
                            double maxval, cudaStream_t stream)
```

- `DataType data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `ThreshType thresh_type`
  - 阈值化类型枚举
- `void *src`
  - 指向输入图片的cuda设备指针
- `void *dst`
  - 指向输出的cuda设备指针
- `unsigned int length`
  - 输入数组长度(元素个数)
- `double thresh`
  - 阈值具体值
- `double maxval`
  - 当thresh_type取kThresh_Binary或者kThresh_Binary_INV时的最大取值
  
## sqrt算子

```c++
template <DataType data_type>
        void Sqrt(void *in, void *out, int length)
template <DataType data_type>
        void SqrtAsync(void *in, void *out, int length, cudaStream_t stream)
```

- `DataType data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `void *in`
  - 指向输入图片的cuda设备指针
- `void *out`
  - 指向输出的cuda设备指针
- `int length`
  - 输入数组长度(元素个数)

## 图像线性融合
```c++
template <ImageType image_type, DataType input_data_type>
void ImageAddWeighted(void *src1, float alpha, void *src2, float beta, float gamma, const Size &src_size, void *dst)

template <ImageType image_type, DataType input_data_type>
void ImageAddWeightedAsync(void *src1, float alpha, void *src2, float beta, float gamma, const Size &src_size, void *dst, cudaStream_t stream)
```
- `ImageType image_type`
  - 源图像的类型
- `DataType input_data_type`
  - 源图像的数据类型
- `void *src1`
  - 待融合图像一
- `float alpha`
  - 图像一融合权重alpha[0,1]
- `void *src2`
  - 待融合图像二，必须与图像一类型、通道数、大小一致
- `float beta`
  - 图像二融合权重beta,一般 `beta = 1 - alpha`
- `float gamma`
  - 融合求和后的偏移，一般置零
- `const Size &src_size`
  - 融合图像大尺寸
- `void *dst`
  - 融合图像输出结果，必须与图像一类型、通道数、大小一致，与可以直接是图像一或者图像二

## 设置边界框copymakeborder

```c++
template <DataType data_type, ImageType image_type, CopyBorderType copyborder_type>
        void ImageCopyMakeBorder(void *src, void *dst, unsigned int height,
                                 unsigned int width, int top, int bottom,
                                 int left, int right, float value = 0.f)
template <DataType data_type, ImageType image_type, CopyBorderType copyborder_type>
        void ImageCopyMakeBorderAsync(void *src, void *dst, unsigned int height,
                                      unsigned int width, int top, int bottom,
                                      int left, int right, float value = 0.f,
                                      cudaStream_t stream = 0)
```

- `DataType data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点、整型
- `ImageType image_type`
  - 输入图像类型的枚举值
- `CopyBorderType copyborder_type`
  - 添加边界的边界的类型
- `void *src`
  - 指向输入图片的cuda设备指针
- `void *dst`
  - 指向输出的cuda设备指针
- `unsigned int height`
  - 输入图像的高度
- `unsigned int width`
  - 输入图像的宽度
- `int top`
  - 设置图像的上边界尺寸
- `int bottom`
  - 设置图像的下边界尺寸
- `int left`
  - 设置图像的左边界尺寸
- `int right`
  - 设置图像的右边界尺寸
- `float value`
  - 当设置边界的类型为kBorder_Constant时候的固定像素值

## 图像alpha融合
前景图像按alpha掩码融合到背景图像，典型应用：打水印

前景图像、背景图像图像类型，数据类型、通道数需要一致
```c++
template <ImageType image_type, DataType input_data_type>
void ImageAlphaBlend(void *foreground, void *background, float *alpha_mask,
                   const Size &background_size, const Rect &blend_roi)

template <ImageType image_type, DataType input_data_type>
void ImageAlphaBlendAsync(void *foreground, void *background, float *alpha_mask, 
                  const Size &background_size, const Rect &blend_roi, cudaStream_t stream)
```
- `ImageType image_type`
  - 图像类型
- `DataType input_data_type`
  - 图像数据类型, 需要为float类型
- `void *foreground`
  - 前景图像
- `void *background`
  - 背景图像
- `float *alpha_mask`
  - alpha 掩码，size需要与前景图像相同，且值范围 [0, 1]
- `const Size &background_size`
  - 背景图尺寸，需要不小于前景图像尺寸
- `const Rect &blend_roi`
  - 前景图像融合到背景图像的目标ROI

## 最小外接矩形算子

```c++
template <DataType data_type>
        void MinAreaRect(void *in, void *out, void *g_data, int n)
template <DataType data_type>
        void MinAreaRectAsync(void *in, void *out, void *g_data, int n, cudaStream_t stream)
```

- `DataType data_type`
  - 输入图像数据类型的枚举值，比如单精度浮点、半精度浮点
- `void *in`
  - 指向输入图片的cuda设备指针
- `void *out`
  - 指向输出的cuda设备指针
- `void *g_data`
  - 规约最后结果的中间数组
- `int n`
  - 点集中点的个数

## Scharr 算子

  用Scharr 运算参数来计算图像的x **或** y 方向的一阶导数

  相比Sobel 算子的默认filter, Scharr 更为精确, 其filter 为:

  ```c++
  [-3,  0, 3]
  [-10, 0, 10]
  [-3,  0, 3]
  ```

  ```c++
  template <DataType input_data_type,
                  DataType output_data_type,
                  BorderType border_type = BorderType::kReflect>
        void ImageScharr(void *src, void *dst,
                         int in_h, int in_w,
                         int dx = 1, int dy = 0,
                         double scale = 1.0, double delta = 0.0);

  template <DataType input_data_type,
                  DataType output_data_type,
                  BorderType border_type = BorderType::kReflect>
        void ImageScharrAsync(void *src, void *dst,
                              int in_h, int in_w,
                              cudaStream_t stream,
                              int dx = 1, int dy = 0,
                              double scale = 1.0, double delta = 0.0);
  ```

- 与OpenCV 保持一致, 每次调用仅能处理单通道(gray) 图像
- 模板参数`data_type` 为输入和输出的图像数据类型
  > 输入int8图像时, 为防止结果溢出等, 输出类型不能是int8
  > 建议输出数据类型精度不低于输入数据
- 模板参数`border_type` 可选, 默认采用反射方式, 新增了*全反射* 和*常量填充* 的边界处理策略
  > OpenCV的常量填充策略仅支持填充0, 这里与OpenCV保持一致

- `src` 和`dst` 分别是输入和输出的设备指针
- `in_h` 和`in_w` 分别是输入图像的高和宽
- `dx` 和`dy` 分别用来指定水平或竖直方向上的一阶导数, 需满足`dx >= 0 && dy >= 0 && dx+dy == 1`
- `scale` 和`delta` 是计算导数后的乘数与累加项
- `stream` 为异步接口中的指定流

## Transpose

  ```c++
   template <DataType data_type>
        void Transpose(void *src, void *dst,
                       int in_h, int in_w);

  template <DataType data_type>
        void TransposeAsync(void *src, void *dst,
                            int in_h, int in_w,
                            cudaStream_t stream);
  ```

- 模板参数`data_type` 为输入和输出的图像数据类型

- `src` 和`dst` 分别是输入和输出的设备指针
- `in_h` 和`in_w` 分别是输入图像的高和宽
- `stream` 为异步接口中的指定流

## Sobel算子

  ```c++
  template <DataType input_data_type,
                  DataType output_data_type,
                  int ksize = 3,
                  BorderType border_type = BorderType::kReflect>
  void ImageSobel(void *src,
                        void *dst,
                        int in_h, int in_w,
                        int dx = 1, int dy = 0,
                        double scale = 1.0, double delta = 0.0);

  template <DataType input_data_type,
            DataType output_data_type,
            int ksize = 3,
            BorderType border_type = BorderType::kReflect>
  void ImageSobelAsync(void *src,
                             void *dst,
                             int in_h, int in_w,
                             cudaStream_t stream,
                             int dx = 1, int dy = 0,
                             double scale = 1.0, double delta = 0.0);
  ```

- 与OpenCV 保持一致, 每次调用仅能处理单通道(gray) 图像
- 模板参数`data_type` 为输入和输出的图像数据类型
  > 输入int8图像时, 为防止结果溢出等, 输出类型不能是int8
  > 建议输出数据类型精度不低于输入数据
- ksize 与OpenCV 保持一致, 可选3或-1, -1代表Scharr
- 模板参数`border_type` 可选, OpenCV的常量填充策略仅支持填充0, 这里与OpenCV保持一致

- `src` 和`dst` 分别是输入和输出的设备指针
- `in_h` 和`in_w` 分别是输入图像的高和宽
- `dx` 和`dy` 目前仅支持最为常见的一阶导数
- `scale` 和`delta` 是计算导数后的乘数与累加项
- `stream` 为异步接口中的指定流

## ArcLength

  ```c++
  template <typename T, bool is_closed>
      void ArcLength(void *curve, int num_point, float *arc_length);

  template <typename T, bool is_closed>
      void ArcLengthAsync(void *curve, int num_point, float *arc_length, cudaStream_t stream);
  ```

- `typename T`, 与OpenCV保持一致, 仅支持int 和float 类型的输入数据
- `bool is_closed`, 标记该段弧是否封闭
- `void *curve` 为曲线坐标的首地址
- `int num_point` 是 **平面坐标点** 的个数(curve数据个数的一半)
- `float *arc_length` 弧长结果记录地址
- `stream` 为异步接口中的指定流

Note: 与工程目前使用的OpenCV 4.2对齐, int坐标相较于float坐标误差较大, 在坐标点规模上万时, 弧长计算会出现1, 2 之类的误差
