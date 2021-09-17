/*******************************************************************************
 *  FILENAME:      cudaop.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Tuesday March 9th 2021
 *
 *  LAST MODIFIED: Wednesday, September 15th 2021, 10:20:07 am
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_CUDAOP_H__
#define __SMARTMORE_CUDAOP_CUDAOP_H__

#include <cudaop/image/arclength.h>
#include <cudaop/image/imgaddweighted.h>
#include <cudaop/image/imgalphablend.h>
#include <cudaop/image/imgchannelmerge.h>
#include <cudaop/image/imgchannelsplit.h>
#include <cudaop/image/imgconvert.h>
#include <cudaop/image/imgcopy.h>
#include <cudaop/image/imgcopymakeborder.h>
#include <cudaop/image/imgflip.h>
#include <cudaop/image/imgguassianfilter.h>
#include <cudaop/image/imghorizonscan.h>
#include <cudaop/image/imgmeanfilter.h>
#include <cudaop/image/imgmedianfilter.h>
#include <cudaop/image/imgpyrdown.h>
#include <cudaop/image/imgresize.h>
#include <cudaop/image/imgscharr.h>
#include <cudaop/image/imgslice.h>
#include <cudaop/image/imgsobel.h>
#include <cudaop/image/imgsqrt.h>
#include <cudaop/image/imgstitching.h>
#include <cudaop/image/minarearect.h>
#include <cudaop/tensor/clamp.h>
#include <cudaop/tensor/concat.h>
#include <cudaop/tensor/datatypeconvert.h>
#include <cudaop/tensor/depth_to_space.h>
#include <cudaop/tensor/einsum.h>
#include <cudaop/tensor/meannormalization.h>
#include <cudaop/tensor/reduce.h>
#include <cudaop/tensor/threshold.h>
#include <cudaop/tensor/transpose.h>

#endif
