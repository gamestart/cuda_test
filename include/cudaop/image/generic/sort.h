/*******************************************************************************
 *  FILENAME:      imgmedianfilter.h
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Saturday May 29th 2021
 *
 *  LAST MODIFIED: Tuesday, June 1st 2021, 1:48:43 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_SORT_H__
#define __SMARTMORE_CUDAOP_SORT_H__

#include <cudaop/generic/swap.h>

namespace smartmore {
namespace cudaop {
inline __device__ void SelectSort(unsigned char arr[], int l, int r) {
    for (int i = l; i < r; i++) {
        int min = i;
        for (int j = i + 1; j <= r; j++)
            if (arr[j] < arr[min]) min = j;
        Swap(arr[i], arr[min]);
    }
}

// https : //www.runoob.com/w3cnote/quick-sort.html
// recursive call, cannot inline
__device__ void QuickSort(unsigned char arr[], int l, int r) {
    if (l < r) {
        // Swap(arr[l], arr[(l + r) / 2]);
        int i = l, j = r, x = arr[l];
        while (i < j) {
            // find first number less than x, from right to left
            while (i < j && arr[j] >= x) j--;
            if (i < j) arr[i++] = arr[j];

            // find first number larger than x, from left to right
            while (i < j && arr[i] < x) i++;
            if (i < j) arr[j--] = arr[i];
        }
        arr[i] = x;
        // recursive call
        QuickSort(arr, l, i - 1);
        QuickSort(arr, i + 1, r);
    }
}

// find kmax number in arr
inline __device__ unsigned char QuickSelectMaxK(unsigned char arr[], int size, int k) {
    int l = 0, r = size;
    int big_index;
    int small_index;
    while (l < r) {
        big_index = r - 1;
        small_index = l;
        while (small_index < big_index) {
            if (arr[small_index + 1] <= arr[small_index]) {
                Swap(arr[small_index + 1], arr[small_index]);
                small_index++;
            } else {
                Swap(arr[small_index + 1], arr[big_index]);
                big_index--;
            }
        }

        if (small_index < k) {
            l = small_index + 1;
        } else {
            r = small_index;
        }
    }
    return arr[l];
}

inline __device__ unsigned char SelectMaxK(unsigned char arr[], int size, int k) {
    for (int i = 0; i <= k; i++) {
        int min = i;
        for (int j = i + 1; j < size; j++)
            if (arr[j] < arr[min]) min = j;
        Swap(arr[i], arr[min]);
    }
    return arr[k];
}

inline __device__ unsigned char BucketSelectMaxK(unsigned char arr[], int size, int k) {
    int bucket[256];
    for (int i = 0; i < 256; i++) {
        bucket[i] = 0;
    }
    for (int i = 0; i < size; i++) {
        bucket[int(arr[i])]++;
    }

    int idx = -1;
    for (int j = 0; j < 256; j++) {
        idx += bucket[j];
        if (idx >= k) {
            return j;
        }
    }
    return 255;
}
}  // namespace cudaop
}  // namespace smartmore

#endif