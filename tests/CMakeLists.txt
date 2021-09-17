include(fetch/Catch2)

FetchContent_GetProperties(Catch2 SOURCE_DIR catch2_SOURCE_DIR)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

include(Catch)

include_directories(common)

set(test_srcs 
    image/test_image_convert.cu
    image/test_image_resize.cu
    image/test_image_channel_split.cu
    image/test_image_channel_merge.cu
    image/test_image_meanfilter.cu
    image/test_image_medianfilter.cu
    image/test_image_gaussianfilter.cu
    image/test_image_copy.cu
    image/test_image_flip.cu
    image/test_image_slice.cu
    image/test_image_stitching.cu
    image/test_image_sqrt.cu
    image/test_image_addweighted.cu    
    image/test_image_copymakeborder.cu   
    image/test_image_alphablend.cu
    image/test_image_minarearect.cu
    image/test_image_scharr.cu
    image/test_image_sobel.cu
    image/test_arc_length.cu
    image/test_image_pyrdown.cu
    
    tensor/test_reduce.cu
    tensor/test_depth_to_space.cu
    tensor/test_clamp.cu
    tensor/test_datatypeconvert.cu
    tensor/test_einsum.cu
    tensor/test_concat.cu
    tensor/test_meannormalization.cu
    tensor/test_threshold.cu
    tensor/test_transpose.cu
)

find_package(OpenCV REQUIRED)

foreach(test ${test_srcs})
    get_filename_component(test_target ${test} NAME_WE)

    message(STATUS "${PROJECT_NAME} add test ${test_target}")

    add_executable(${test_target} ${test})

    target_include_directories(${test_target}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(${test_target}
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        CUDA::cudart
        Catch2::Catch2
    )

    catch_discover_tests(${test_target})
endforeach()
