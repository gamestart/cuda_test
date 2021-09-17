# CUDA-operators

## 算子

[支持的算子介绍](./doc/op-introduction.md)

## 项目目录结构

项目目录以以下方式进行组织。

``` bash
.
├── src                 # 源文件目录
├── tests               # 单元测试目录
├── utils               # 工具代码目录
├── scripts             # 脚本目录
├── reference           # 参考代码目录
├── third_party         # 第三方依赖目录
├── demo                # 样例代码目录
├── README.md
├── .clang-format
├── .gitignore
└── CMakeLists.txt
```

## CMake集成

1. 在项目根目录下创建`cmake/fetch/cuda-operators.cmake`文件，并写入以下内容

```cmake
include(FetchContent)

FetchContent_Declare(cuda-operators
    GIT_REPOSITORY https://github.com/smartmore/cuda-operators.git
    GIT_TAG     v1.0
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(cuda-operators)
```

2. 在项目`CMakeLists.txt`中加入下面的内容

```cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
```

3. 然后在需要依赖`cuda-operators`的地方加入

```cmake
include(fetch/cuda-operators)
target_link_libraries(main PRIVATE cuda-operators)
```
