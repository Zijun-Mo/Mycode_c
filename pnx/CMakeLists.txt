# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.13)
# 定义项目名称
project(AUTO_AIM)
# 指定C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 启用 -O2 优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# 静态库生成的路径
set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# 测试程序生成的路径
set(EXEC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)
# 头文件目录
set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
# 源文件目录
set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)

#加载OpenCV库
find_package(OpenCV REQUIRED)
if(NOT ${OpenCV_FOUND})
    message(FATAL_ERROR "OpenCV not found")
endif()

#库名称
set(LIBS_OpenCV ${OpenCV_LIBS})

#可执行文件名称
set(EXEC_AIM auto_aim "")

#添加子目录
add_subdirectory(main)
add_subdirectory(armor_detector)