# 指定最低版本和项目名称
cmake_minimum_required(VERSION 3.10)
project(MAIN)

# 指定C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 -O2 优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# 指定搜索路径
include_directories(${HEAD_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})
add_executable(${EXEC_AIM} main.cpp)

# 添加宏
add_definitions(${OpenCV_DEFINITIONS})

# 链接库文件
target_link_libraries(${EXEC_AIM} ${LIBS_OpenCV})

target_sources(${EXEC_AIM} PRIVATE ${SRC_FILES})