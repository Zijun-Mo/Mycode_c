# 静态库生成的路径
set(LIB_PATH ${LIB_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/lib PERANT_SCOPE)
# 头文件目录
set(HEAD_PATH ${HEAD_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/include PERANT_SCOPE)
# 源文件目录
set(SRC_PATH ${SRC_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src PERANT_SCOPE)

#库名称
set(LIBS_OpenCV ${OpenCV_LIBS})

# 链接源文件
set(SRC_FILES ${SRC_PATH} ${SRC_PATH}/detector.cpp
                          ${SRC_PATH}/number_classifier.cpp
                          ${SRC_PATH}/pnp_solver.cpp PERANT_SCOPE)