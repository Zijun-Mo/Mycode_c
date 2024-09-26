#ifndef ARMOR_HPP
#define ARMOR_HPP
#include <opencv2/opencv.hpp>
#include <string>
#include "pnp_solver.hpp"


struct Armor {
    bool is_small; // 是否为小装甲板
    std::string classification; //装甲板类型
    cv::Mat position; // 装甲板位置(4*4矩阵表示，包括位置和旋转矩阵)
    double probability; 
    PnPSolver pnp_solver; // PnP解算器
};
#endif