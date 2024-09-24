#ifndef PNP_SOLVER_HPP_
#define PNP_SOLVER_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class PnPSolver {
public:
    PnPSolver();

    cv::Mat solvePnPWithIPPE(const std::vector<cv::Point2f>& imagePoints, const std::string& filename, const bool issmall); // PnP解算器函数

private:
    std::vector<cv::Point3f> l_points, s_points; 
    std::vector<cv::Point3f> objectPoints; 
    cv::Mat rvec, tvec, rotationMatrix, transformMatrix; 
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F), distCoeffs = cv::Mat::zeros(5, 1, CV_64F); 
    bool success; 
    
    bool readCameraParameters(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs); // 读取相机参数
};

#endif  // PNP_SOLVER_HPP_