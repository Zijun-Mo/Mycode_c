#include "pnp_solver.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

PnPSolver::PnPSolver() {
    l_points = {
        cv::Point3f(-0.135, -0.0635, 0.0), 
        cv::Point3f(0.135, -0.0635, 0.0), 
        cv::Point3f(0.135, 0.0635, 0.0), 
        cv::Point3f(-0.135, 0.0635, 0.0) 
    }; 
    s_points = {
        cv::Point3f(-0.0635, -0.0625, 0.0), 
        cv::Point3f(0.0635, -0.0625, 0.0), 
        cv::Point3f(0.0635, 0.0625, 0.0), 
        cv::Point3f(-0.0635, 0.0625, 0.0) 
    };// 世界坐标系中的四个点
}
// PnP解算器函数
cv::Mat PnPSolver::solvePnPWithIPPE(const std::vector<cv::Point2f>& imagePoints, const std::string& filename, const bool issmall)
{
    if(issmall){ 
        objectPoints = s_points; 
    }
    else{
        objectPoints = l_points; 
    }
    if (!readCameraParameters(filename, cameraMatrix, distCoeffs)) {
        std::cerr << "读取相机参数失败" << std::endl;
    }
    success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE); 
    if (!success) {
        throw std::runtime_error("PnP解算失败");
    }
    // 将旋转向量转换为旋转矩阵
    cv::Rodrigues(rvec, rotationMatrix); 

    // 构建4x4变换矩阵
    transformMatrix = cv::Mat::eye(4, 4, CV_64F); 
    rotationMatrix.copyTo(transformMatrix(cv::Rect(0, 0, 3, 3))); 
    tvec.copyTo(transformMatrix(cv::Rect(3, 0, 1, 3))); 
    transformMatrix.at<double>(3, 3) = 1.0; 

    return transformMatrix;
}
// 读取相机参数的函数
bool PnPSolver::readCameraParameters(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    // 打开YAML文件
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return false;
    }

    // 读取相机内参矩阵和失真系数
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    // 关闭文件
    fs.release();

    return true;
}
