#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "detector.hpp"
// 将图像转换为灰度图像并进行二值化
Detector::Detector(){

}
cv::Mat Detector::convertToAdaptiveBinary(const cv::Mat& img, const cv::Ptr<cv::CLAHE> clahe) {
    cv::Mat grayImg, equalizedImg, binaryImg; 

    // 将图像转换为红色通道减去蓝色通道的灰度图像
    cv::Mat channels[3];
    cv::split(img, channels); // 分割图像为三个通道
    grayImg = channels[2] - 0.3 * channels[0]; // 红色通道减去蓝色通道

    // 对灰度图像进行亮度自适应（CLAHE）
    // 应用CLAHE到灰度图像
    clahe->apply(grayImg, equalizedImg);
    // imshow("equalizedImg", equalizedImg); 
    // cv::waitKey(30); 

    // 进行全局二值化
    cv::threshold(equalizedImg, binaryImg, 180, 255, cv::THRESH_BINARY);
    // std::cout << (cv::getTickCount() - start) / cv::getTickFrequency() << "\n"; 

    // 去除小于9个像素的明亮噪点
    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_OPEN, morphKernel); 

    return binaryImg;
}
// 处理轮廓
std::vector<cv::RotatedRect> Detector::processContours(const cv::Mat& binaryImg, cv::Mat& originalImg) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::RotatedRect> rectangles;

    // 查找轮廓
    cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        // 跳过包围像素过少的轮廓
        if (cv::contourArea(contour) < 10 || cv::contourArea(contour) > 2000) {
            continue;
        }

        // 计算轮廓的最小外接可旋转矩形
        cv::RotatedRect minRect = cv::minAreaRect(contour);
        if(!this->isLight(minRect, contour)) {
            continue; 
        }

        // 判断矩形区域内的像素颜色
        if (!this->isRedDominant(originalImg, minRect)) {
            continue; 
        }
        // 绘制矩形在原图上
        cv::Point2f rectPoints[4];
        minRect.points(rectPoints);
        for (int j = 0; j < 4; j++) {
            cv::line(originalImg, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
        }
        // 保存矩形在动态数组中
        rectangles.push_back(minRect);
    }

    return rectangles;
}
bool Detector::isLight(cv::RotatedRect& rect, const std::vector<cv::Point>& contour) {
    if (rect.size.width > rect.size.height) {
        std::swap(rect.size.width, rect.size.height); 
        rect.angle -= 90.0; // 调整角度，使其与短边一致
    }
    if(rect.size.height < 1.0 * rect.size.width) return false; 
    if(std::abs(rect.angle) > 40.0) return false; 

    // 计算最小外接矩形的面积
    float rectArea = rect.size.width * rect.size.height;
    // 计算轮廓的面积
    float contourArea = cv::contourArea(contour);
    // 判断面积和面积比
    if (rectArea < 50 && contourArea / rectArea <= 0.4) return false;
    if (rectArea >= 50 && contourArea / rectArea <= 0.6) return false;

    return true; 
}
bool Detector::isSimilarRotatedRect(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, bool &issmall) {
    // 计算旋转角度差
    if (std::abs(rect1.angle - rect2.angle) > 10.0) return false; 
    // 计算形状大小差异
    float height1 = rect1.size.height;
    float height2 = rect2.size.height; 

    float size_diff_height = std::abs(height1 - height2) / std::max(height1, height2);
    if (size_diff_height > 0.3) return false; 

    // 计算两个矩形之间的间距与矩形长边的比
    float distance_ratio = cv::norm(rect1.center - rect2.center) / std::max(height1, height2);
    if (distance_ratio < 1.0 || distance_ratio > 6.0) return false; 
    if (distance_ratio < 3.0) issmall = true; 
    else issmall = false;

    // std::cout << "distance_ratio: " << distance_ratio << std::endl; 

    return true;
}
bool Detector::isRedDominant(const cv::Mat& originalImg, const cv::RotatedRect& minRect) {
    // 获取旋转矩形的四个顶点
    cv::Point2f vertices[4];
    minRect.points(vertices);

    // 获取旋转矩形的边界矩形
    cv::Rect boundingRect = minRect.boundingRect();

    // 初始化红色通道和蓝色通道的和
    double redSum = 0;
    double blueSum = 0;

    // 枚举边界矩形中的每个点
    for (int y = boundingRect.y; y < boundingRect.y + boundingRect.height; ++y) {
        for (int x = boundingRect.x; x < boundingRect.x + boundingRect.width; ++x) {
            // 检查点是否在旋转矩形内
            if (cv::pointPolygonTest(std::vector<cv::Point2f>(vertices, vertices + 4), cv::Point2f(x, y), false) >= 0) {
                // 获取像素值
                cv::Vec3b pixel = originalImg.at<cv::Vec3b>(y, x);
                // 累加红色通道和蓝色通道的值
                redSum += pixel[2];  // 红色通道
                blueSum += pixel[0]; // 蓝色通道
            }
        }
    }

    // 判断红色通道的和是否大于蓝色通道的和
    return redSum > blueSum * 1.1;
}
std::vector<cv::Point2f> Detector::mergeSimilarRects(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    // 获取两个旋转矩形的四个顶点
    cv::Point2f vertices1[4];
    cv::Point2f vertices2[4];
    rect1.points(vertices1);
    rect2.points(vertices2);

    // 找到左侧矩形的右侧两点和右侧矩形的左侧两点
    std::vector<cv::Point2f> newRect(4); 
    cv::Point2f v1 = vertices1[2] - vertices1[3]; 
    cv::Point2f v2 = vertices2[1] - vertices2[0];
    newRect[0] = rect1.center + v1 * 1.2; 
    newRect[3] = rect1.center - v1 * 1.2; 
    newRect[2] = rect2.center - v2 * 1.2;
    newRect[1] = rect2.center + v2 * 1.2;

    return newRect;
}
cv::Mat Detector::warpToRectangle(const cv::Mat& img, const std::vector<cv::Point2f>& quad, int width, int height) {
    // 定义矩形的四个顶点
    std::vector<cv::Point2f> rectangle = {
        cv::Point2f(0, 0),
        cv::Point2f(width - 1, 0),
        cv::Point2f(width - 1, height - 1),
        cv::Point2f(0, height - 1)
    };

    // 计算透视变换矩阵
    cv::Mat transformMatrix = cv::getPerspectiveTransform(quad, rectangle);

    // 进行透视变换
    cv::Mat warpedImg;
    cv::warpPerspective(img, warpedImg, transformMatrix, cv::Size(width, height));

    return warpedImg;
}