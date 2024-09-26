#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include "number_classifier.hpp"
#include "pnp_solver.hpp"
#include "detector.hpp"
#include "armor.hpp"
#include "tracker.hpp"

Armor armor; // 装甲板结构体

// 函数声明
bool readVideo(const std::string& filename, cv::VideoCapture& cap); // 从文件中读取视频
void Draw(cv::Mat& frame, const std::vector<cv::Point2f>& mergedRect, const Armor& armor); // 绘制矩形在原图上

int64 start;
// 初始化跟踪器列表
std::vector<Tracker> trackers;
int main() {
    // 打开视频文件
    cv::VideoCapture cap;
    if (!readVideo("test.mp4", cap)) {
        return -1;
    }
    // 创建CLAHE对象，用于均衡亮度
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4.0);
    // 获取视频的帧率和帧大小
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS); 
    // 创建视频写入对象
    cv::VideoWriter video("../img_output/output_video.mp4", cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(frame_width, frame_height));
    start = cv::getTickCount(); 
    cv::Mat frame;

    while (cap.read(frame)) {
        // 翻转蓝色和红色通道(可选,在敌方为蓝方是需要)
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        std::swap(channels[0], channels[2]);
        cv::merge(channels, frame); 
        // 将图像转换为灰度图像并进行二值化
        Detector detector; 
        cv::Mat binaryImg = detector.convertToAdaptiveBinary(frame, clahe);
        // 处理轮廓并获取最小外接可旋转矩形
        std::vector<cv::RotatedRect> rectangles = detector.processContours(binaryImg, frame); 

        // 判断两个旋转矩形是否相似
        for (int i = 0; i < rectangles.size(); i++) {
            for (int j = i + 1; j < rectangles.size(); j++) {
                bool issmall; 
                if (detector.isSimilarRotatedRect(rectangles[i], rectangles[j], issmall)) {
                    // 合并相似的矩形
                    std::vector<cv::Point2f> mergedRect;
                    mergedRect = rectangles[i].center.x < rectangles[j].center.x
                                    ? detector.mergeSimilarRects(rectangles[i], rectangles[j])
                                    : detector.mergeSimilarRects(rectangles[j], rectangles[i]); 
                    // 将四边形内容投影为长方形
                    cv::Mat squareImg; 
                    squareImg = issmall ? detector.warpToRectangle(frame, mergedRect, 32, 28)(cv::Rect(6, 0, 20, 28)) 
                                        : detector.warpToRectangle(frame, mergedRect, 54, 28)(cv::Rect(17, 0, 20, 28));
                    // 数字识别
                    NumberClassifier number_classifier("mlp.onnx", "label.txt", 0.5);  
                    std::pair<std::string, double> result = number_classifier.classifyNumber(squareImg, issmall); 
                    if(result.first == "negative") continue; 
                    armor.is_small = issmall; 
                    armor.classification = result.first; 
                    armor.probability = result.second; 
                    // PnP解算相机外参
                    armor.position = armor.pnp_solver.solvePnPWithIPPE(mergedRect, "../input/2BDFA1701242.yaml", issmall); 
                    // 绘制矩形在原图上
                    Draw(frame, mergedRect, armor); 
                    // 更新或添加跟踪器
                    bool update = false;
                    for (auto& tracker : trackers) {
                        if (isSameArmor(tracker, armor)) {
                            tracker.update(armor);// 更新跟踪器
                            tracker.pnp_solver = armor.pnp_solver; 
                            update = true;
                            break;
                        }
                    }
                    if (!update) {
                        trackers.emplace_back(armor);
                        trackers.back().pnp_solver = armor.pnp_solver; 
                    }
                }
            }
        }
        // 检查并删除过期的 Tracker
        auto it = trackers.begin();
        while (it != trackers.end()) {
            if (it->isExpired()) {
                it = trackers.erase(it);
            } else {
                ++it;
            }
        }
        // 预测并绘制结果
        for (auto& tracker : trackers) {
            cv::Mat prediction = tracker.predict();

            // 将预测的世界坐标系或相机坐标系的点转换为图像坐标系的点
            std::cout << prediction.at<double>(0) << " " << prediction.at<double>(1) << " " << prediction.at<double>(4) << " "<< std::endl; 
            cv::Point3f worldPoint(prediction.at<double>(0), prediction.at<double>(1), prediction.at<double>(4));
            cv::Point2f imagePoint = tracker.pnp_solver.worldToImage(worldPoint);
            std::cout << imagePoint << std::endl; 

            // 获取图像坐标系的 x 和 y 坐标
            cv::Rect2d predRect(imagePoint.x, imagePoint.y, 50, 50);
            cv::rectangle(frame, predRect, cv::Scalar(0, 255, 0), 2);

            // 获取预测位置
            cv::Point3f position = tracker.getPosition();
            std::string positionText = "Pos: (" + std::to_string(position.x) + ", " + std::to_string(position.y) + ", " + std::to_string(position.z) + ")";
            cv::putText(frame, positionText, cv::Point(predRect.x, predRect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            // 如果 Tracker 丢失，绘制丢失标记
            if (tracker.isLost()) {
                std::string lostText = "LOST";
                cv::putText(frame, lostText, cv::Point(predRect.x, predRect.y - 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
        }
        // 将处理后的帧写入视频文件
        video.write(frame); 
    }
    std::cout << (cv::getTickCount() - start) / cv::getTickFrequency() << "\n"; 
    
    cap.release();
    video.release();
    cv::destroyAllWindows();

    return 0;
}

// 从文件中读取视频
bool readVideo(const std::string& filename, cv::VideoCapture& cap) {
    cap.open("../img_input/"+filename);

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open or find the video." << std::endl;
        return false;
    }
    return true;
} 
// 绘制矩形在原图上
void Draw(cv::Mat& frame, const std::vector<cv::Point2f>& mergedRect, const Armor& armor) {
    cv::putText(frame, armor.classification, mergedRect[3], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, std::to_string(armor.probability), mergedRect[2], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    for (int k = 0; k < 4; k++) {
        cv::line(frame, mergedRect[k], mergedRect[(k + 1) % 4], cv::Scalar(255, 255, 0), 2); 
    }
    std::string text = "(" + std::to_string(armor.position.at<double>(0, 3)) + 
                        ", " + std::to_string(armor.position.at<double>(1, 3)) + 
                        ", " + std::to_string(armor.position.at<double>(2, 3)) + ")"; 
    cv::putText(frame, text, cv::Point(mergedRect[0].x, mergedRect[0].y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2); 
}