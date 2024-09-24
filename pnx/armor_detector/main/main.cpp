#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "number_classifier.hpp"
#include "pnp_solver.hpp"
#include "detector.hpp"

struct Armor {
    bool is_small; 
    std::string classification;
    cv::Mat position; 
    double probability; 
}armor; // 装甲板结构体

// 函数声明
bool readVideo(const std::string& filename, cv::VideoCapture& cap); // 从文件中读取视频
void Draw(cv::Mat& frame, const std::vector<cv::Point2f>& mergedRect, const Armor& armor); // 绘制矩形在原图上

int64 start; 
std::vector<Armor> armors; 
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
                    PnPSolver pnp_solver; 
                    cv::Mat armor_camera = pnp_solver.solvePnPWithIPPE(mergedRect, "../input/2BDFA1701242.yaml", issmall); 
                    armor.position = armor_camera; 
                    armors.push_back(armor); 
                    // 绘制矩形在原图上
                    Draw(frame, mergedRect, armor); 
                }
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