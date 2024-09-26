#include "tracker.hpp"
#include "armor.hpp"
#include "pnp_solver.hpp"

Tracker::Tracker(const Armor& armor)
    : kf_(6, 3, 0, CV_64F), state_(6, 1, CV_64F), meas_(3, 1, CV_64F) {
    // kf_是卡尔曼滤波器对象, state_是状态向量, meas_是测量向量
    // 初始化状态向量
    state_.at<double>(0) = armor.position.at<double>(0, 3); // x
    state_.at<double>(1) = armor.position.at<double>(1, 3); // y
    state_.at<double>(2) = 0; // v_x
    state_.at<double>(3) = 0; // v_y
    state_.at<double>(4) = armor.position.at<double>(2, 3); // z
    state_.at<double>(5) = 0; // v_z

    // 初始化测量矩阵 H
    kf_.measurementMatrix = cv::Mat::zeros(3, 6, CV_64F);
    kf_.measurementMatrix.at<double>(0) = 1.0f;
    kf_.measurementMatrix.at<double>(7) = 1.0f;
    kf_.measurementMatrix.at<double>(16) = 1.0f;

    // 初始化过程噪声协方差矩阵 Q
    kf_.processNoiseCov = cv::Mat::eye(6, 6, CV_64F) * 1e-2;

    // 初始化测量噪声协方差矩阵 R
    kf_.measurementNoiseCov = cv::Mat::eye(3, 3, CV_64F) * 1e-1;

    // 初始化后验错误估计协方差矩阵 P
    kf_.errorCovPost = cv::Mat::eye(6, 6, CV_64F);

    // 初始化状态
    kf_.statePost = state_;

    // 初始化分类和是否是小装甲板
    classification = armor.classification;
    this->issmall = armor.is_small;

    // 初始化更新时间
    last_update_time_ = std::chrono::steady_clock::now();
}

Tracker::Tracker(){}
cv::Mat Tracker::predict() {
    return kf_.predict();
}

void Tracker::update(const Armor& armor) {
    meas_.at<double>(0) = armor.position.at<double>(0, 3); // x
    meas_.at<double>(1) = armor.position.at<double>(1, 3); // y
    meas_.at<double>(2) = armor.position.at<double>(2, 3); // z

    kf_.correct(meas_);

    // 更新更新时间
    last_update_time_ = std::chrono::steady_clock::now();
    lost_ = false;
}

cv::Point3f Tracker::getPosition() const {
    return cv::Point3f(state_.at<double>(0), state_.at<double>(1), state_.at<double>(4));
}

bool Tracker::isLost() const {
    return lost_;
}

void Tracker::markLost() {
    lost_ = true;
}

bool Tracker::isExpired() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_update_time_);
    return duration.count() > 1;
}

bool isSameArmor(const Tracker& tracker, const Armor& armor) {
    if(tracker.classification != armor.classification) {
        return false;
    }
    if(tracker.issmall != armor.is_small) {
        return false;
    }
    cv::Point3f trackerPos = tracker.getPosition();
    cv::Point3f armorPos(armor.position.at<double>(0, 3), armor.position.at<double>(1, 3), armor.position.at<double>(2, 3));

    // 计算两个装甲板中心点之间的距离
    double distance = cv::norm(trackerPos - armorPos);

    // 如果距离小于某个阈值，则认为是同一个目标
    const double distanceThreshold = 0.05f; // 你可以根据实际情况调整这个阈值
    return distance < distanceThreshold;
}