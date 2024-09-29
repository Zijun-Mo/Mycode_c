#include "tracker.hpp"
#include "armor.hpp"
#include "pnp_solver.hpp"

Tracker::Tracker(const Armor& armor, const double& dt)
    : kf_(6, 3, 0, CV_64F), state_(6, 1, CV_64F), meas_(3, 1, CV_64F) {
    // kf_是卡尔曼滤波器对象, state_是状态向量, meas_是测量向量
    // 初始化状态向量 x
    state_.at<double>(0) = armor.position.at<double>(0, 3); // x
    state_.at<double>(1) = armor.position.at<double>(1, 3); // y
    state_.at<double>(2) = armor.position.at<double>(2, 3); // z
    state_.at<double>(3) = 0; // v_x
    state_.at<double>(4) = 0; // v_y
    state_.at<double>(5) = 0; // v_z

    // 初始化测量矩阵 H 用于去除速度信息
    kf_.measurementMatrix = cv::Mat::zeros(3, 6, CV_64F);
    kf_.measurementMatrix.at<double>(0) = 1.0f;
    kf_.measurementMatrix.at<double>(7) = 1.0f;
    kf_.measurementMatrix.at<double>(14) = 1.0f;

    // 初始化过程噪声协方差矩阵 Q 
    

    // 初始化加速度转移矩阵 
    cv::Mat acceleration = (cv::Mat_<double>(6, 3) <<
     0.5 * dt * dt, 0, 0, 
     0, 0.5 * dt * dt, 0, 
     0, 0, 0.5 * dt * dt, 
     dt, 0, 0, 
     0, dt, 0, 
     0 ,0, dt);

    kf_.processNoiseCov = acceleration * cv::Mat::eye(3, 3, CV_64F) * acceleration.t() * 1e1;

    // 初始化测量噪声协方差矩阵 R
    kf_.measurementNoiseCov = cv::Mat::eye(3, 3, CV_64F) * 1e-5;

    // 初始化后验错误估计协方差矩阵 P
    kf_.errorCovPost = cv::Mat::eye(6, 6, CV_64F);

    // 初始化状态
    kf_.statePost = state_;

    // 初始化状态转移矩阵
    kf_.transitionMatrix = (cv::Mat_<double>(6, 6) <<
        1, 0, 0, dt, 0, 0,
        0, 1, 0, 0, dt, 0,
        0, 0, 1, 0, 0, dt,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1
    );

    // 初始化分类和是否是小装甲板
    classification = armor.classification;
    this->issmall = armor.is_small;

    // 初始化更新时间
    last_update_time = armor.frame_id;
}

Tracker::Tracker(){}
cv::Mat Tracker::predict() {
    cv::Mat prediction = kf_.predict();
    this->pnp_solver.updateExtrinsic(prediction.at<double>(0), prediction.at<double>(1), prediction.at<double>(2));
    return prediction; 
}

void Tracker::update(const Armor& armor) {
    meas_.at<double>(0) = armor.position.at<double>(0, 3); // x
    meas_.at<double>(1) = armor.position.at<double>(1, 3); // y
    meas_.at<double>(2) = armor.position.at<double>(2, 3); // z
    this->pnp_solver = armor.pnp_solver; 

    kf_.correct(meas_);

    // 更新更新时间
    last_update_time = armor.frame_id; 
    lost_ = false;
}

cv::Point3f Tracker::getPosition() const {
    return cv::Point3f(state_.at<double>(0), state_.at<double>(1), state_.at<double>(2));
}

cv::Point3f Tracker::getVelocity() const {
    return cv::Point3f(state_.at<double>(3), state_.at<double>(4), state_.at<double>(5));
}

bool Tracker::isLost() const {
    return lost_;
}

void Tracker::markLost(const int64& frame_id) {
    if(last_update_time != frame_id) lost_ = true;
}

bool Tracker::isExpired(const int64& frame_id, const int64& gap_time) const {
    return frame_id - last_update_time > gap_time;
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
    const double distanceThreshold = 0.2f; // 你可以根据实际情况调整这个阈值
    return distance < distanceThreshold;
}