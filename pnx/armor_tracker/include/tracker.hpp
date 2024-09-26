#ifndef TRACKER_HPP_
#define TRACKER_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include "armor.hpp"
#include "pnp_solver.hpp"

class Tracker {
public:
    Tracker(const Armor& armor); 
    Tracker(); 
    cv::Mat predict();
    void update(const Armor& armor); 
    cv::Point3f getPosition() const;
    bool isLost() const;
    void markLost();
    bool isExpired() const;
    // 判断两个装甲板是否是同一个目标
    friend bool isSameArmor(const Tracker& tracker, const Armor& armor);
    PnPSolver pnp_solver; 

private:
    cv::KalmanFilter kf_;
    cv::Mat state_;
    cv::Mat meas_;
    bool issmall, lost_; 
    std::string classification; 
    int64 last_update_time; 
    std::chrono::steady_clock::time_point last_update_time_; 
};


#endif // TRACKER_HPP_