#ifndef TRACKER_HPP_
#define TRACKER_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include "armor.hpp"
#include "pnp_solver.hpp"

class Tracker {
public:
    Tracker(const Armor& armor, const double& dt);  
    Tracker(); 
    cv::Mat predict();
    void update1(const Armor& armor);  
    void update2(const Armor& armor1, const Armor& armor2);  
    cv::Point3f getPosition() const; 
    cv::Point3f getVelocity() const;
    bool isLost() const;
    void markLost(const int64& frame_id);
    bool isExpired(const int64& frame_id, const int64& gap_time) const;
    void initializeMeasurementMatrix1(double theta1, double r); 
    void initializeMeasurementMatrix2(double theta1, double theta2, double r1, double r2); 
    // 判断两个装甲板是否是同一个目标
    friend bool isSameArmor(const Tracker& tracker, const Armor& armor);

private:
    cv::KalmanFilter kf_;
    cv::Mat state_;
    cv::Mat meas1_, meas2_;
    bool issmall, lost_; 
    std::string classification; 
    int64 last_update_time; 
};
double abs_yaw(double x); 
double yawinrange(double x); 


#endif // TRACKER_HPP_