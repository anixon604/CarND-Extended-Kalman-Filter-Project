#include "kalman_filter.h"
#include <iostream>
#define PI 3.14159265

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateHelper(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  float px, py, pvx, pvy, rho, phi, rho_dot;
  px = x_[0];
  py = x_[1];
  pvx = x_[2];
  pvy = x_[3];

  rho = sqrt((px*px)+(py*py));
  // atan2 returns a value between -pi and pi.
  phi = atan2 (py,px);
  rho_dot = ((px*pvx) + (py*pvy))/rho;

  VectorXd hx = VectorXd(3);
  hx << rho, phi, rho_dot;

  VectorXd y = z - hx;

  // normalize y to between -pi and pi
  while(true) {
    if(y[1] < -PI) {
      y[1] += PI;
      continue;
    } else if(y[1] > PI) {
      y[1] -= PI;
      continue;
    } else break;
  }

  //std::cout << "Y - " << y[1] << std::endl;

  UpdateHelper(y);
}

// shared between KF and EKF updates
void KalmanFilter::UpdateHelper(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
