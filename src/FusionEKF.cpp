#include "FusionEKF.h"
#include "tools.h"
#include "math.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  // measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

  // transition matrix
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1,0,0,0
            0,1,0,0
            0,0,1,0
            0,0,0,1;

  // process covariance matrix
  ekf_.Q_ = MatrixXd(4,4);

  noise_ax = 9;
  noise_ay = 9;


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    // state covariance matrix P
  	ekf_.P_ = MatrixXd(4, 4);
  	ekf_.P_ << 1, 0, 0, 0,
  			  0, 1, 0, 0,
  			  0, 0, 1000, 0,
  			  0, 0, 0, 1000;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      Radar measurementpack - rho (range), phi (angle), rho-dot (range rate)

      x = r*cos(theta)
      y = r*sin(theta)
      */

      float rho, phi, rho_dot;
      rho = measurement_pack.raw_measurements_[0];
      phi = measurement_pack.raw_measurements_[1];
      rho_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_[0] = rho * cos(phi);
      ekf_.x_[1] = rho * sin(phi);
      ekf_.x_[2] = rho_dot * sin(phi);
      ekf_.x_[3] = rho_dot * cos(phi);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      input vector is x, y
      ekf_.x_ is x, y, vx, vy
      */
      ekf_.x_[0] = measurement_pack.raw_measurements_[0];
      ekf_.x_[1] = measurement_pack.raw_measurements_[1];

    }

    previous_timestamp_ = measurement_pack.timestamp_; // grab time

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float deltaTime = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = deltaTime * deltaTime;
	float dt_3 = dt_2 * deltaTime;
	float dt_4 = dt_3 * deltaTime;

  // update state transition matrix
  ekf_.F_[0,2] = deltaTime;
  ekf_.F_[1,3] = deltaTime;

  // update process noise covariance matrix Q
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
       0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
       dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
       0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj = CalculateJacobian(ekf_.x_);

    ekf_.R_ = R_radar_;
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF();
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update();
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

// Calculate the Jacobian for the EKF update
MatrixXd calculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if(fabs(c1) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}
