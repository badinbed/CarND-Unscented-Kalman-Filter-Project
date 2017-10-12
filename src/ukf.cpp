#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // state dimension
  n_x_ = 5;

  // augmented state dimension
  n_aug_ = n_x_ + 2;

  // number of sigma points
  n_sigma_ = 2 * n_aug_ + 1;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);


  // set weights
  weights_ = VectorXd(n_sigma_);
  weights_.fill(0.5 / (n_aug_ + lambda_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
bool UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) {
        return false;
    }
    if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) {
        return false;
    }

    if (!is_initialized_) {

      // first measurement
      cout << "UKF: " << endl;

      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
          double p = meas_package.raw_measurements_[0];
          double phi = meas_package.raw_measurements_[1];
          x_ << (p * cos(phi)), (p*sin(phi)), 0, 0, 0;
      }
      else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
          x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      }


      //state covariance matrix P
      P_  << 0.1, 0, 0, 0, 0,
             0, 0.1, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 1, 0,
             0, 0, 0, 0, 1;

      time_us_ = meas_package.timestamp_;
      // done initializing, no need to predict or update
      is_initialized_ = true;
      return true;
    }

    //compute the time elapsed between the current and previous measurements
    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    } else {
      UpdateLidar(meas_package);
    }

    // print the output
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl;
    return true;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  DONE:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

    //create augmented mean state
    VectorXd x_aug(n_aug_);
    x_aug.head(n_x_) = x_;
    x_aug(n_aug_ - 2) = 0;
    x_aug(n_aug_ - 1) = 0;

    //create augmented covariance matrix
    MatrixXd P_aug(n_aug_, n_aug_);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
    P_aug(n_aug_ - 2, n_aug_ - 2) = std_a_ * std_a_;
    P_aug(n_aug_ - 1, n_aug_ - 1) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    MatrixXd Xsig_aug(n_aug_, n_sigma_);
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }


    //predict sigma points
    for (int i = 0; i < n_sigma_; ++i) {

      //extract values for better readability
      double p_x = Xsig_aug(0, i);
      double p_y = Xsig_aug(1, i);
      double v = Xsig_aug(2, i);
      double yaw = Xsig_aug(3, i);
      double yawd = Xsig_aug(4, i);
      double nu_a = Xsig_aug(5, i);
      double nu_yawdd = Xsig_aug(6, i);

      //predicted state values
      double px_p, py_p;

      //avoid division by zero
      if (std::abs(yawd) > 0.001) {
          px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
          py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
      }
      else {
          px_p = p_x + v*delta_t*cos(yaw);
          py_p = p_y + v*delta_t*sin(yaw);
      }

      double v_p = v;
      double yaw_p = yaw + yawd*delta_t;
      double yawd_p = yawd;

      //add noise
      px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
      py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
      v_p = v_p + nu_a*delta_t;

      yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
      yawd_p = yawd_p + nu_yawdd*delta_t;

      //write predicted sigma point into right column
      Xsig_pred_(0,i) = px_p;
      Xsig_pred_(1,i) = py_p;
      Xsig_pred_(2,i) = v_p;
      Xsig_pred_(3,i) = yaw_p;
      Xsig_pred_(4,i) = yawd_p;
    }

    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < n_sigma_; ++i) {  //iterate over sigma points
      x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_; ++i) {  //iterate over sigma points

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      //angle normalization
      AngleNormalization(x_diff(3));

      P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    // measurement dimension
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++) {

      // extract values for better readibility
      double p_x = Xsig_pred_(0, i);
      double p_y = Xsig_pred_(1, i);

      // measurement model
      Zsig(0, i) = p_x; // px
      Zsig(1, i) = p_y; // py
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < n_sigma_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
      //residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;
    S = S + R;


    //calculate cross correlation matrix
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {

      //residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      AngleNormalization(x_diff(3));

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    AngleNormalization(x_(3));
    P_ = P_ - K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

    // measurement dimension
    int n_z = 3;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++) {

      // extract values for better readibility
      double p_x = Xsig_pred_(0, i);
      double p_y = Xsig_pred_(1, i);
      double v  = Xsig_pred_(2, i);
      double yaw = Xsig_pred_(3, i);

      double v1 = cos(yaw)*v;
      double v2 = sin(yaw)*v;

      // measurement model
      Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
      Zsig(1, i) = atan2(p_y,p_x);                                 //phi
      Zsig(2, i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < n_sigma_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
      //residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      //angle normalization
      AngleNormalization(z_diff(1));

      S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_* std_radrd_;
    S = S + R;


    //calculate cross correlation matrix
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {

      //residual
      VectorXd z_diff = Zsig.col(i) - z_pred;
      AngleNormalization(z_diff(1));

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      AngleNormalization(x_diff(3));

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    //angle normalization
    AngleNormalization(z_diff(1));

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    AngleNormalization(x_(3));
    P_ = P_ - K * S * K.transpose();
}

void UKF::AngleNormalization(double &angle) {
    if (angle > M_PI)
        angle -= static_cast<int>((angle+M_PI)/(2. * M_PI)) * 2. * M_PI;
    else if (angle < -M_PI)
        angle -= static_cast<int>((angle-M_PI)/(2. * M_PI)) * 2. * M_PI;;
}

void UKF::TestPrediction() {
    //set example state
    x_ <<   5.7441,
           1.3800,
           2.2049,
           0.5015,
           0.3528;

    //set example covariance matrix
    P_ <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
            -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
             0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
            -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
            -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    //Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.2;

    //Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.2;

    Prediction(0.1);
    cout << x_ << endl;
    cout << P_ << endl;

}
