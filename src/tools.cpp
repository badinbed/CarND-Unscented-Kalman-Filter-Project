#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    if(estimations.size() != ground_truth.size() || estimations.size() == 0) {
      cout << "estimations and ground_truth data size mismatch" << endl;
      return rmse;
    }

    for(unsigned int i = 0; i < estimations.size(); ++i) {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array()*residual.array();
      rmse += residual;
    }

    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    return rmse;
}
