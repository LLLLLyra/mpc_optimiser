#pragma once

#include <Eigen/Dense>
#include <vector>

namespace math_utils {

void Dare(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
          const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
          const double tolerance, const uint max_num_iteration,
          Eigen::MatrixXd *ptr_K);

void Dare(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
          const std::vector<double> &diag_q, const std::vector<double> &diag_r,
          const double tolerance, const uint max_num_iteration,
          Eigen::MatrixXd *ptr_K);
}