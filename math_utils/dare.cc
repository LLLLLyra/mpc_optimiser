#include "dare.h"

#include "glog/logging.h"

namespace math_utils {
void Dare(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
          const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
          const double tolerance, const uint max_num_iteration,
          Eigen::MatrixXd *ptr_K) {
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(Q.rows(), R.cols());
  if (A.rows() != A.cols() || B.rows() != A.rows() || Q.rows() != Q.cols() ||
      Q.rows() != A.rows() || R.rows() != R.cols() || R.rows() != B.cols() ||
      M.rows() != Q.rows() || M.cols() != R.cols()) {
    LOG(ERROR) << "Dare: one or more matrices have incompatible dimensions.";
    return;
  }

  Eigen::MatrixXd AT = A.transpose();
  Eigen::MatrixXd BT = B.transpose();
  Eigen::MatrixXd MT = M.transpose();

  // Solves a discrete-time Algebraic Riccati equation (DARE)
  // Calculate Matrix Difference Riccati Equation, initialize P and Q
  Eigen::MatrixXd P = Q;
  uint num_iteration = 0;
  double diff = std::numeric_limits<double>::max();
  while (num_iteration++ < max_num_iteration && diff > tolerance) {
    Matrix P_next =
        AT * P * A -
        (AT * P * B + M) * (R + BT * P * B).inverse() * (BT * P * A + MT) + Q;
    // check the difference between P and P_next
    diff = fabs((P_next - P).maxCoeff());
    P = P_next;
  }

  if (num_iteration >= max_num_iteration) {
    LOG(DEBUG) << "Dare cannot converge to a solution, "
                  "last consecutive result diff is: "
               << diff;
  } else {
    LOG(DEBUG) << "Dare converged at iteration: " << num_iteration
               << ", max consecutive result diff.: " << diff;
  }
  *ptr_K = (R + BT * P * B).inverse() * (BT * P * A + MT);
}

void Dare(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
          const std::vector<double> &diag_q, const std::vector<double> &diag_r,
          const double tolerance, const uint max_num_iteration,
          Eigen::MatrixXd *ptr_K) {
  Eigen::MatrixXd Q = Eigen::MatrixXd(diag_q.data()).asDiagnal();
  Eigen::MatrixXd R = Eigen::MatrixXd(diag_r.data()).asDiagnal();
  Dare(A, B, Q, R, tolerance, max_num_iteration, ptr_K);
}
}  // namespace math_utils