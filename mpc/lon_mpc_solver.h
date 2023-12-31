#pragma once

#include <Eigen/Dense>

#include "glog/logging.h"
#include "mpc_solver.h"
#include "utils/common_utils/macros.h"

namespace mpc {
class LongitudinalMPCSolver : public MPCSolver {
  using Bounds = std::vector<std::pair<double, double>>;

 public:
  LongitudinalMPCSolver(const MPCConfig &config);

  PROP_SET(s_ref, std::vector<double>, s_ref, this->horizon_ + 1);
  PROP_SET(ds_ref, std::vector<double>, ds_ref, this->horizon_ + 1);
  PROP_SET(w_s_slack_l, std::vector<double>, diag_matrix_w_s_slack_l,
           this->horizon_ + 1);
  PROP_SET(w_s_slack_u, std::vector<double>, diag_matrix_w_s_slack_u,
           this->horizon_ + 1);
  PROP_SET(w_ds_slack_l, std::vector<double>, diag_matrix_w_ds_slack_l,
           this->horizon_ + 1);
  PROP_SET(w_ds_slack_u, std::vector<double>, diag_matrix_w_ds_slack_u,
           this->horizon_ + 1);

  PROP_SET(s_slack_u, std::vector<double>, s_slack_u, this->horizon_ + 1);
  PROP_SET(ds_slack_u, std::vector<double>, ds_slack_u, this->horizon_ + 1);

  PROP_SET(x_bounds, Bounds, x_bounds, this->horizon_ + 1);
  PROP_SET(dx_bounds, Bounds, x_bounds, this->horizon_ + 1);
  PROP_SET(ddx_bounds, Bounds, x_bounds, this->horizon_);
  PROP_SET(dddx_bounds, Bounds, x_bounds, this->horizon_);

 protected:
  void CalculateKernel(std::vector<OSQPFloat> *P_data,
                       std::vector<OSQPInt> *P_indices,
                       std::vector<OSQPInt> *P_indptr) override;

  void CalculateOffset(std::vector<OSQPFloat> *q) override;

  void CalculateAffineConstraint(std::vector<OSQPFloat> *A_data,
                                 std::vector<OSQPInt> *A_indices,
                                 std::vector<OSQPInt> *A_indptr,
                                 std::vector<OSQPFloat> *lower_bounds,
                                 std::vector<OSQPFloat> *upper_bounds) override;

 protected:
  void InitWeights(const google::protobuf::RepeatedField<double> &matrix,
                   const double num, std::vector<double> *const diag_matrix);

  void CompensateSlackWeights(std::vector<double> *weights);

  void InitStateMatrices();

 protected:
  size_t num_of_state_;
  size_t num_of_control_;
  size_t num_of_slack_var_;

  double s_init_;
  double ds_init_;
  double prev_dds_;

  Eigen::MatrixXd matrix_A_k_;
  Eigen::MatrixXd matrix_B_k_;

  Eigen::MatrixXd matrix_q_n_;

  std::vector<double> diag_matrix_q_;
  std::vector<double> diag_matrix_r_;
  std::vector<double> diag_matrix_r_dot_;
  std::vector<double> diag_matrix_w_s_slack_l_;
  std::vector<double> diag_matrix_w_s_slack_u_;
  std::vector<double> diag_matrix_w_ds_slack_l_;
  std::vector<double> diag_matrix_w_ds_slack_u_;

  std::vector<double> s_slack_u_;
  std::vector<double> ds_slack_u_;
  Bounds x_bounds_;
  Bounds dx_bounds_;
  Bounds ddx_bounds_;
  Bounds dddx_bounds_;

  std::vector<double> s_ref_;
  std::vector<double> ds_ref_;
};
}  // namespace mpc