#pragma once

#include <Eigen/Dense>

#include "glog/logging.h"
#include "mpc_solver.h"
#include "utils/common_utils/macros.h"

namespace mpc {
class LateralMPCSolver : public MPCSolver {
  using Bounds = std::vector<std::pair<double, double>>;

 public:
  LateralMPCSolver(const MPCConfig &config);

  PROP_SET(l_ref, std::vector<double>, l_ref, this->horizon_ + 1);
  PROP_SET(l_dot_ref, std::vector<double>, l_dot_ref, this->horizon_ + 1);
  PROP_SET(psi_ref, std::vector<double>, psi_ref, this->horizon_ + 1);
  PROP_SET(psi_dot_ref, std::vector<double>, psi_dot_ref, this->horizon_ + 1);

  PROP_SET(w_l_slack_l, std::vector<double>, diag_matrix_w_l_slack_l,
           this->horizon_ + 1);
  PROP_SET(w_l_dot_slack_l, std::vector<double>, diag_matrix_w_l_dot_slack_l,
           this->horizon_ + 1);
  PROP_SET(w_l_slack_u, std::vector<double>, diag_matrix_w_l_slack_u,
           this->horizon_ + 1);
  PROP_SET(w_l_dot_slack_u, std::vector<double>, diag_matrix_w_l_dot_slack_u,
           this->horizon_ + 1);

  PROP_SET(w_psi_slack_l, std::vector<double>, diag_matrix_w_psi_slack_l,
           this->horizon_ + 1);
  PROP_SET(w_psi_dot_slack_l, std::vector<double>,
           diag_matrix_w_psi_dot_slack_l, this->horizon_ + 1);
  PROP_SET(w_psi_slack_u, std::vector<double>, diag_matrix_w_psi_slack_u,
           this->horizon_ + 1);
  PROP_SET(w_psi_dot_slack_u, std::vector<double>,
           diag_matrix_w_psi_dot_slack_u, this->horizon_ + 1);

  PROP_SET(l_bounds, Bounds, l_bounds, this->horizon_ + 1);
  PROP_SET(l_dot_bounds, Bounds, l_dot_bounds, this->horizon_ + 1);
  PROP_SET(psi_bounds, Bounds, psi_bounds, this->horizon_ + 1);
  PROP_SET(psi_dot_bounds, Bounds, psi_dot_bounds, this->horizon_ + 1);

  PROP_SET(delta_bounds, Bounds, delta_bounds, this->horizon_);
  PROP_SET(delta_dot_bounds, Bounds, delta_dot_bounds, this->horizon_);

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

  void InitAngularVelocity(
      const google::protobuf::RepeatedField<double> &kappa,
      const google::protobuf::RepeatedField<double> &linear_vel,
      const double num, std::vector<double> *const diag_matrix);

  void CompensateSlackWeights(std::vector<double> *weights);

  void CompensateSlackConstraints(const bool is_negative, const size_t size,
                                  std::vector<double> *constraints);

  void InitStateMatrices(int k, Eigen::MatrixXd *matrix_A_k,
                         Eigen::MatrixXd *matrix_B_k,
                         Eigen::MatrixXd *matrix_B_tilde);

  void ExtractSolution(OSQPSolution *osqp_solution,
                       OSQPInt num_of_var) override;

 protected:
  size_t num_of_state_;
  size_t num_of_control_;
  size_t num_of_slack_var_;

  double l_init_;
  double l_dot_init_;
  double psi_init_;
  double psi_dot_init_;

  double prev_delta_;

  const LateralMPCConfig &lateral_mpc_config_;

  std::vector<double> diag_matrix_q_;
  std::vector<double> diag_matrix_r_;
  std::vector<double> diag_matrix_r_dot_;

  std::vector<double> angular_vel_;

  Eigen::MatrixXd matrix_q_n_;

  std::vector<double> diag_matrix_w_l_slack_l_;
  std::vector<double> diag_matrix_w_l_slack_u_;
  std::vector<double> diag_matrix_w_l_dot_slack_l_;
  std::vector<double> diag_matrix_w_l_dot_slack_u_;
  std::vector<double> diag_matrix_w_psi_slack_l_;
  std::vector<double> diag_matrix_w_psi_slack_u_;
  std::vector<double> diag_matrix_w_psi_dot_slack_l_;
  std::vector<double> diag_matrix_w_psi_dot_slack_u_;

  std::vector<double> l_ref_;
  std::vector<double> l_dot_ref_;
  std::vector<double> psi_ref_;
  std::vector<double> psi_dot_ref_;

  std::vector<double> l_slack_u_;
  std::vector<double> l_dot_slack_u_;
  std::vector<double> l_slack_l_;
  std::vector<double> l_dot_slack_l_;
  std::vector<double> psi_slack_u_;
  std::vector<double> psi_dot_slack_u_;
  std::vector<double> psi_slack_l_;
  std::vector<double> psi_dot_slack_l_;
  Bounds l_bounds_;
  Bounds l_dot_bounds_;
  Bounds psi_bounds_;
  Bounds psi_dot_bounds_;
  Bounds delta_bounds_;
  Bounds delta_dot_bounds_;

  std::vector<double> opt_l_;
  std::vector<double> opt_l_dot_;
  std::vector<double> opt_psi_;
  std::vector<double> opt_psi_dot_;
  std::vector<double> opt_delta_;
};
}  // namespace mpc