#pragma once

#include "mpc_solver.h"

namespace mpc {
class LongitudinalMPCSolver : public MPCSolver {
 public:
  LongitudinalMPCSolver(const MPCConfig &config);

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

 protected:
  size_t num_of_state_;
  size_t num_of_control_;
  size_t num_of_slack_var_;

  double s_init_;
  double ds_init_;
  double prev_dds_;

  std::vector<double> diag_matrix_q_;
  std::vector<double> diag_matrix_r_;
  std::vector<double> diag_matrix_r_dot_;
  std::vector<double> diag_matrix_w_s_slack_l_;
  std::vector<double> diag_matrix_w_s_slack_u_;
  std::vector<double> diag_matrix_w_ds_slack_l_;
  std::vector<double> diag_matrix_w_ds_slack_u_;

  std::vector<double> s_ref_;
  std::vector<double> ds_ref_;
};
}  // namespace mpc