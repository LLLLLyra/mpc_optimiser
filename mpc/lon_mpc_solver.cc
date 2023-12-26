#include "lon_mpc_solver.h"

#include "glog/logging.h"

namespace mpc {
LongitudinalMPCSolver::LongitudinalMPCSolver(const MPCConfig& config)
    : MPCSolver(config) {
  num_of_state_ = 2;
  num_of_control_ = 1;
  num_of_slack_var_ = 4;

  const auto& lon_mpc_config = config_.lon_mpc_config();

  s_init_ = lon_mpc_config.s_init();
  ds_init_ = lon_mpc_config.ds_init();
  prev_dds_ = lon_mpc_config.prev_dds();

  InitWeights(lon_mpc_config.matrix_q(), num_of_state_, &diag_matrix_q_);
  InitWeights(lon_mpc_config.matrix_r(), num_of_control_, &diag_matrix_r_);
  InitWeights(lon_mpc_config.matrix_r_dot(), num_of_control_,
              &diag_matrix_r_dot_);

  // TODO: init (I + A_c Delta_t); B_c Delta_t
}

void LongitudinalMPCSolver::InitWeights(
    const google::protobuf::RepeatedField<double>& matrix, const double num,
    std::vector<double>* const diag_matrix) {
  CHECK(matrix.size() == num || matrix.size() == 1);
  diag_matrix->resize(num);
  if (matrix.size() == 1) {
    std::fill_n(diag_matrix->begin(), num, matrix[0]);
  } else {
    diag_matrix->assign(matrix.begin(), matrix.end());
  }
}

void LongitudinalMPCSolver::CompensateSlackWeights(
    std::vector<double>* weights) {
  if (weights->empty()) {
    weights->resize(horizon_ + 1);
    std::fill_n(weights->begin(), horizon_ + 1, 0.0);
  }

  CHECK(weights->size() == horizon_ + 1);
}

void LongitudinalMPCSolver::CalculateKernel(std::vector<OSQPFloat>* P_data,
                                            std::vector<OSQPInt>* P_indices,
                                            std::vector<OSQPInt>* P_indptr) {
  const size_t kNumParam = num_of_state_ * (horizon_ + 1) +
                           num_of_state_ * horizon_ +
                           num_of_slack_var_ * (horizon_ + 1);

  CompensateSlackWeights(&diag_matrix_w_s_slack_l_);
  CompensateSlackWeights(&diag_matrix_w_s_slack_u_);
  CompensateSlackWeights(&diag_matrix_w_ds_slack_l_);
  CompensateSlackWeights(&diag_matrix_w_ds_slack_u_);

  std::vector<std::vector<std::pair<OSQPInt, OSQPFloat>>> columns;
  columns.resize(kNumParam);
  int index = 0;

  // states
  for (size_t i = 0; i < horizon_; ++i) {
    for (size_t j = 0; j < num_of_state_; ++j) {
      columns[index].emplace_back(index, diag_matrix_q_[j]);
      index++;
    }
  }

  // TODO: Q_n

  // control
  const double dt_squared = delta_t_ * delta_t_;
  columns[num_of_state_ * (horizon_ + 1)].emplace_back(
      num_of_state_ * (horizon_ + 1),
      diag_matrix_r_[0] + diag_matrix_r_dot_[0] / dt_squared);
  columns[num_of_state_ * (horizon_ + 1)].emplace_back(
      num_of_state_ * (horizon_ + 1) + 1, -diag_matrix_r_dot_[0] / dt_squared);

  for (size_t i = 1; i + 1 < horizon_; ++i) {
    columns[i + num_of_state_ * (horizon_ + 1)].emplace_back(
        i + num_of_state_ * (horizon_ + 1),
        diag_matrix_r_[0] + 2.0 * diag_matrix_r_dot_[0] / dt_squared);
    columns[i + num_of_state_ * (horizon_ + 1)].emplace_back(
        i + num_of_state_ * (horizon_ + 1) - 1,
        -diag_matrix_r_dot_[0] / dt_squared);
    columns[i + num_of_state_ * (horizon_ + 1)].emplace_back(
        i + num_of_state_ * (horizon_ + 1) + 1,
        -diag_matrix_r_dot_[0] / dt_squared);
  }

  columns[num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ - 1]
      .emplace_back(
          num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ - 1,
          diag_matrix_r_[0] + diag_matrix_r_dot_[0] / dt_squared);
  columns[num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ - 1]
      .emplace_back(
          num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ - 2,
          -diag_matrix_r_dot_[0] / dt_squared);

  // slack var
  index = num_of_state_ * (horizon_ + 1) + num_of_state_ * horizon_;
  for (size_t i = 0; i < horizon_ + 1; ++i) {
    if (diag_matrix_w_s_slack_l_[i] != 0.0) {
      columns[index + i].emplace_back(index + i, diag_matrix_w_s_slack_l_[i]);
    }
    if (diag_matrix_w_s_slack_u_[i] != 0.0) {
      columns[index + i + horizon_ + 1].emplace_back(
          index + i + horizon_ + 1, diag_matrix_w_s_slack_u_[i]);
    }
    if (diag_matrix_w_ds_slack_l_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 2].emplace_back(
          index + i + (horizon_ + 1) * 2, diag_matrix_w_ds_slack_l_[i]);
    }
    if (diag_matrix_w_ds_slack_u_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 3].emplace_back(
          index + i + (horizon_ + 1) * 3, diag_matrix_w_ds_slack_u_[i]);
    }
    ++index;
  }

  CSCInitHelper<double>(columns, P_data, P_indices, P_indptr);
}

void LongitudinalMPCSolver::CalculateOffset(std::vector<OSQPFloat>* q) {
  const size_t kNumParam = num_of_state_ * (horizon_ + 1) +
                           num_of_state_ * horizon_ +
                           num_of_slack_var_ * (horizon_ + 1);
  q->resize(kNumParam);
  int index = 0;
  for (size_t i = 0; i < horizon_; i++) {
    q->at(index++) = s_ref_[i];
    q->at(index++) = ds_ref_[i];
  }

  // TODO Q_n;

  q->at(num_of_state_ * (horizon_ + 1)) =
      -diag_matrix_r_dot_[0] * prev_dds_ / delta_t_ / delta_t_;
}

}  // namespace mpc