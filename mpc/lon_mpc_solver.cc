#include "lon_mpc_solver.h"

#include "utils/math_utils/dare.h"

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

  num_of_var_ = num_of_state_ * (horizon_ + 1) + num_of_state_ * horizon_ +
                num_of_slack_var_ * (horizon_ + 1);

  InitWeights(lon_mpc_config.matrix_q(), num_of_state_, &diag_matrix_q_);
  InitWeights(lon_mpc_config.matrix_r(), num_of_control_, &diag_matrix_r_);
  InitWeights(lon_mpc_config.matrix_r_dot(), num_of_control_,
              &diag_matrix_r_dot_);

  InitStateMatrices();
  math_utils::Dare(matrix_A_k_, matrix_B_k_, diag_matrix_q_, diag_matrix_r_,
                   lon_mpc_config.dare_tol(), lon_mpc_config.dare_max_itr(),
                   &matrix_q_n_);
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

void LongitudinalMPCSolver::InitStateMatrices() {
  Eigen::MatrixXd A_c(2, 2), B_c(2, 1);
  A_c << 0, 1, 0, 0;
  B_c << 0, 1;
  matrix_A_k_ = A_c * delta_t_ + Eigen::Matrix2d::Identity();
  matrix_B_k_ = B_c * delta_t_;
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

  // Q_n
  columns[index].emplace_back(index, matrix_q_n_(0, 0));
  columns[index].emplace_back(index + 1, matrix_q_n_(1, 0));
  index++;
  columns[index].emplace_back(index, matrix_q_n_(1, 1));
  columns[index].emplace_back(index - 1, matrix_q_n_(0, 1));

  // control
  const double dt_squared = delta_t_ * delta_t_;
  columns[num_of_state_ * (horizon_ + 1)].emplace_back(
      num_of_state_ * (horizon_ + 1),
      diag_matrix_r_[0] + 2.0 * diag_matrix_r_dot_[0] / dt_squared);
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
    if (diag_matrix_w_s_slack_u_[i] != 0.0) {
      columns[index + i].emplace_back(index + i, diag_matrix_w_s_slack_u_[i]);
    }
    if (diag_matrix_w_ds_slack_u_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 2].emplace_back(
          index + i + (horizon_ + 1) * 2, diag_matrix_w_ds_slack_u_[i]);
    }
    if (diag_matrix_w_s_slack_l_[i] != 0.0) {
      columns[index + i + horizon_ + 1].emplace_back(
          index + i + horizon_ + 1, diag_matrix_w_s_slack_l_[i]);
    }
    if (diag_matrix_w_ds_slack_l_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 3].emplace_back(
          index + i + (horizon_ + 1) * 3, diag_matrix_w_ds_slack_l_[i]);
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

  Eigen::MatrixXd x_n_ref(2, 1);
  x_n_ref << s_ref_.back(), ds_ref_.back();
  auto q_n_ref = matrix_q_n_ * x_n_ref;
  q->at(index++) = q_n_ref(0, 0);
  q->at(index++) = q_n_ref(1, 0);

  CHECK_EQ(index, num_of_state_ * (horizon_ + 1));

  q->at(index) = -diag_matrix_r_dot_[0] * prev_dds_ / delta_t_ / delta_t_;
}

void LongitudinalMPCSolver::CalculateAffineConstraint(
    std::vector<OSQPFloat>* A_data, std::vector<OSQPInt>* A_indices,
    std::vector<OSQPInt>* A_indptr, std::vector<OSQPFloat>* lower_bounds,
    std::vector<OSQPFloat>* upper_bounds) {
  const size_t kNumParam = num_of_state_ * (horizon_ + 1) +
                           num_of_control_ * horizon_ +
                           num_of_slack_var_ * (horizon_ + 1);
  const size_t kNumIneqConstraints = num_of_state_ * (horizon_ + 1) * 2 +
                                     num_of_control_ * horizon_ * 2 +
                                     2 * (horizon_ + 1);
  const size_t kNumEqConstraints = num_of_state_ * (horizon_ + 1);
  const size_t kNumConstraints = kNumIneqConstraints + kNumEqConstraints;
  lower_bounds->resize(kNumConstraints);
  upper_bounds->resize(kNumConstraints);

  if (s_slack_u_.empty()) {
    s_slack_u_.resize(horizon_ + 1);
    std::fill_n(s_slack_u_.begin(), horizon_ + 1, OSQP_INFTY);
  }

  if (ds_slack_u_.empty()) {
    ds_slack_u_.resize(horizon_ + 1);
    std::fill_n(ds_slack_u_.begin(), horizon_ + 1, OSQP_INFTY);
  }

  std::vector<std::vector<std::pair<OSQPInt, OSQPFloat>>> variables(kNumParam);

  int constraint_index = 0;
  int slack_start_index =
      num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_;
  int slack_l_start_index = slack_start_index + 2 * (horizon_ + 1);
  for (size_t i = 0; i < kNumParam;) {
    // x [s, v]
    if (i < num_of_state_ * (horizon_ + 1)) {
      int index = i / num_of_state_;
      // x_upper
      variables[i].emplace_back(constraint_index, 1.0);
      upper_bounds->at(constraint_index) = x_bounds_[index].second;
      lower_bounds->at(constraint_index) = -OSQP_INFTY;

      // slack_x_u
      if (diag_matrix_w_s_slack_u_[index] != 0.0) {
        variables[slack_start_index + i].emplace_back(constraint_index, -1.0);
      }

      // x_lower
      variables[i].emplace_back(constraint_index + 1, 1.0);
      lower_bounds->at(constraint_index + 1) = x_bounds_[index].first;
      upper_bounds->at(constraint_index + 1) = OSQP_INFTY;

      // slack_x_l
      if (diag_matrix_w_s_slack_l_[index] != 0.0) {
        variables[slack_l_start_index + i].emplace_back(constraint_index + 1,
                                                        1.0);
      }

      constraint_index += 2;

      // dx_upper
      variables[i + 1].emplace_back(constraint_index, 1.0);
      upper_bounds->at(constraint_index) = dx_bounds_[index].second;
      lower_bounds->at(constraint_index) = -OSQP_INFTY;

      // slack_dx_u
      if (diag_matrix_w_ds_slack_u_[index] != 0.0) {
        variables[slack_start_index + i + (horizon_ + 1)].emplace_back(
            constraint_index, -1.0);
      }

      // dx_lower
      variables[i + 1].emplace_back(constraint_index + 1, 1.0);
      lower_bounds->at(constraint_index + 1) = dx_bounds_[index].first;
      upper_bounds->at(constraint_index + 1) = OSQP_INFTY;

      // slack_dx_l
      if (diag_matrix_w_ds_slack_l_[index] != 0.0) {
        variables[slack_l_start_index + i + (horizon_ + 1)].emplace_back(
            constraint_index + 1, 1.0);
      }

      constraint_index += 2;
      i += 2;
      continue;
    } else if (i <
               num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_) {
      // a
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) =
          ddx_bounds_[i - num_of_state_ * (horizon_ + 1)].first;
      upper_bounds->at(constraint_index) =
          ddx_bounds_[i - num_of_state_ * (horizon_ + 1)].second;
    } else if (i < slack_start_index + (horizon_ + 1)) {
      // s_slack_u
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) = s_slack_u_[i - slack_start_index];
    } else if (i < slack_start_index + 2.0 * (horizon_ + 1)) {
      // ds_slack_u
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) =
          ds_slack_u_[i - slack_start_index - (horizon_ + 1)];
    } else {
      break;
    }
    ++constraint_index;
    ++i;
  }

  // u_dot_k = (u_k - u_k-1) / delta_t
  for (int i = 0; i < horizon_; ++i) {
    if (i == 0) {
      variables[num_of_state_ * (horizon_ + 1) + i].emplace_back(
          constraint_index, 1.0);
      lower_bounds->at(constraint_index) =
          prev_dds_ + dddx_bounds_[i].first * delta_t_;
      upper_bounds->at(constraint_index) =
          prev_dds_ + dddx_bounds_[i].second * delta_t_;
    } else {
      variables[num_of_state_ * (horizon_ + 1) + i].emplace_back(
          constraint_index, 1.0);
      variables[num_of_state_ * (horizon_ + 1) + i - 1].emplace_back(
          constraint_index, -1.0);
      lower_bounds->at(constraint_index) = dddx_bounds_[i].first * delta_t_;
      upper_bounds->at(constraint_index) = dddx_bounds_[i].second * delta_t_;
    }
    ++constraint_index;
  }

  CHECK_EQ(constraint_index, kNumIneqConstraints);

  // init states
  variables[0].emplace_back(constraint_index, -1);
  lower_bounds->at(constraint_index) = -s_init_;
  upper_bounds->at(constraint_index) = -s_init_;
  ++constraint_index;

  variables[1].emplace_back(constraint_index, -1);
  lower_bounds->at(constraint_index) = -ds_init_;
  upper_bounds->at(constraint_index) = -ds_init_;
  ++constraint_index;

  // x_k+1 = Ak * x_k + Bk * u_k;
  for (int i = 2; i < num_of_state_ * (horizon_ + 1); i += num_of_state_) {
    int k = i / num_of_state_;
    // s
    variables[i].emplace_back(constraint_index, -1.0);
    variables[i - 2].emplace_back(constraint_index, 1.0);
    variables[i - 1].emplace_back(constraint_index, delta_t_);
    lower_bounds->at(constraint_index) = 0.0;
    upper_bounds->at(constraint_index) = 0.0;

    // v
    variables[i + 1].emplace_back(constraint_index + 1, -1.0);
    variables[i - 1].emplace_back(constraint_index + 1, 1.0);
    variables[k + num_of_state_ * (horizon_ + 1)].emplace_back(
        constraint_index + 1, delta_t_);

    lower_bounds->at(constraint_index + 1) = 0.0;
    upper_bounds->at(constraint_index + 1) = 0.0;

    constraint_index += 2;
  }

  CHECK_EQ(constraint_index, kNumConstraints);

  CSCInitHelper(variables, A_data, A_indices, A_indptr);
}

}  // namespace mpc