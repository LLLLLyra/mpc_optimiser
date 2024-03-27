#include "lateral_mpc_solver.h"

#include "utils/math_utils/dare.h"

namespace mpc {
LateralMPCSolver::LateralMPCSolver(const MPCConfig& config)
    : MPCSolver(config), lateral_mpc_config_(config.lat_mpc_config()) {
  num_of_state_ = 4;
  num_of_control_ = 1;
  num_of_slack_var_ = 8;

  l_init_ = lateral_mpc_config_.l_init();
  l_dot_init_ = lateral_mpc_config_.l_dot_init();
  psi_init_ = lateral_mpc_config_.psi_init();
  psi_dot_init_ = lateral_mpc_config_.psi_dot_init();
  prev_delta_ = lateral_mpc_config_.prev_delta();

  num_of_var_ = num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ +
                num_of_slack_var_ * (horizon_ + 1);

  InitWeights(lateral_mpc_config_.matrix_q(), num_of_state_, &diag_matrix_q_);
  InitWeights(lateral_mpc_config_.matrix_r(), num_of_control_, &diag_matrix_r_);
  InitWeights(lateral_mpc_config_.matrix_r_dot(), num_of_control_,
              &diag_matrix_r_dot_);

  InitAngularVelocity(lateral_mpc_config_.kappa(),
                      lateral_mpc_config_.velocity(), horizon_ + 1,
                      &angular_vel_);

  Eigen::MatrixXd matrix_A_k;
  Eigen::MatrixXd matrix_B_k;
  Eigen::MatrixXd matrix_B_tilde_k;
  InitStateMatrices(horizon_, &matrix_A_k, &matrix_B_k, &matrix_B_tilde_k);
  math_utils::Dare(matrix_A_k, matrix_B_k, diag_matrix_q_, diag_matrix_r_,
                   lateral_mpc_config_.dare_tol(),
                   lateral_mpc_config_.dare_max_itr(), &matrix_q_n_);
}

void LateralMPCSolver::InitWeights(
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

void LateralMPCSolver::InitAngularVelocity(
    const google::protobuf::RepeatedField<double>& kappa,
    const google::protobuf::RepeatedField<double>& linear_vel, const double num,
    std::vector<double>* const diag_matrix) {
  CHECK(kappa.size() == linear_vel.size());
  CHECK(kappa.size() == num);
  angular_vel_.clear();
  angular_vel_.reserve(num);
  for (int i = 0; i < num; ++i) {
    angular_vel_.emplace_back(kappa[i] * linear_vel[i]);
  }
}

void LateralMPCSolver::CompensateSlackWeights(std::vector<double>* weights) {
  if (weights->empty()) {
    weights->resize(horizon_ + 1);
    std::fill_n(weights->begin(), horizon_ + 1, 0.0);
  }

  CHECK(weights->size() == horizon_ + 1);
}

void LateralMPCSolver::InitStateMatrices(int k, Eigen::MatrixXd* matrix_A_k,
                                         Eigen::MatrixXd* matrix_B_k,
                                         Eigen::MatrixXd* matrix_B_tilde) {
  Eigen::MatrixXd Ac(4, 4), Bc(4, 1), Bc_tidle(4, 1);

  double C_af = lateral_mpc_config_.c_af();
  double C_ar = lateral_mpc_config_.c_ar();
  double l_f = lateral_mpc_config_.l_f();
  double l_r = lateral_mpc_config_.l_r();
  double I_z = lateral_mpc_config_.i_z();
  double m = lateral_mpc_config_.m();
  double v_k = lateral_mpc_config_.velocity(k);

  Ac << 0, 1, 0, 0,  // NOLINT
      0, -(C_af + C_ar) / (m * v_k), (C_af + C_ar) / m,
      (-C_af * l_f + C_ar * l_r) / (m * v_k),  // NOLINT
      0, 0, 0, 1,                              // NOLINT
      0, -(C_af * l_f - C_ar * l_r) / (I_z * v_k),
      (C_af * l_f - C_ar * l_r) / I_z,
      -(C_af * l_f * l_f + C_ar * l_r * l_r) / (I_z * v_k);  // NOLINT

  Bc << 0, C_af / m, 0, C_af * l_f / I_z;

  Bc_tidle << 0, -(C_af * l_f - C_ar * l_r) / (m * v_k) - v_k, 0,
      -(C_af * l_f * l_f - C_ar * l_r * l_r) / (I_z * v_k);

  *matrix_A_k = (2.0 * Eigen::MatrixXd::Identity(num_of_state_, num_of_state_) -
                 Ac * delta_t_)
                    .inverse() *
                (2.0 * Eigen::MatrixXd::Identity(num_of_state_, num_of_state_) +
                 Ac * delta_t_);
  *matrix_B_k = (2.0 * Eigen::MatrixXd::Identity(num_of_state_, num_of_state_) -
                 Ac * delta_t_)
                    .inverse() *
                Bc * delta_t_;
  *matrix_B_tilde =
      (2.0 * Eigen::MatrixXd::Identity(num_of_state_, num_of_state_) -
       Ac * delta_t_)
          .inverse() *
      Bc_tidle * delta_t_;
}

void LateralMPCSolver::CalculateKernel(std::vector<OSQPFloat>* P_data,
                                       std::vector<OSQPInt>* P_indices,
                                       std::vector<OSQPInt>* P_indptr) {
  const size_t kNumParam = num_of_state_ * (horizon_ + 1) +
                           num_of_control_ * horizon_ +
                           num_of_slack_var_ * (horizon_ + 1);

  CompensateSlackWeights(&diag_matrix_w_l_slack_l_);
  CompensateSlackWeights(&diag_matrix_w_l_slack_u_);
  CompensateSlackWeights(&diag_matrix_w_l_dot_slack_l_);
  CompensateSlackWeights(&diag_matrix_w_l_dot_slack_u_);
  CompensateSlackWeights(&diag_matrix_w_psi_slack_l_);
  CompensateSlackWeights(&diag_matrix_w_psi_slack_u_);
  CompensateSlackWeights(&diag_matrix_w_psi_dot_slack_l_);
  CompensateSlackWeights(&diag_matrix_w_psi_dot_slack_u_);

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
  for (int row = 0; row < num_of_state_; ++row) {
    for (int col = row; col < num_of_state_; ++col) {
      columns[index + col].emplace_back(index + row, matrix_q_n_(row, col));
    }
  }
  index += num_of_state_;

  CHECK_EQ(index, num_of_state_ * (horizon_ + 1));

  // control
  const double dt_squared = delta_t_ * delta_t_;
  columns[num_of_state_ * (horizon_ + 1)].emplace_back(
      num_of_state_ * (horizon_ + 1),
      diag_matrix_r_[0] + 2.0 * diag_matrix_r_dot_[0] / dt_squared);

  for (size_t i = 1; i + 1 < horizon_; ++i) {
    columns[i + num_of_state_ * (horizon_ + 1)].emplace_back(
        i + num_of_state_ * (horizon_ + 1) - 1,
        -diag_matrix_r_dot_[0] / dt_squared);
    columns[i + num_of_state_ * (horizon_ + 1)].emplace_back(
        i + num_of_state_ * (horizon_ + 1),
        diag_matrix_r_[0] + 2.0 * diag_matrix_r_dot_[0] / dt_squared);
  }

  columns[num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ - 1]
      .emplace_back(
          num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ - 2,
          -diag_matrix_r_dot_[0] / dt_squared);

  columns[num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ - 1]
      .emplace_back(
          num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_ - 1,
          diag_matrix_r_[0] + diag_matrix_r_dot_[0] / dt_squared);

  // slack var
  index = num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_;
  for (size_t i = 0; i < horizon_ + 1; ++i) {
    if (diag_matrix_w_l_slack_u_[i] != 0.0) {
      columns[index + i].emplace_back(index + i, diag_matrix_w_l_slack_u_[i]);
    }
    if (diag_matrix_w_l_dot_slack_u_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 1].emplace_back(
          index + i + (horizon_ + 1) * 1, diag_matrix_w_l_dot_slack_u_[i]);
    }
    if (diag_matrix_w_psi_slack_u_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 2].emplace_back(
          index + i + (horizon_ + 1) * 2, diag_matrix_w_psi_slack_u_[i]);
    }
    if (diag_matrix_w_psi_dot_slack_u_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 3].emplace_back(
          index + i + (horizon_ + 1) * 3, diag_matrix_w_psi_dot_slack_u_[i]);
    }

    if (diag_matrix_w_l_slack_l_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 4].emplace_back(
          index + i + (horizon_ + 1) * 4, diag_matrix_w_l_slack_l_[i]);
    }
    if (diag_matrix_w_l_dot_slack_l_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 5].emplace_back(
          index + i + (horizon_ + 1) * 5, diag_matrix_w_l_dot_slack_l_[i]);
    }
    if (diag_matrix_w_psi_slack_l_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 6].emplace_back(
          index + i + (horizon_ + 1) * 6, diag_matrix_w_psi_slack_l_[i]);
    }
    if (diag_matrix_w_psi_dot_slack_l_[i] != 0.0) {
      columns[index + i + (horizon_ + 1) * 7].emplace_back(
          index + i + (horizon_ + 1) * 7, diag_matrix_w_psi_dot_slack_l_[i]);
    }
  }

  CSCInitHelper<double>(columns, P_data, P_indices, P_indptr);
}

void LateralMPCSolver::CalculateOffset(std::vector<OSQPFloat>* q) {
  const size_t kNumParam = num_of_state_ * (horizon_ + 1) +
                           num_of_control_ * horizon_ +
                           num_of_slack_var_ * (horizon_ + 1);
  q->resize(kNumParam);
  int index = 0;
  for (size_t i = 0; i < horizon_; i++) {
    q->at(index++) = l_ref_[i];
    q->at(index++) = l_dot_ref_[i];
    q->at(index++) = psi_ref_[i];
    q->at(index++) = psi_dot_ref_[i];
  }

  Eigen::MatrixXd x_n_ref(num_of_state_, 1);
  x_n_ref << l_ref_.back(), l_dot_ref_.back(), psi_ref_.back(),
      psi_dot_ref_.back();
  auto q_n_ref = matrix_q_n_ * x_n_ref;
  q->at(index++) = q_n_ref(0, 0);
  q->at(index++) = q_n_ref(1, 0);
  q->at(index++) = q_n_ref(2, 0);
  q->at(index++) = q_n_ref(3, 0);

  CHECK_EQ(index, num_of_state_ * (horizon_ + 1));

  q->at(index) = -diag_matrix_r_dot_[0] * prev_delta_ / delta_t_ / delta_t_;
}

void LateralMPCSolver::CompensateSlackConstraints(
    const bool is_negative, const size_t size,
    std::vector<double>* constraints) {
  if (constraints->empty()) {
    constraints->resize(size);
    std::fill_n(constraints->begin(), horizon_ + 1,
                is_negative ? -OSQP_INFTY : OSQP_INFTY);
  }
}

void LateralMPCSolver::CalculateAffineConstraint(
    std::vector<OSQPFloat>* A_data, std::vector<OSQPInt>* A_indices,
    std::vector<OSQPInt>* A_indptr, std::vector<OSQPFloat>* lower_bounds,
    std::vector<OSQPFloat>* upper_bounds) {
  const size_t kNumParam = num_of_state_ * (horizon_ + 1) +
                           num_of_control_ * horizon_ +
                           num_of_slack_var_ * (horizon_ + 1);
  const size_t kNumIneqConstraints = num_of_state_ * (horizon_ + 1) * 4 +
                                     num_of_control_ * horizon_ * 2 +
                                     num_of_slack_var_ * (horizon_ + 1);
  const size_t kNumEqConstraints = num_of_state_ * (horizon_ + 1);
  const size_t kNumConstraints = kNumIneqConstraints + kNumEqConstraints;
  lower_bounds->resize(kNumConstraints);
  upper_bounds->resize(kNumConstraints);

  CompensateSlackConstraints(true, horizon_ + 1, &l_slack_l_);
  CompensateSlackConstraints(true, horizon_ + 1, &l_dot_slack_l_);
  CompensateSlackConstraints(true, horizon_ + 1, &psi_slack_l_);
  CompensateSlackConstraints(true, horizon_ + 1, &psi_dot_slack_l_);

  CompensateSlackConstraints(false, horizon_ + 1, &l_slack_u_);
  CompensateSlackConstraints(false, horizon_ + 1, &l_dot_slack_u_);
  CompensateSlackConstraints(false, horizon_ + 1, &psi_slack_u_);
  CompensateSlackConstraints(false, horizon_ + 1, &psi_dot_slack_u_);

  std::vector<std::vector<std::pair<OSQPInt, OSQPFloat>>> variables(kNumParam);

  int constraint_index = 0;
  int slack_start_index =
      num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_;
  int slack_l_start_index =
      slack_start_index + num_of_slack_var_ * (horizon_ + 1);
  for (size_t i = 0; i < kNumParam;) {
    // x
    if (i < num_of_state_ * (horizon_ + 1)) {
      int index = i / num_of_state_;
      // l_upper
      variables[i].emplace_back(constraint_index, 1.0);
      upper_bounds->at(constraint_index) = l_bounds_[index].second;
      lower_bounds->at(constraint_index) = -OSQP_INFTY;

      // slack_l_u
      if (diag_matrix_w_l_slack_u_[index] != 0.0) {
        variables[slack_start_index + i].emplace_back(constraint_index, -1.0);
      }

      // l_lower
      variables[i].emplace_back(constraint_index + 1, 1.0);
      lower_bounds->at(constraint_index + 1) = l_bounds_[index].first;
      upper_bounds->at(constraint_index + 1) = OSQP_INFTY;

      // slack_l_l
      if (diag_matrix_w_l_slack_l_[index] != 0.0) {
        variables[slack_l_start_index + i].emplace_back(constraint_index + 1,
                                                        1.0);
      }

      constraint_index += 2;

      // dl_upper
      variables[i + 1].emplace_back(constraint_index, 1.0);
      upper_bounds->at(constraint_index) = l_dot_bounds_[index].second;
      lower_bounds->at(constraint_index) = -OSQP_INFTY;

      // slack_dl_u
      if (diag_matrix_w_l_dot_slack_u_[index] != 0.0) {
        variables[slack_start_index + i + (horizon_ + 1)].emplace_back(
            constraint_index, -1.0);
      }

      // dl_lower
      variables[i + 1].emplace_back(constraint_index + 1, 1.0);
      lower_bounds->at(constraint_index + 1) = l_dot_bounds_[index].first;
      upper_bounds->at(constraint_index + 1) = OSQP_INFTY;

      // slack_dl_l
      if (diag_matrix_w_l_dot_slack_l_[index] != 0.0) {
        variables[slack_l_start_index + i + (horizon_ + 1)].emplace_back(
            constraint_index + 1, 1.0);
      }

      constraint_index += 2;

      // psi_upper
      variables[i + 2].emplace_back(constraint_index, 1.0);
      upper_bounds->at(constraint_index) = psi_bounds_[index].second;
      lower_bounds->at(constraint_index) = -OSQP_INFTY;

      // slack_l_u
      if (diag_matrix_w_psi_slack_u_[index] != 0.0) {
        variables[slack_start_index + i + (horizon_ + 1) * 2].emplace_back(
            constraint_index, -1.0);
      }

      // psi_lower
      variables[i + 2].emplace_back(constraint_index + 1, 1.0);
      lower_bounds->at(constraint_index + 1) = psi_bounds_[index].first;
      upper_bounds->at(constraint_index + 1) = OSQP_INFTY;

      // slack_psi_l
      if (diag_matrix_w_psi_slack_l_[index] != 0.0) {
        variables[slack_l_start_index + i + (horizon_ + 1) * 2].emplace_back(
            constraint_index + 1, 1.0);
      }

      constraint_index += 2;

      // dpsi_upper
      variables[i + 3].emplace_back(constraint_index, 1.0);
      upper_bounds->at(constraint_index) = psi_dot_bounds_[index].second;
      lower_bounds->at(constraint_index) = -OSQP_INFTY;

      // slack_dpsi_u
      if (diag_matrix_w_psi_dot_slack_u_[index] != 0.0) {
        variables[slack_start_index + i + (horizon_ + 1) * 3].emplace_back(
            constraint_index, -1.0);
      }

      // dpsi_lower
      variables[i + 3].emplace_back(constraint_index + 1, 1.0);
      lower_bounds->at(constraint_index + 1) = psi_dot_bounds_[index].first;
      upper_bounds->at(constraint_index + 1) = OSQP_INFTY;

      // slack_dl_l
      if (diag_matrix_w_psi_dot_slack_l_[index] != 0.0) {
        variables[slack_l_start_index + i + (horizon_ + 1) * 3].emplace_back(
            constraint_index + 1, 1.0);
      }

      constraint_index += 2;

      i += num_of_state_;
      continue;
    } else if (i <
               num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_) {
      // delta
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) =
          delta_bounds_[i - num_of_state_ * (horizon_ + 1)].first;
      upper_bounds->at(constraint_index) =
          delta_bounds_[i - num_of_state_ * (horizon_ + 1)].second;
    } else if (i < slack_start_index + (horizon_ + 1)) {
      // l_slack_u
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) = l_slack_u_[i - slack_start_index];
    } else if (i < slack_start_index + 2 * (horizon_ + 1)) {
      // dl_slack_u
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) =
          l_dot_slack_u_[i - slack_start_index - (horizon_ + 1)];
    } else if (i < slack_start_index + (horizon_ + 1) * 2) {
      // psi_slack_u
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) =
          psi_slack_u_[i - slack_start_index - (horizon_ + 1) * 2];
    } else if (i < slack_start_index + (horizon_ + 1) * 3) {
      // dpsi_slack_u
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) =
          psi_dot_slack_u_[i - slack_start_index - (horizon_ + 1) * 3];
    } else if (i < slack_l_start_index + (horizon_ + 1)) {
      // l_slack_l
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) = l_slack_l_[i - slack_l_start_index];
    } else if (i < slack_l_start_index + 2 * (horizon_ + 1)) {
      // dl_slack_l
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) =
          l_dot_slack_l_[i - slack_l_start_index - (horizon_ + 1)];
    } else if (i < slack_l_start_index + (horizon_ + 1) * 2) {
      // psi_slack_l
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) =
          psi_slack_l_[i - slack_l_start_index - (horizon_ + 1) * 2];
    } else if (i < slack_l_start_index + (horizon_ + 1) * 3) {
      // dpsi_slack_l
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) = 0.0;
      upper_bounds->at(constraint_index) =
          psi_dot_slack_l_[i - slack_l_start_index - (horizon_ + 1) * 3];
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
          prev_delta_ + delta_dot_bounds_[i].first * delta_t_;
      upper_bounds->at(constraint_index) =
          prev_delta_ + delta_dot_bounds_[i].second * delta_t_;
    } else {
      variables[num_of_state_ * (horizon_ + 1) + i].emplace_back(
          constraint_index, 1.0);
      variables[num_of_state_ * (horizon_ + 1) + i - 1].emplace_back(
          constraint_index, -1.0);
      lower_bounds->at(constraint_index) =
          delta_dot_bounds_[i].first * delta_t_;
      upper_bounds->at(constraint_index) =
          delta_dot_bounds_[i].second * delta_t_;
    }
    ++constraint_index;
  }

  CHECK_EQ(constraint_index, kNumIneqConstraints);

  // init states
  variables[0].emplace_back(constraint_index, -1);
  lower_bounds->at(constraint_index) = -l_init_;
  upper_bounds->at(constraint_index) = -l_init_;
  ++constraint_index;

  variables[1].emplace_back(constraint_index, -1);
  lower_bounds->at(constraint_index) = -l_dot_init_;
  upper_bounds->at(constraint_index) = -l_dot_init_;
  ++constraint_index;

  variables[2].emplace_back(constraint_index, -1);
  lower_bounds->at(constraint_index) = -psi_init_;
  upper_bounds->at(constraint_index) = -psi_init_;
  ++constraint_index;

  variables[3].emplace_back(constraint_index, -1);
  lower_bounds->at(constraint_index) = -psi_dot_init_;
  upper_bounds->at(constraint_index) = -psi_dot_init_;
  ++constraint_index;

  // x_k+1 = Ak * x_k + Bk * u_k + B'_k * u'_k;
  for (int i = num_of_state_; i < num_of_state_ * (horizon_ + 1);
       i += num_of_state_) {
    int k = i / num_of_state_;

    Eigen::MatrixXd matrix_A_k;
    Eigen::MatrixXd matrix_B_k;
    Eigen::MatrixXd matrix_B_tilde_k;
    InitStateMatrices(k, &matrix_A_k, &matrix_B_k, &matrix_B_tilde_k);

    for (int row = 0; row < num_of_state_; ++row) {
      variables[i].emplace_back(constraint_index + row, -1.0);
      for (int col = 0; col < num_of_state_; ++col) {
        variables[i - num_of_state_ + col].emplace_back(constraint_index + row,
                                                        matrix_A_k(row, col));
      }
      variables[k + num_of_state_ * (horizon_ + 1)].emplace_back(
          constraint_index + row, matrix_B_k(row, 0));

      lower_bounds->at(constraint_index + row) =
          -matrix_B_tilde_k(row, 0) * lateral_mpc_config_.kappa(k) *
          lateral_mpc_config_.velocity(k);
      upper_bounds->at(constraint_index + row) =
          -matrix_B_tilde_k(row, 0) * lateral_mpc_config_.kappa(k) *
          lateral_mpc_config_.velocity(k);
    }

    constraint_index += num_of_state_;
  }

  CHECK_EQ(constraint_index, kNumConstraints);

  CSCInitHelper(variables, A_data, A_indices, A_indptr);
}

void LateralMPCSolver::ExtractSolution(OSQPSolution* osqp_solution,
                                       OSQPInt num_of_var) {
  MPCSolver::ExtractSolution(osqp_solution, num_of_var);

  std::vector<double>().swap(opt_l_);
  std::vector<double>().swap(opt_l_dot_);
  std::vector<double>().swap(opt_psi_);
  std::vector<double>().swap(opt_psi_dot_);
  std::vector<double>().swap(opt_delta_);

  opt_l_.reserve(horizon_ + 1);
  opt_l_dot_.reserve(horizon_ + 1);
  opt_psi_.reserve(horizon_);
  opt_psi_dot_.reserve(horizon_);
  opt_delta_.reserve(horizon_);

  for (size_t i = 0; i < horizon_ + 1; ++i) {
    opt_l_.emplace_back(solution_[i]);
  }
  for (size_t i = horizon_ + 1; i < 2 * (horizon_ + 1); ++i) {
    opt_l_dot_.emplace_back(solution_[i]);
  }
  for (size_t i = 2 * (horizon_ + 1); i < 3 * (horizon_ + 1); ++i) {
    opt_psi_.emplace_back(solution_[i]);
  }
  for (size_t i = 3 * (horizon_ + 1); i < 4 * (horizon_ + 1); ++i) {
    opt_psi_dot_.emplace_back(solution_[i]);
  }
  for (size_t i = num_of_state_ * (horizon_ + 1);
       i < num_of_state_ * (horizon_ + 1) + horizon_; ++i) {
    opt_delta_.emplace_back(solution_[i]);
  }
}
}  // namespace mpc