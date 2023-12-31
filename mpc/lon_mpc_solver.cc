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

  InitWeights(lon_mpc_config.matrix_q(), num_of_state_, &diag_matrix_q_);
  InitWeights(lon_mpc_config.matrix_r(), num_of_control_, &diag_matrix_r_);
  InitWeights(lon_mpc_config.matrix_r_dot(), num_of_control_,
              &diag_matrix_r_dot_);

  InitStateMatrices();
  math_utils::Dare(matrix_A_k_.back(), matrix_B_k_.back(), diag_matrix_q_,
                   diag_matrix_r_, lon_mpc_config.dare_tol(),
                   lon_mpc_config.dare_max_itr(), &matrix_q_n_);
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
  for (size_t i = 0; i < horizon_; ++i) {
    A_c = A_c * delta_t_ + Eigen::Matrix2d::Identity();
    B_c *= delta_t_;
    matrix_A_k_.emplace_back(A_c);
    matrix_B_k_.emplace_back(B_c);
  }
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
  // TODO: construct constraints
const size_t kNumParam = num_of_state_ * (horizon_ + 1) +
num_of_control_ * horizon_ + num_of_slack_var_ * (horizon_ + 1);
const size_t kNumConstraints = kNumParam + num_of_state_ * (horizon_ + 1) + horizon_ + num_of_slack_var_ * (horizon_ + 1);
lower_bounds->resize(kNumConstraints);
upper_bounds->resize(kNumConstraints);

// TODO move to .h
Eigen::MatrixXd INF(2, 1);
INF << inf, inf;
std::vector<Eigen::MatrixXd> matrix_x_l;
std::vector<Eigen::MatrixXd> matrix_x_u;
std::vector<double> diag_matrix_s_slack_l_;
std::vector<double> diag_matrix_s_slack_u_;
std::vector<double> diag_matrix_ds_slack_l_;
std::vector<double> diag_matrix_ds_slack_u_;
std::pair<double, double> ddx_bounds;
std::pair<double, double> dddx_bounds;

std::vector<std::vector<std::pair<OSQPInt, OSQPFloat>>> variables(kNumParam);

int constraint_index = 0;
int slack_start_index = num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_;
int slack_l_start_index = slack_start_index + 2 * (horizon_ + 1);
for (size_t i = 0; i < kNumConstraints; ++i) {
    // x [s, v]
    if (i < num_of_state_ * (horizon_ + 1)) {
        int index = i % num_of_state_;
        // x [s, v] + slack_x_l [s_slack_l, v_slack_l] + slack_x_u[s_slack_u, v_slack_u]
        variables[i].emplace_back(constraint_index, 1.0);
        variables[i + 1].emplace_back(constraint_index + 1, 1.0);
        lower_bounds->at(constraint_index) = matrix_x_l[index];
        upper_bounds->at(constraint_index) = matrix_x_u[index];
        constraint_index += 2;
        
        // slack_u
        if (diag_matrxi_w_s_slack_u_[i] != 0 || diag_matrxi_w_ds_slack_u_[i] != 0) {
            variables[slack_start_index + i].emplace_back(constraint_index, -1.0);
            variables[slack_start_index + i + 1].emplace_back(constraint_index + 1, -1.0)
            lower_bounds->at(constraint_index) = -INF;
            upper_bounds->at(constraint_index) = matrix_x_u[index];
        }
        constraint_index += 2;

        // slack_l
        if (diag_matrxi_w_s_slack_l_[i] != 0 || diag_matrxi_w_ds_slack_l_[i] != 0) {
            variables[slack_l_start_index + i].emplace_back(constraint_index, 1.0);
            variables[slack_l_start_index + i + 1].emplace_back(constraint_index + 1, 1.0);
            lower_bounds->at(constraint_index) = matrix_x_l[index];
            upper_bounds->at(constraint_index) = INF;
        }
        ++constraint_index;
    } else if (i < num_of_state_ * (horizon_ + 1) + num_of_control_ * horizon_) {
        // a 
        variables[num_of_state_ * (horizon_ + 1) + i].emplace_back(constraint_index, 1.0);
        lower_bounds->at(constraint_index) = ddx_bounds[i - ].first;
        upper_bounds->at(constraint_index) = ddx_bounds[i].second;
    } else if (i < slack_start_index + (horizon_ + 1)){
        // s_slack_u
        variables[slack_start_index + i].emplace_back(constraint_index, 1.0);
        lower_bounds->at(constraint_index) = 0.0;
        upper_bounds->at(constraint_index) = diag_matrix_s_slack_u_[i - slack_start_index];
    } else if (i < slack_start_index + 2.0 * (horizon_ + 1)) {
        // ds_slack_u
        variables[slack_start_index + (horizon_ + 1) + i].emplace_back(constraint_index, 1.0);
        lower_bounds->at(constraint_index) = 0.0;
        upper_bounds->at(constraint_index) = diag_matrix_ds_slack_u_[i - slack_start_index - (horizon_ + 1)];
    } else if (i < slack_start_index + 3 * (horizon_ + 1)) {
        // s_slack_l
        variables[slack_l_start_index + i].emplace_back(constraint_index, 1.0);
        lower_bounds->at(constraint_index) = 0.0;
        upper_bounds->at(constraint_index) = diag_matrix_s_slack_l_[i - slack_l_start_index].second;
    } else {
        // ds_slack_l
        variables[slack_l_start_index + (horizon_ + 1) + i].emplace_back(constraint_index, 1.0);
        lower_bounds->at(constraint_index) = 0.0;
        upper_bounds->at(constraint_index) = diag_matrix_ds_slack_l_[i - slack_l_start_index - (horizon_ + 1)].second;
    }
    ++constraint_index;
}

// x_k+1 = Ac * x_k + Bc * u_k;
Eigen::MatrixXd I(2,2);
I << 1, 0, 0, 1;
Eigen::MatrixXd X_0(2, 1), Zero(2, 1);
X_0 << s_init_, ds_init_;
Zero << 0, 0;
for (int i = 0; i < horizon_; ++i) {
    variables[i].emplace_back(constraint_index, -I);
    variables[i].emplace_back(constraint_index + 1, matrix_A_k[i]);
    variables[horizon_ + 1 + i].emplace_back(constraint_index, 0);
    variables[horizon_ + 1 + i].emplace_back(constraint_index + 1, matrix_B_k[i]);

    if (i == 0) {
        lower_bounds->at(constraint_index) = -X_0;
        upper_bounds->at(constraint_index) = -X_0;
    } else {
        lower_bounds->at(constraint_index) = Zero;
        upper_bounds->at(constraint_index) = Zero;
    }
    constraint_index += 2;
}

variables[horizon_].emplace_back(constraint_index, -I);
variables[num_of_state_ * (horizon_ + 1) + horizon_].emplace_back(constraint_index, Zero);
lower_bounds->at(constraint_index) = Zero;
upper_bounds->at(constraint_index) = Zero;
constraint_index += 2;

// u_dot_k = (u_k - u_k-1) / delta_t
for (int i = 0; i < horizon_; ++i) {
    if (i == horizon_ - 1) {
        variables[num_of_state_ * (horizon_ + 1) + i].emplace_back(constraint_index, 1.0);
    } else {
        variables[num_of_state_ * (horizon_ + 1) + i].emplace_back(constraint_index, 1.0);
        variables[num_of_state_ * (horizon_ + 1) + i + 1].emplace_back(constraint_index, -1.0);
    }

    if (i == 0) {
        lower_bounds->at(constraint_index) = prev_dds_ + dddx_bounds[i].first * delta_t;
        upper_bounds->at(constraint_index) = prev_dds_ + dddx_bounds[i].second * delta_t;
    } else {
        lower_bounds->at(constraint_index) = dddx_bounds[i].first * delta_t;
        upper_bounds->at(constraint_index) = dddx_bounds[i].second * delta_t;
    }
    ++constraint_index;
}

int ind_p = 0;
  for (int i = 0; i< kNumParam; ++i) {
    A_indptr->push_back(ind_p);
    for (const auto& variable_nz : variables[i]) {
        // coefficient
        A_data->push_back(variable_nz.second);
        // constraint index
        A_indices->push_back(variable_nz.first);
        ++ind_p;
    }
}
A_indptr->push_back(ind_p);
}

}  // namespace mpc