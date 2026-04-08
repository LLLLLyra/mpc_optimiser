#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "mpc/lateral_mpc_solver.h"
#include "mpc/lon_mpc_solver.h"

namespace {

constexpr double kEps = 1e-6;

using DenseMatrix = std::vector<std::vector<double>>;
using Bounds = std::vector<std::pair<double, double>>;

class LateralMPCSolverTestHarness : public mpc::LateralMPCSolver {
 public:
  using mpc::LateralMPCSolver::LateralMPCSolver;

  void BuildOffset(std::vector<OSQPFloat>* q) { CalculateOffset(q); }

  void BuildAffine(std::vector<OSQPFloat>* A_data,
                   std::vector<OSQPInt>* A_indices,
                   std::vector<OSQPInt>* A_indptr,
                   std::vector<OSQPFloat>* lower_bounds,
                   std::vector<OSQPFloat>* upper_bounds) {
    CalculateAffineConstraint(A_data, A_indices, A_indptr, lower_bounds,
                              upper_bounds);
  }

  void BuildStateMatrices(int k, Eigen::MatrixXd* matrix_A_k,
                          Eigen::MatrixXd* matrix_B_k,
                          Eigen::MatrixXd* matrix_B_tilde_k) {
    InitStateMatrices(k, matrix_A_k, matrix_B_k, matrix_B_tilde_k);
  }

  void ExtractFromVector(const std::vector<double>& solution) {
    OSQPSolution osqp_solution{};
    osqp_solution.x = const_cast<OSQPFloat*>(solution.data());
    ExtractSolution(&osqp_solution, solution.size());
  }

  const Eigen::MatrixXd& terminal_q() const { return matrix_q_n_; }
  size_t num_state() const { return num_of_state_; }
  size_t num_var() const { return num_of_var_; }
  size_t horizon_size() const { return horizon_; }
};

class LongitudinalMPCSolverTestHarness : public mpc::LongitudinalMPCSolver {
 public:
  using mpc::LongitudinalMPCSolver::LongitudinalMPCSolver;

  void BuildOffset(std::vector<OSQPFloat>* q) { CalculateOffset(q); }

  void BuildKernel(std::vector<OSQPFloat>* P_data,
                   std::vector<OSQPInt>* P_indices,
                   std::vector<OSQPInt>* P_indptr) {
    CalculateKernel(P_data, P_indices, P_indptr);
  }

  void ExtractFromVector(const std::vector<double>& solution) {
    OSQPSolution osqp_solution{};
    osqp_solution.x = const_cast<OSQPFloat*>(solution.data());
    ExtractSolution(&osqp_solution, solution.size());
  }

  const Eigen::MatrixXd& terminal_q() const { return matrix_q_n_; }
  size_t num_state() const { return num_of_state_; }
  size_t num_var() const { return num_of_var_; }
};

mpc::MPCConfig MakeLateralConfig(size_t horizon) {
  mpc::MPCConfig config;
  config.set_delta_t(0.1);
  config.set_num_of_knots(horizon);
  config.set_max_iter(4000);
  config.set_eps(1e-4);

  auto* lat_config = config.mutable_lat_mpc_config();
  lat_config->set_dare_max_itr(200);
  lat_config->set_dare_tol(1e-6);
  lat_config->set_prev_delta(0.15);
  lat_config->set_l_init(0.0);
  lat_config->set_l_dot_init(0.0);
  lat_config->set_psi_init(0.0);
  lat_config->set_psi_dot_init(0.0);
  lat_config->set_c_af(16000.0);
  lat_config->set_c_ar(17000.0);
  lat_config->set_l_f(1.2);
  lat_config->set_l_r(1.6);
  lat_config->set_i_z(2250.0);
  lat_config->set_m(1500.0);

  lat_config->add_matrix_q(10.0);
  lat_config->add_matrix_q(20.0);
  lat_config->add_matrix_q(30.0);
  lat_config->add_matrix_q(40.0);
  lat_config->add_matrix_r(5.0);
  lat_config->add_matrix_r_dot(2.0);

  for (size_t i = 0; i < horizon + 1; ++i) {
    lat_config->add_velocity(8.0 + static_cast<double>(i));
    lat_config->add_kappa(0.01 * static_cast<double>(i + 1));
  }
  return config;
}

mpc::MPCConfig MakeLongitudinalConfig(size_t horizon) {
  mpc::MPCConfig config;
  config.set_delta_t(0.2);
  config.set_num_of_knots(horizon);
  config.set_max_iter(4000);
  config.set_eps(1e-4);

  auto* lon_config = config.mutable_lon_mpc_config();
  lon_config->set_dare_max_itr(200);
  lon_config->set_dare_tol(1e-6);
  lon_config->set_prev_dds(0.3);
  lon_config->set_s_init(0.0);
  lon_config->set_ds_init(0.0);
  lon_config->add_matrix_q(10.0);
  lon_config->add_matrix_q(20.0);
  lon_config->add_matrix_r(5.0);
  lon_config->add_matrix_r_dot(2.0);
  return config;
}

DenseMatrix DenseFromCSC(size_t rows, size_t cols,
                         const std::vector<OSQPFloat>& data,
                         const std::vector<OSQPInt>& indices,
                         const std::vector<OSQPInt>& indptr) {
  if (indptr.size() != cols + 1) {
    throw std::runtime_error("CSC indptr size mismatch");
  }
  if (data.size() != indices.size()) {
    throw std::runtime_error("CSC data/indices size mismatch");
  }
  if (!indptr.empty() &&
      static_cast<size_t>(indptr.back()) != data.size()) {
    throw std::runtime_error("CSC final indptr does not match nnz");
  }
  DenseMatrix dense(rows, std::vector<double>(cols, 0.0));
  for (size_t col = 0; col < cols; ++col) {
    if (indptr[col] > indptr[col + 1]) {
      throw std::runtime_error("CSC indptr not monotonic");
    }
    for (OSQPInt idx = indptr[col]; idx < indptr[col + 1]; ++idx) {
      if (idx < 0 || static_cast<size_t>(idx) >= data.size()) {
        throw std::runtime_error("CSC index pointer out of range");
      }
      if (indices[idx] < 0 || static_cast<size_t>(indices[idx]) >= rows) {
        throw std::runtime_error("CSC row index out of range");
      }
      dense[indices[idx]][col] += data[idx];
    }
  }
  return dense;
}

void ExpectTrue(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void ExpectNear(double actual, double expected, const std::string& message,
                double tolerance = kEps) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                             ", expected=" + std::to_string(expected));
  }
}

void TestLateralOffsetUsesWeightedNegativeReference() {
  auto config = MakeLateralConfig(2);
  LateralMPCSolverTestHarness solver(config);

  solver.set_l_ref({1.0, 2.0, 3.0});
  solver.set_l_dot_ref({4.0, 5.0, 6.0});
  solver.set_psi_ref({7.0, 8.0, 9.0});
  solver.set_psi_dot_ref({-1.0, -2.0, -3.0});

  std::vector<OSQPFloat> q;
  solver.BuildOffset(&q);

  ExpectNear(q[0], -10.0, "l_0 offset");
  ExpectNear(q[1], -80.0, "l_dot_0 offset");
  ExpectNear(q[2], -210.0, "psi_0 offset");
  ExpectNear(q[3], 40.0, "psi_dot_0 offset");
  ExpectNear(q[4], -20.0, "l_1 offset");
  ExpectNear(q[5], -100.0, "l_dot_1 offset");
  ExpectNear(q[6], -240.0, "psi_1 offset");
  ExpectNear(q[7], 80.0, "psi_dot_1 offset");

  Eigen::Vector4d terminal_ref;
  terminal_ref << 3.0, 6.0, 9.0, -3.0;
  const Eigen::VectorXd terminal_q = -solver.terminal_q() * terminal_ref;
  ExpectNear(q[8], terminal_q(0), "terminal l offset");
  ExpectNear(q[9], terminal_q(1), "terminal l_dot offset");
  ExpectNear(q[10], terminal_q(2), "terminal psi offset");
  ExpectNear(q[11], terminal_q(3), "terminal psi_dot offset");

  ExpectNear(q[12], -30.0, "first delta offset");
  for (size_t i = 13; i < q.size(); ++i) {
    ExpectNear(q[i], 0.0, "remaining offset entries should be zero");
  }
}

void TestLateralAffineConstraintUsesCorrectSlackAndControlIndices() {
  constexpr size_t kHorizon = 2;
  auto config = MakeLateralConfig(kHorizon);
  LateralMPCSolverTestHarness solver(config);

  const std::vector<double> zero_ref(kHorizon + 1, 0.0);
  solver.set_l_ref(zero_ref);
  solver.set_l_dot_ref(zero_ref);
  solver.set_psi_ref(zero_ref);
  solver.set_psi_dot_ref(zero_ref);

  const Bounds state_bounds(kHorizon + 1, {-1.0, 1.0});
  const Bounds control_bounds(kHorizon, {-0.5, 0.5});
  solver.set_l_bounds(state_bounds);
  solver.set_l_dot_bounds(state_bounds);
  solver.set_psi_bounds(state_bounds);
  solver.set_psi_dot_bounds(state_bounds);
  solver.set_delta_bounds(control_bounds);
  solver.set_delta_dot_bounds(control_bounds);

  const std::vector<double> enabled_weights(kHorizon + 1, 1.0);
  solver.set_w_l_slack_u(enabled_weights);
  solver.set_w_l_dot_slack_u(enabled_weights);
  solver.set_w_psi_slack_u(enabled_weights);
  solver.set_w_psi_dot_slack_u(enabled_weights);
  solver.set_w_l_slack_l(enabled_weights);
  solver.set_w_l_dot_slack_l(enabled_weights);
  solver.set_w_psi_slack_l(enabled_weights);
  solver.set_w_psi_dot_slack_l(enabled_weights);

  solver.set_l_slack_u({0.5, 1.5, 2.5});
  solver.set_l_dot_slack_u({3.5, 4.5, 5.5});
  solver.set_psi_slack_u({6.5, 7.5, 8.5});
  solver.set_psi_dot_slack_u({9.5, 10.5, 11.5});
  solver.set_l_slack_l({12.5, 13.5, 14.5});
  solver.set_l_dot_slack_l({15.5, 16.5, 17.5});
  solver.set_psi_slack_l({18.5, 19.5, 20.5});
  solver.set_psi_dot_slack_l({21.5, 22.5, 23.5});

  std::vector<OSQPFloat> A_data;
  std::vector<OSQPInt> A_indices;
  std::vector<OSQPInt> A_indptr;
  std::vector<OSQPFloat> lower_bounds;
  std::vector<OSQPFloat> upper_bounds;
  solver.BuildAffine(&A_data, &A_indices, &A_indptr, &lower_bounds,
                     &upper_bounds);

  const DenseMatrix dense =
      DenseFromCSC(lower_bounds.size(), solver.num_var(), A_data, A_indices,
                   A_indptr);

  const size_t state_offset = solver.num_state() * (kHorizon + 1);
  const size_t control_offset = state_offset;
  const size_t slack_start = state_offset + kHorizon;
  const size_t lower_slack_start = slack_start + 4 * (kHorizon + 1);
  const size_t state_constraint_count = solver.num_state() * (kHorizon + 1) * 2;
  const size_t control_constraint_start = state_constraint_count;
  const size_t slack_constraint_start = control_constraint_start + kHorizon;
  const size_t ineq_constraints =
      state_constraint_count + kHorizon + kHorizon + 8 * (kHorizon + 1);
  const size_t eq_constraints_start = ineq_constraints;

  ExpectNear(dense[8][slack_start + 1], -1.0,
             "timestep-1 l upper should use l_slack_u[1]");
  ExpectNear(dense[12][slack_start + 2 * (kHorizon + 1) + 1], -1.0,
             "timestep-1 psi upper should use psi_slack_u[1]");
  ExpectNear(dense[1][lower_slack_start], 1.0,
             "timestep-0 l lower should use l_slack_l[0]");
  ExpectNear(dense[13][lower_slack_start + 2 * (kHorizon + 1) + 1], 1.0,
             "timestep-1 psi lower should use psi_slack_l[1]");

  ExpectNear(upper_bounds[slack_constraint_start + 0], 0.5,
             "l_slack_u[0] bound should stay aligned");
  ExpectNear(upper_bounds[slack_constraint_start + 2 * (kHorizon + 1) + 1],
             7.5, "psi_slack_u[1] bound should stay aligned");
  ExpectNear(upper_bounds[slack_constraint_start + 6 * (kHorizon + 1) + 1],
             19.5, "psi_slack_l[1] bound should stay aligned");

  Eigen::MatrixXd matrix_A_k;
  Eigen::MatrixXd matrix_B_k;
  Eigen::MatrixXd matrix_B_tilde_k;
  solver.BuildStateMatrices(0, &matrix_A_k, &matrix_B_k, &matrix_B_tilde_k);
  for (size_t row = 0; row < solver.num_state(); ++row) {
    ExpectNear(dense[eq_constraints_start + 4 + row][control_offset + 0],
               matrix_B_k(row, 0), "x1 should depend on u0");
    ExpectNear(dense[eq_constraints_start + 4 + row][control_offset + 1], 0.0,
               "x1 should not depend on u1");
    ExpectNear(lower_bounds[eq_constraints_start + 4 + row],
               -matrix_B_tilde_k(row, 0) * 0.01 * 8.0,
               "x1 disturbance should use step-0 curvature and speed");
  }

  solver.BuildStateMatrices(1, &matrix_A_k, &matrix_B_k, &matrix_B_tilde_k);
  for (size_t row = 0; row < solver.num_state(); ++row) {
    ExpectNear(dense[eq_constraints_start + 8 + row][control_offset + 1],
               matrix_B_k(row, 0), "x2 should depend on u1");
    ExpectNear(lower_bounds[eq_constraints_start + 8 + row],
               -matrix_B_tilde_k(row, 0) * 0.02 * 9.0,
               "x2 disturbance should use step-1 curvature and speed");
  }
}

void TestLateralExtractSolutionMatchesInterleavedStateLayout() {
  auto config = MakeLateralConfig(2);
  LateralMPCSolverTestHarness solver(config);

  std::vector<double> solution(solver.num_var(), 0.0);
  for (size_t i = 0; i < solution.size(); ++i) {
    solution[i] = static_cast<double>(i);
  }

  solver.ExtractFromVector(solution);

  ExpectTrue((solver.opt_l() == std::vector<double>{0.0, 4.0, 8.0}),
             "opt_l should follow interleaved state layout");
  ExpectTrue((solver.opt_l_dot() == std::vector<double>{1.0, 5.0, 9.0}),
             "opt_l_dot should follow interleaved state layout");
  ExpectTrue((solver.opt_psi() == std::vector<double>{2.0, 6.0, 10.0}),
             "opt_psi should follow interleaved state layout");
  ExpectTrue((solver.opt_psi_dot() == std::vector<double>{3.0, 7.0, 11.0}),
             "opt_psi_dot should follow interleaved state layout");
  ExpectTrue((solver.opt_delta() == std::vector<double>{12.0, 13.0}),
             "opt_delta should follow control layout");
}

void TestLateralSolveZeroReferenceProblem() {
  constexpr size_t kHorizon = 4;
  auto config = MakeLateralConfig(kHorizon);
  auto* lat_config = config.mutable_lat_mpc_config();
  lat_config->set_prev_delta(0.0);
  for (int i = 0; i < lat_config->kappa_size(); ++i) {
    lat_config->set_kappa(i, 0.0);
  }
  mpc::LateralMPCSolver solver(config);

  const std::vector<double> zero_ref(kHorizon + 1, 0.0);
  const Bounds state_bounds(kHorizon + 1, {-1.0, 1.0});
  const Bounds control_bounds(kHorizon, {-0.5, 0.5});

  solver.set_l_ref(zero_ref);
  solver.set_l_dot_ref(zero_ref);
  solver.set_psi_ref(zero_ref);
  solver.set_psi_dot_ref(zero_ref);
  solver.set_l_bounds(state_bounds);
  solver.set_l_dot_bounds(state_bounds);
  solver.set_psi_bounds(state_bounds);
  solver.set_psi_dot_bounds(state_bounds);
  solver.set_delta_bounds(control_bounds);
  solver.set_delta_dot_bounds(control_bounds);

  const bool solved = solver.Solve(4000);
  ExpectTrue(solved, "zero-reference lateral MPC should solve");

  for (double value : solver.opt_l()) {
    ExpectNear(value, 0.0, "opt_l should stay at zero", 1e-4);
  }
  for (double value : solver.opt_l_dot()) {
    ExpectNear(value, 0.0, "opt_l_dot should stay at zero", 1e-4);
  }
  for (double value : solver.opt_psi()) {
    ExpectNear(value, 0.0, "opt_psi should stay at zero", 1e-4);
  }
  for (double value : solver.opt_psi_dot()) {
    ExpectNear(value, 0.0, "opt_psi_dot should stay at zero", 1e-4);
  }
  for (double value : solver.opt_delta()) {
    ExpectNear(value, 0.0, "opt_delta should stay at zero", 1e-4);
  }
}

void TestLongitudinalOffsetUsesWeightedNegativeReference() {
  auto config = MakeLongitudinalConfig(2);
  LongitudinalMPCSolverTestHarness solver(config);

  solver.set_s_ref({1.0, 2.0, 3.0});
  solver.set_ds_ref({4.0, 5.0, 6.0});

  std::vector<OSQPFloat> q;
  solver.BuildOffset(&q);

  ExpectNear(q[0], -10.0, "s_0 offset");
  ExpectNear(q[1], -80.0, "ds_0 offset");
  ExpectNear(q[2], -20.0, "s_1 offset");
  ExpectNear(q[3], -100.0, "ds_1 offset");

  Eigen::Vector2d terminal_ref;
  terminal_ref << 3.0, 6.0;
  const Eigen::VectorXd terminal_q = -solver.terminal_q() * terminal_ref;
  ExpectNear(q[4], terminal_q(0), "terminal s offset");
  ExpectNear(q[5], terminal_q(1), "terminal ds offset");
  ExpectNear(q[6], -15.0, "first acceleration offset");
  for (size_t i = 7; i < q.size(); ++i) {
    ExpectNear(q[i], 0.0, "remaining longitudinal offset entries");
  }
}

void TestLongitudinalExtractSolutionMatchesInterleavedStateLayout() {
  auto config = MakeLongitudinalConfig(2);
  LongitudinalMPCSolverTestHarness solver(config);

  std::vector<double> solution(solver.num_var(), 0.0);
  for (size_t i = 0; i < solution.size(); ++i) {
    solution[i] = static_cast<double>(i);
  }

  solver.ExtractFromVector(solution);

  ExpectTrue((solver.opt_x() == std::vector<double>{0.0, 2.0, 4.0}),
             "opt_x should follow interleaved state layout");
  ExpectTrue((solver.opt_dx() == std::vector<double>{1.0, 3.0, 5.0}),
             "opt_dx should follow interleaved state layout");
  ExpectTrue((solver.opt_ddx() == std::vector<double>{6.0, 7.0}),
             "opt_ddx should follow control layout");
}

void TestLongitudinalKernelHorizonOneControlWeight() {
  auto config = MakeLongitudinalConfig(1);
  LongitudinalMPCSolverTestHarness solver(config);

  std::vector<OSQPFloat> P_data;
  std::vector<OSQPInt> P_indices;
  std::vector<OSQPInt> P_indptr;
  solver.BuildKernel(&P_data, &P_indices, &P_indptr);

  const DenseMatrix dense =
      DenseFromCSC(solver.num_var(), solver.num_var(), P_data, P_indices,
                   P_indptr);
  const size_t control_index = solver.num_state() * (1 + 1);
  const double dt = config.delta_t();
  const double expected =
      5.0 + 2.0 / (dt * dt);
  ExpectNear(dense[control_index][control_index], expected,
             "single-step control Hessian weight");
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"offset_signs", TestLateralOffsetUsesWeightedNegativeReference},
      {"constraint_indices",
       TestLateralAffineConstraintUsesCorrectSlackAndControlIndices},
      {"solution_layout", TestLateralExtractSolutionMatchesInterleavedStateLayout},
      {"solve_smoke", TestLateralSolveZeroReferenceProblem},
      {"lon_offset_signs", TestLongitudinalOffsetUsesWeightedNegativeReference},
      {"lon_solution_layout",
       TestLongitudinalExtractSolutionMatchesInterleavedStateLayout},
      {"lon_kernel_h1", TestLongitudinalKernelHorizonOneControlWeight},
  };

  int failed = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << std::endl;
    } catch (const std::exception& e) {
      ++failed;
      std::cerr << "[FAIL] " << name << ": " << e.what() << std::endl;
    }
  }

  return failed == 0 ? 0 : 1;
}