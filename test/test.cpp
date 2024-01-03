#include <random>

#include "mpc/lon_mpc_solver.h"

double RUnif(double mn, double mx) {
  std::random_device seed;
  std::ranlux48 engine(seed());
  std::uniform_int_distribution<> distrib(mn, mx);
  int random = distrib(engine);
}

int main() {
  const double jerk = 0.01;
  const double total_t = 8.0;
  const size_t horizon = 36;
  const double delta_t = total_t / horizon;

  mpc::MPCConfig config;
  config.set_delta_t(delta_t);
  config.set_num_of_knots(horizon);
  config.set_max_iter(4000);
  config.set_eps(1e-2);

  auto* lon_config = config.mutable_lon_mpc_config();
  lon_config->set_dare_max_itr(1000);
  lon_config->set_dare_tol(1e-4);
  lon_config->set_prev_dds(0.0);
  lon_config->set_s_init(0.0);
  lon_config->set_ds_init(0.0);
  lon_config->add_matrix_q(100.0);
  lon_config->add_matrix_r(50.0);
  lon_config->add_matrix_r_dot(10.0);

  std::vector<double> x_ref(horizon + 1, 0.0);
  std::vector<double> dx_ref(horizon + 1, 0.0);
  for (int i = 1; i <= horizon; i++) {
    double a = jerk * delta_t;
    a = std::min(2.0, a);
    double& v = dx_ref[i];
    v += a * delta_t + RUnif(0.0, 0.01);
    v = std::min(v, 8.0);
    double& s = x_ref[i];
    s += v * delta_t + RUnif(0.0, 0.1);
  }

  std::vector<std::pair<double, double>> x_bounds(horizon + 1),
      dx_bounds(horizon + 1), ddx_bounds(horizon), dddx_bounds(horizon);
  std::fill_n(x_bounds.begin(), horizon + 1,
              std::make_pair<double, double>(0.0, 1.0));
  std::fill_n(dx_bounds.begin(), horizon + 1,
              std::make_pair<double, double>(0.0, 8.0));
  std::fill_n(ddx_bounds.begin(), horizon,
              std::make_pair<double, double>(0.0, 2.0));
  std::fill_n(dddx_bounds.begin(), horizon,
              std::make_pair<double, double>(0.0, 2.0));

  std::vector<double> s_slack_w(horizon + 1, 100),
      s_slack_u(horizon + 1, OSQP_INFTY);

  mpc::LongitudinalMPCSolver solver(config);
  solver.set_s_ref(x_ref);
  solver.set_ds_ref(dx_ref);
  solver.set_x_bounds(x_bounds);
  solver.set_dx_bounds(dx_bounds);
  solver.set_ddx_bounds(ddx_bounds);
  solver.set_dddx_bounds(dddx_bounds);

  solver.set_w_s_slack_u(s_slack_w);
  solver.set_w_s_slack_l(s_slack_w);
  solver.set_s_slack_u(s_slack_u);

  bool suc = solver.Solve(4000);

  return 0;
}