#include "mpc_solver.h"

#include "glog/logging.h"

namespace mpc {
MPCSolver::MPCSolver(const MPCConfig& config)
    : config_(config), horizon_(config_.num_of_knots()) {}

bool MPCSolver::Solve(int max_iter) {
  OSQPData* data = FormulateProblem();

  OSQPSettings* settings = Settings();
  settings->max_iter = max_iter;

  OSQPSolver* osqp_solver = nullptr;
  osqp_setup(&osqp_solver, data->P, data->q, data->A, data->l, data->u, data->m,
             data->n, settings);

  osqp_solve(osqp_solver);

  auto status = osqp_solver->info->status_val;

  if (status < 0 || (status != 1 && status != 2)) {
    LOG(ERROR) << "failed optimization status:\t" << osqp_solver->info->status;
    osqp_cleanup(osqp_solver);
    FreeData(data);
    free(settings);
    return false;
  } else if (osqp_solver->solution == nullptr) {
    LOG(ERROR) << "The solution from OSQP is nullptr";
    osqp_cleanup(osqp_solver);
    FreeData(data);
    free(settings);
    return false;
  }

  // extract solution
  ExtractSolution(osqp_solver->solution, data->n);

  // Cleanup
  osqp_cleanup(osqp_solver);
  FreeData(data);
  free(settings);
  return true;
}

OSQPData* MPCSolver::FormulateProblem() {
  // calculate kernel
  std::vector<OSQPFloat> P_data;
  std::vector<OSQPInt> P_indices;
  std::vector<OSQPInt> P_indptr;
  CalculateKernel(&P_data, &P_indices, &P_indptr);

  // calculate affine constraints
  std::vector<OSQPFloat> A_data;
  std::vector<OSQPInt> A_indices;
  std::vector<OSQPInt> A_indptr;
  std::vector<OSQPFloat> lower_bounds;
  std::vector<OSQPFloat> upper_bounds;
  CalculateAffineConstraint(&A_data, &A_indices, &A_indptr, &lower_bounds,
                            &upper_bounds);

  // calculate offset
  std::vector<OSQPFloat> q;
  CalculateOffset(&q);

  OSQPData* data = reinterpret_cast<OSQPData*>(malloc(sizeof(OSQPData)));
  CHECK_EQ(lower_bounds.size(), upper_bounds.size());

  CHECK_GT(P_indptr.back(), 0);
  size_t num_of_var = P_indptr.back() - 1;
  CHECK(num_of_var % horizon_ == 0);

  size_t kernel_dim = num_of_var;
  size_t num_affine_constraint = lower_bounds.size();

  data->n = kernel_dim;
  data->m = num_affine_constraint;
  csc_set_data(data->P, kernel_dim, kernel_dim, P_data.size(), CopyData(P_data),
               CopyData(P_indices), CopyData(P_indptr));
  data->q = CopyData(q);
  csc_set_data(data->A, num_affine_constraint, kernel_dim, A_data.size(),
               CopyData(A_data), CopyData(A_indices), CopyData(A_indptr));
  data->l = CopyData(lower_bounds);
  data->u = CopyData(upper_bounds);
  return data;
}

void MPCSolver::FreeData(OSQPData* data) {
  delete[] data->q;
  delete[] data->l;
  delete[] data->u;

  delete[] data->P->i;
  delete[] data->P->p;
  delete[] data->P->x;

  delete[] data->A->i;
  delete[] data->A->p;
  delete[] data->A->x;
}

OSQPSettings* MPCSolver::Settings() {
  OSQPSettings* settings =
      reinterpret_cast<OSQPSettings*>(malloc(sizeof(OSQPSettings)));
  if (settings == nullptr) {
    return nullptr;
  } else {
    osqp_set_default_settings(settings);
    settings->scaled_termination = true;
    settings->verbose = false;
    settings->max_iter = config_.max_iter();
    settings->eps_abs = config_.eps();
    return settings;
  }
}

void MPCSolver::ExtractSolution(OSQPSolution* osqp_solution,
                                OSQPInt num_of_var) {
  std::vector<double>().swap(solution_);
  solution_.reserve(num_of_var);
  for (size_t i = 0; i < num_of_var; ++i) {
    solution_.emplace_back(osqp_solution->x[i]);
  }
}

}  // namespace mpc