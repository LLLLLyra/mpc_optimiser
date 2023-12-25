#pragma once

#include <algorithm>
#include <vector>

#include "osqp/osqp.h"
#include "proto/mpc_config.pb.h"

namespace mpc {
struct OSQPData {
  OSQPCscMatrix *P;
  OSQPFloat *q;
  OSQPCscMatrix *A;
  OSQPInt m;
  OSQPInt n;
  OSQPFloat *l;
  OSQPFloat *u;
};

class MPCSolver {
 public:
  MPCSolver(const MPCConfig &config);

  virtual bool Solve(int max_itr);

 protected:
  virtual void CalculateKernel(std::vector<OSQPFloat> *P_data,
                               std::vector<OSQPInt> *P_indices,
                               std::vector<OSQPInt> *P_indptr) = 0;

  virtual void CalculateOffset(std::vector<OSQPFloat> *q) = 0;

  virtual void CalculateAffineConstraint(
      std::vector<OSQPFloat> *A_data, std::vector<OSQPInt> *A_indices,
      std::vector<OSQPInt> *A_indptr, std::vector<OSQPFloat> *lower_bounds,
      std::vector<OSQPFloat> *upper_bounds) = 0;

  OSQPSettings *Settings();
  OSQPData *FormulateProblem();
  void FreeData(OSQPData *data);
  void ExtractSolution(OSQPSolution *osqp_solution, OSQPInt num_of_var);

  template <typename T>
  T *CopyData(const std::vector<T> &vec) {
    T *data = new T[vec.size()];
    memcpy(data, vec.data(), sizeof(T) * vec.size());
    return data;
  }

 protected:
  MPCConfig config_;
  int horizon_;

  std::vector<double> solution_;
};
}  // namespace mpc