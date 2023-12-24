#pragma once

#include <algorithm>
#include <cfloat>
#include <utility>
#include <vector>

#include "Eigen/Eigen"
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

  bool Solve();

 protected:
  void CalculateKernel(std::vector<OSQPFloat> *P_data,
                       std::vector<OSQPInt> *P_indices,
                       std::vector<OSQPInt> *P_indptr);

  void CalculateOffset(std::vector<OSQPFloat> *q);

  void CalculateAffineConstraint(std::vector<OSQPFloat> *A_data,
                                 std::vector<OSQPInt> *A_indices,
                                 std::vector<OSQPInt> *A_indptr,
                                 std::vector<OSQPFloat> *lower_bounds,
                                 std::vector<OSQPFloat> *upper_bounds);

  OSQPSettings *Settings();
  OSQPData *FormulateProblem();
  void FreeData(OSQPData *data);

  template <typename T>
  T *CopyData(const std::vector<T> &vec) {
    T *data = new T[vec.size()];
    memcpy(data, vec.data(), sizeof(T) * vec.size());
    return data;
  }

 protected:
  MPCConfig config_;
  int horizon_;
};
}  // namespace mpc