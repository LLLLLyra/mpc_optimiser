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

  template <typename T>
  void CSCInitHelper(
      const std::vector<std::vector<std::pair<OSQPInt, T>>> &columns,
      std::vector<T> *P_data, std::vector<OSQPInt> *P_indices,
      std::vector<OSQPInt> *P_indptr) {
    int ind_p = 0;
    for (size_t i = 0; i < columns.size(); ++i) {
      P_indptr->push_back(ind_p);
      for (const auto &row_data_pair : columns[i]) {
        P_data->push_back(row_data_pair.second);
        P_indices->push_back(row_data_pair.first);
        ++ind_p;
      }
    }
    P_indptr->push_back(ind_p);
  }

 protected:
  MPCConfig config_;
  size_t horizon_;
  double delta_t_;

  std::vector<double> solution_;
};
}  // namespace mpc