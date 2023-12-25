#pragma once

#include "mpc_solver.h"

namespace mpc {
class LongitudinalMPCSolver : public MPCSolver {
 public:
  LongitudinalMPCSolver(const MPCConfig &config);

 protected:
  void CalculateKernel(std::vector<OSQPFloat> *P_data,
                       std::vector<OSQPInt> *P_indices,
                       std::vector<OSQPInt> *P_indptr) override;

  void CalculateOffset(std::vector<OSQPFloat> *q) override;

  void CalculateAffineConstraint(std::vector<OSQPFloat> *A_data,
                                 std::vector<OSQPInt> *A_indices,
                                 std::vector<OSQPInt> *A_indptr,
                                 std::vector<OSQPFloat> *lower_bounds,
                                 std::vector<OSQPFloat> *upper_bounds) override;
};
}  // namespace mpc