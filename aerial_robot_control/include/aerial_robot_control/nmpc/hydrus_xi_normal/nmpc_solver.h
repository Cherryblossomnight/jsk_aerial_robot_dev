//

//

#ifndef HYDRUS_XI_NMPC_SOLVER_H
#define HYDRUS_XI_NMPC_SOLVER_H

#include "aerial_robot_control/nmpc/base_mpc_solver.h"
#include "aerial_robot_control/nmpc/hydrus_xi_normal/c_generated_code/acados_solver_hydrus_xi_normal.h"

namespace aerial_robot_control
{

namespace mpc_solver
{

class HydrusXiMPCSolver : public BaseMPCSolver
{
public:
  HydrusXiMPCSolver()
  {
    // acados macro
    NN_ = HYDRUS_XI_NORMAL_N;
    NX_ = HYDRUS_XI_NORMAL_NX;
    NZ_ = HYDRUS_XI_NORMAL_NZ;
    NU_ = HYDRUS_XI_NORMAL_NU;
    NP_ = HYDRUS_XI_NORMAL_NP;
    NBX_ = HYDRUS_XI_NORMAL_NBX;
    NBX0_ = HYDRUS_XI_NORMAL_NBX0;
    NBU_ = HYDRUS_XI_NORMAL_NBU;
    NSBX_ = HYDRUS_XI_NORMAL_NSBX;
    NSBU_ = HYDRUS_XI_NORMAL_NSBU;
    NSH_ = HYDRUS_XI_NORMAL_NSH;
    NSH0_ = HYDRUS_XI_NORMAL_NSH0;
    NSG_ = HYDRUS_XI_NORMAL_NSG;
    NSPHI_ = HYDRUS_XI_NORMAL_NSPHI;
    NSHN_ = HYDRUS_XI_NORMAL_NSHN;
    NSGN_ = HYDRUS_XI_NORMAL_NSGN;
    NSPHIN_ = HYDRUS_XI_NORMAL_NSPHIN;
    NSPHI0_ = HYDRUS_XI_NORMAL_NSPHI0;
    NSBXN_ = HYDRUS_XI_NORMAL_NSBXN;
    NS_ = HYDRUS_XI_NORMAL_NS;
    NS0_ = HYDRUS_XI_NORMAL_NS0;
    NSN_ = HYDRUS_XI_NORMAL_NSN;
    NG_ = HYDRUS_XI_NORMAL_NG;
    NBXN_ = HYDRUS_XI_NORMAL_NBXN;
    NGN_ = HYDRUS_XI_NORMAL_NGN;
    NY0_ = HYDRUS_XI_NORMAL_NY0;
    NY_ = HYDRUS_XI_NORMAL_NY;
    NYN_ = HYDRUS_XI_NORMAL_NYN;
    NH_ = HYDRUS_XI_NORMAL_NH;
    NHN_ = HYDRUS_XI_NORMAL_NHN;
    NH0_ = HYDRUS_XI_NORMAL_NH0;
    NPHI0_ = HYDRUS_XI_NORMAL_NPHI0;
    NPHI_ = HYDRUS_XI_NORMAL_NPHI;
    NPHIN_ = HYDRUS_XI_NORMAL_NPHIN;
    NR_ = HYDRUS_XI_NORMAL_NR;

    // acados functions that only using once
    acados_ocp_capsule_ = hydrus_xi_normal_acados_create_capsule();

    int status = hydrus_xi_normal_acados_create(acados_ocp_capsule_);
    if (status)
      throw std::runtime_error("hydrus_xi_normal_acados_create() returned status " + std::to_string(status) +
                               ". Exiting.");

    nlp_config_ = hydrus_xi_normal_acados_get_nlp_config(acados_ocp_capsule_);
    nlp_dims_ = hydrus_xi_normal_acados_get_nlp_dims(acados_ocp_capsule_);
    nlp_in_ = hydrus_xi_normal_acados_get_nlp_in(acados_ocp_capsule_);
    nlp_out_ = hydrus_xi_normal_acados_get_nlp_out(acados_ocp_capsule_);
    nlp_solver_ = hydrus_xi_normal_acados_get_nlp_solver(acados_ocp_capsule_);
    nlp_opts_ = hydrus_xi_normal_acados_get_nlp_opts(acados_ocp_capsule_);
  };

  ~HydrusXiMPCSolver() override
  {
    int status = hydrus_xi_normal_acados_free(acados_ocp_capsule_);
    if (status)
      std::cout << "hydrus_xi_normal_acados_free() returned status " << status << ". \n" << std::endl;

    status = hydrus_xi_normal_acados_free_capsule(acados_ocp_capsule_);
    if (status)
      std::cout << "hydrus_xi_normal_acados_free_capsule() returned status " << status << ". \n" << std::endl;
  };

protected:
  hydrus_xi_normal_solver_capsule* acados_ocp_capsule_ = nullptr;

  // acados functions that using multiple times
  inline int acadosUpdateParams(int stage, std::vector<double>& value) override
  {
    return hydrus_xi_normal_acados_update_params(acados_ocp_capsule_, stage, value.data(), NP_);
  }

  inline int acadosUpdateParamsSparse(int stage, std::vector<int>& idx, std::vector<double>& p, int n_update) override
  {
    return hydrus_xi_normal_acados_update_params_sparse(acados_ocp_capsule_, stage, idx.data(), p.data(), n_update);
  }

  inline int acadosSolve() override
  {
    return hydrus_xi_normal_acados_solve(acados_ocp_capsule_);
  }

  inline void acadosPrintStats() override
  {
    hydrus_xi_normal_acados_print_stats(acados_ocp_capsule_);
  }
};

}  // namespace mpc_solver

}  // namespace aerial_robot_control

#endif  // HYDRUS_XI_NMPC_SOLVER_H
