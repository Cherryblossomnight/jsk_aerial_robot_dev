#!/usr/bin/env python
# -*- encoding: ascii -*-
import numpy as np
import casadi as ca
from hydrus_base import HydrusBase
from tilt_qd import phys_param_beetle_omni as phys_omni


class HydrusThrust(HydrusBase):
    """
    Controller Name: Tiltable Quadrotor NMPC including Servo and Thrust Model
    The controller itself is constructed in base class. This file is used to define the properties
    of the controller, specifically, the weights and cost function for the acados solver.
    The output of the controller is the thrust and servo angle command for each rotor.
    
    :param bool overwrite: Flag to overwrite existing c generated code for the OCP solver. Default: False
    """
    def __init__(self, overwrite: bool = False, phys=phys_omni):
        # Model name
        self.model_name = "hydrus_thrust"
        self.phys = phys

        self.tilt = False
        self.include_servo_model = True
        self.include_servo_derivative = False
        self.include_servo_dynamic = True
        self.include_thrust_model = False   # TODO extend to include_thrust_derivative
        self.include_cog_dist_model = True
        self.include_cog_dist_parameter = False
        self.include_impedance = False
        self.include_end_effector_dist_model = True

        # Read parameters from configuration file in the robot's package
        self.read_params("controller", "nmpc", "beetle", "BeetleNMPCFull.yaml")

        # Create acados model & solver and generate c code
        super().__init__(overwrite)

    def get_cost_function(self, lin_acc_w=None, ang_acc_b=None):
        # Cost function
        # see https://docs.acados.org/python_interface/#acados_template.acados_ocp_cost.AcadosOcpCost for details
        # NONLINEAR_LS = error^T @ Q @ error; error = y - y_ref
        # qe = qr^* multiply q
        qe_x = self.qwr * self.qx - self.qw * self.qxr - self.qyr * self.qz + self.qy * self.qzr
        qe_y = self.qwr * self.qy - self.qw * self.qyr + self.qxr * self.qz - self.qx * self.qzr
        qe_z = -self.qxr * self.qy + self.qx * self.qyr + self.qwr * self.qz - self.qw * self.qzr

        state_y = ca.vertcat(
            self.p,
            self.v,
            self.qwr,
            qe_x + self.qxr,
            qe_y + self.qyr,
            qe_z + self.qzr,
            self.w,
            self.j_s,
            self.w_s,
            self.fds_w,
            self.tau_ds_b,
            self.fde_w,
        )

        state_y_e = state_y

        control_y = ca.vertcat(self.ft_c, self.j_c)

        return state_y, state_y_e, control_y

    def get_weights(self):
        # Define Weights
        Q = np.diag(
            [
                self.params["Qp_xy"],
                self.params["Qp_xy"],
                self.params["Qp_z"],
                self.params["Qv_xy"],
                self.params["Qv_xy"],
                self.params["Qv_z"],
                0,
                self.params["Qq_xy"],
                self.params["Qq_xy"],
                self.params["Qq_z"],
                self.params["Qw_xy"],
                self.params["Qw_xy"],
                self.params["Qw_z"],
                1,
                1,
                1, # joint angles
                1,
                1,
                1, # joint velocities
                1,
                1,
                1, 
                1,
                1,
                1, # disturbance
                1,
                1,
                1, # end_effector
            ]
        )
        print("Q: \n", Q)

        R = np.diag(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )
        print("R: \n", R)

        return Q, R

    def get_reference(self, target_xyz, target_qwxyz):
        """
        Assemble reference trajectory from target pose and reference control values.
        Gets called from reference generator class.
        Note: The definition of the reference is closely linked to the definition of the cost function.
        Therefore, this is explicitly stated in each controller file to increase comprehensiveness.

        :param target_xyz: Target position
        :param target_qwxy: Target quarternions
        :param ft_ref: Target thrust
        :param a_ref: Target servo angles
        :return xr: Reference for the state x
        :return ur: Reference for the input u
        """
        # Get dimensions
        ocp = self.get_ocp(); nn = ocp.dims.N
        nx = ocp.dims.nx; nu = ocp.dims.nu

        # Assemble state reference
        xr = np.zeros([nn + 1, nx])
        xr[:, 0] = target_xyz[0]       # x
        xr[:, 1] = target_xyz[1]       # y
        xr[:, 2] = target_xyz[2]       # z
        # No reference for vx, vy, vz (idx: 3, 4, 5)
        xr[:, 6] = target_qwxyz[0]     # qx
        xr[:, 7] = target_qwxyz[1]     # qx
        xr[:, 8] = target_qwxyz[2]     # qy
        xr[:, 9] = target_qwxyz[3]     # qz
        # No reference for wx, wy, wz (idx: 10, 11, 12)

        # Assemble control reference
        # Note: Reference has to be zero if variable is included as state in cost function!
        ur = np.zeros([nn, nu])
        # ur[:, 0] = ft_ref[0]
        # ur[:, 1] = ft_ref[1]
        # ur[:, 2] = ft_ref[2]
        # ur[:, 3] = ft_ref[3]
        
        return xr, ur


if __name__ == "__main__":
    overwrite = False
    pid = HydrusThrust(overwrite)

    # print("Successfully initialized acados ocp: ", acados_ocp_solver.acados_ocp)
    # print("number of states: ", acados_ocp_solver.acados_ocp.dims.nx)
    # print("number of controls: ", acados_ocp_solver.acados_ocp.dims.nu)
    # print("number of parameters: ", acados_ocp_solver.acados_ocp.dims.np)
    print("T_samp: ", pid.params["T_samp"])
    print("T_horizon: ", pid.params["T_horizon"])
    print("T_step: ", pid.params["T_step"])
    print("N_steps: ", pid.params["N_steps"])
