import sys, os
from copy import deepcopy
import time
import numpy as np
import argparse
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/hydrus_xi")

from imp_viz import Visualizer

# Hydrus
from hydrus_xi.hydrus_xi_normal import HydrusXiNormal
from hydrus_xi.hydrus_xi_ext_wrench import HydrusXiExtWrench

import hydrus_xi.phys_param_hydrus_xi as phys_hydrus_xi

def main(args):
    # ========== Init ==========
    # ---------- Controller ----------
    if args.arch == 'qd':

        if args.model == 0:
            nmpc = HydrusXiNormal(phys=phys_hydrus_xi)
        elif args.model == 1:
            nmpc = HydrusXiExtWrench(phys=phys_hydrus_xi)
        else:
            raise ValueError(f"Invalid control model {args.model}.")

    else:
        raise ValueError(f"Invalid robot architecture {args.arch}.")
   
    # Get time constants
    if nmpc.include_servo_model:
        t_servo_ctrl = nmpc.phys.t_servo
    else:
        t_servo_ctrl = 0.0
    ts_ctrl = nmpc.params["T_samp"]

    # OCP solver
    ocp_solver = nmpc.get_ocp_solver()
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    n_param = ocp_solver.acados_ocp.dims.np

    x_init = np.zeros(nx)
    x_init[6] = 1.0  # qw
    x_init[13:16] = [np.pi/2, np.pi/2, np.pi/2]   # joint angles state
    x_init[16:20] = [np.pi, 0.0, np.pi, 0.0]   # gimbal angles state
    u_init = np.zeros(nu)  # joint angles command
    u_init[4:8] = [np.pi, 0.0, np.pi, 0.0]   # gimbal angles command
    for stage in range(ocp_solver.N + 1):
        ocp_solver.set(stage, "x", x_init)
    for stage in range(ocp_solver.N):
        ocp_solver.set(stage, "u", u_init)
    x_history = []
    u_history = []

    # ---------- Simulator ----------
    if args.arch == 'qd':
        if args.sim_model == 0:
            sim_nmpc = HydrusXiNormal(phys=phys_hydrus_xi)  # Consider both the servo delay and the thrust delay
        elif args.sim_model == 1:
            sim_nmpc = HydrusXiExtWrench(phys=phys_hydrus_xi) 
        else:
            raise ValueError(f"Invalid sim model {args.sim_model}.")

    else:
        raise ValueError(f"Invalid robot architecture {args.arch}.")

    # Get time constants
    if sim_nmpc.include_servo_model:
        t_servo_sim = sim_nmpc.phys.t_servo
    else:
        t_servo_sim = 0.0
    if sim_nmpc.include_thrust_model:
        t_rotor_sim = sim_nmpc.phys.t_rotor
    else:
        t_rotor_sim = 0.0

    ts_sim = 0.005  # or 0.001

    t_total_sim = 18.0

    N_sim = int(t_total_sim / ts_sim)
 

    # Sim solver
    sim_solver = sim_nmpc.create_acados_sim_solver(ts_sim, build=True)
    nx_sim = sim_solver.acados_sim.dims.nx
    nr = 15
    r_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # State Initialization
    x_init_sim = np.zeros(nx_sim)
    x_init_sim[6] = 1.0  # qw
    x_init_sim[13:16] = [np.pi/2, np.pi/2, np.pi/2]   # joint angles state
    x_init_sim[16:20] = [np.pi, 0.0, np.pi, 0.0]   # gimbal angles state
    # ---------- Reference ----------
    reference_generator = nmpc.get_reference_generator()

    # ---------- Visualization ----------
    viz = Visualizer(
        args.arch,
        N_sim,
        nx_sim,
        nr,
        nu,
        x_init_sim,
        r_init,
        tilt=nmpc.tilt,
        include_servo_model=sim_nmpc.include_servo_model,
        include_thrust_model=sim_nmpc.include_thrust_model,
        include_cog_dist_model=sim_nmpc.include_cog_dist_model,
        include_end_effector_dist_model=sim_nmpc.include_end_effector_dist_model,
        is_reference=True,
    )

    # Prepare containers to record simulation data (x and u) for future comparison
    x_history = []
    u_history = []

    is_sqp_change = False
    t_sqp_start = 2.5
    t_sqp_end = 3.0

    # ========== Run simulation ==========
    u_cmd = u_init
    t_ctl = 0.0
    x_now_sim = x_init_sim
    target_roll = 0
    target_pitch = 0
    external_wrench_I_term = np.zeros(6)
    est_external_wrench = np.zeros(6)
    est_external_wrench_plt = np.zeros(6)
    init_sum_momentum = np.zeros(6)
    take_off = False
    take_off_i = 0
    # y_target = 0.5
    # z_target = 2.0
    # t_ctrl = 4.0
    target_vxyz = np.zeros(3)
    target_wxyz = np.zeros(3)
    target_pxyz = np.array([[0.0, 0.5, 2.0]]).T
    target_prpy = np.array([[0.0, 0.0, 0.0]]).T
    nmpc.acados_init_p[30:33] = [np.pi/2, np.pi/2, np.pi/2]  # Initial joint angles
    # nmpc.acados_init_p[33:39] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial disturbance forces and torques
    # nmpc.acados_init_p[33:37] = [np.pi, 0.0, np.pi, 0.0]  # Initial gimbal angles
    # xd_ddot = 0.0
    # xd_dot = 0.0
    # xd = 1.03923048

    # xref = xd
    # yd_ddot = 0.0
    # yd_dot = 0.0
    # yd = 0.52230762
    # yref = yd
    # cmd = False
    # pe_cog_sim = np.zeros(3)
    # pe_world_sim = np.zeros(3)
    # lam = 1.0
    # last_err = 0.0
    # last_q = u_cmd[4:7]
    # sim_start = False
    # collision = False
    for i in range(N_sim):
        # print(sim_solver.acados_sim.
        # --------- Update time ---------
        t_now = i * ts_sim
        t_ctl += ts_sim
        # --------- Add state constaints ---------
       

        # --------- Update state estimation ---------
        # Assemble state from simulation and disturbance estimation 
        # if nmpc.include_cog_dist_model:
        #     x_now = np.zeros(nx)
        #     if nmpc.include_end_effector_dist_model:
        #         x_now[: nx - 9] = deepcopy(x_now_sim[: nx - 9])
        #     else:
        #         x_now[: nx - 6] = deepcopy(x_now_sim[: nx - 6])
        # else:
        x_now = deepcopy(x_now_sim[:nx])  # The dimension of x_now may be smaller than x_now_sim

        # Access from less indices
        if (nmpc.include_thrust_model and not nmpc.include_servo_model) and (
                sim_nmpc.include_servo_model and sim_nmpc.include_thrust_model):
            if args.arch == 'bi':
                x_now[13:15] = deepcopy(x_now_sim[15:17])
            elif args.arch == 'tri':
                x_now[13:16] = deepcopy(x_now_sim[16:19])
            elif args.arch == 'qd':
                x_now[13:17] = deepcopy(x_now_sim[17:21])

        # -------- Update control target --------
        target_xyz = np.array([[0.0, 0.0, 2.0]]).T
        target_rpy = np.array([[0.0, 0.0, 0.0]]).T
        target_vxyz = np.array([[0.0, 0.0, 0.0]]).T
        target_wrpy = np.array([[0.0, 0.0, 0.0]]).T
        # gravity compensation
        target_force = np.array([[0.0, 0.0, nmpc.acados_init_p[9] * nmpc.acados_init_p[10]]]).T
        target_torque = np.array([[0.0, 0.0, 0.0]]).T
        if args.plot_type == 2:
            target_xyz = np.array([[1.0, 1.5, 2.0]]).T
            target_rpy = np.array([[0.0, 0.0, 0.0]]).T

        if t_total_sim > 2.0:
           # print(nmpc.acados_init_p)
            pass
            if t_now >= 4:
                pass
                #target_xyz = np.array([[1.0, 0.5, 2.0]]).T
               # nmpc.acados_init_p[30:33] = [np.pi/3, np.pi/3, -np.pi/6]
               # x_now_sim[16:22] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                # u_cmd[4:7] = [np.pi/2, 0, 0] 
            # if 3.0 <= t_now < 5.5:
            #     assert t_sqp_end <= 3.0
            #     target_xyz = np.array([[1.0, 1.0, 1.0]]).T
            #     target_rpy = np.array([[0.0, 0.0, 0.0]]).T
            #u_cmd[4:7] = [np.pi/2, np.pi/2, np.pi/2]
            #if t_now >= 4.0:
                # nmpc.acados_init_p[30:33] = [0.0, 0.0, 0.0]
                # target_xyz = np.array([[1.0, 0.0, 0.0]]).T
                # nmpc.acados_init_p[30:33] = [np.pi/3, np.pi/3, -np.pi/6]
           # if t_now >= 6.0: 
                #target_xyz = np.array([[1.0, 0.5, 2.0]]).T
                #target_rpy = np.array([[0.0, 0.0, -np.pi/6]]).T

                # if x_now_sim[0] > 0.99:
                #     #x_force = 1000 * (0.99 - x_now_sim[0])
                #     target_force[0, 0] = -est_external_wrench[0]
                #     target_force[1, 0] = -est_external_wrench[1]
                #     target_force[2, 0] = nmpc.acados_init_p[9] * nmpc.acados_init_p[10] - est_external_wrench[2]
                #     target_torque[0, 0] = -est_external_wrench[3]
                #     target_torque[1, 0] = -est_external_wrench[4]
                #     target_torque[2, 0] = -est_external_wrench[5]
                #     nmpc.acados_init_p[33] = -1.0
                #u_cmd[4:7] = [np.pi/3, np.pi/3, -np.pi/6]
                # target_rpy = np.array([[0.0, 0.0, 0.0]]).T
                # target_xyz = np.array([[1.0, 0.5, 2.0]]).T
            # if t_now >= 10:
            #if t_now >= 8.0:
            
                #target_rpy = np.array([[0.2, 0.1, 0.3]]).T
                # nmpc.acados_init_p[33:36] = [0.0, 10.0, 0.0]
                # target_force = np.array([[0.0, -10.0, nmpc.acados_init_p[9] * nmpc.acados_init_p[10]]]).T
                # nmpc.acados_init_p[36:39] = [2.0, 1.0, 0.0]
                # target_torque = np.array([[-2.0, -1.0, 0.0]]).T
                #target_xyz = np.array([[0.5, 1.0, 2.0]]).T
                #target_rpy = np.array([[roll, pitch, -x_now_sim[14]-x_now_sim[15]]]).T
                 
                #target_xyz = np.array([[1.0, 1.5, 2]]).T
                #target_rpy = np.array([[0.0, 0.0, 0.2]]).T
            
               # target_rpy = np.array([[roll, pitch, -x_now_sim[14]-x_now_sim[15]]]).T
            #     roll = 30.0 / 180.0 * np.pi
            #     pitch = 0.0 / 180.0 * np.pi
            #     yaw = 0.0 / 180.0 * np.pi
            #     target_rpy = np.array([[roll, pitch, yaw]]).T

            # if t_now >= 8.0:
            #     nmpc.acados_init_p[30:33] = [np.pi/4, np.pi/4, np.pi/3]
            #     assert t_sqp_end <= 3.0
                # if np.sqrt((y_target-x_now_sim[1])**2 + (z_target-x_now_sim[2])**2) < 0.1:
                #     t_ctrl += ts_sim
                #y_target = 0.5 * np.cos(np.pi/3*(t_now-6))
                #z_target = 2 + 0.8 * np.sin(np.pi/3*(t_now-6))
            # if t_now >= 12.0:
            #     target_rpy = np.array([[-0.1, 0.3, 0.1]]).T
               # y_target = 0.5 * np.cos(np.pi/2*(t_now-9))
               # z_target = 2 + 0.8 * np.sin(np.pi/2*(t_now-9))
                #vy_target = -0.5*np.pi/2*np.sin(np.pi/2*(t_now-9))
               # vz_target = 0.8*np.pi/2*np.cos(np.pi/2*(t_now-9))
                #target_xyz = np.array([[1.0,  y_target, z_target]]).T
                #target_vxyz = np.array([[0.0,vy_target, vz_target]]).T
       
               # target_rpy = np.array([[roll, pitch, -x_now_sim[14]-x_now_sim[15]]]).T
            # if t_now >= 13.0:
            #     x_now_sim[25:28] = [0.0, 0.0, 0.0]
      

            
            # if  x_now_sim[3] > 0:
            #     collision = False
            # if t_now >= 14:
            #     x_now_sim[25:28] = [-3.0, 0.0, 0.0]
            # if t_now >= 18:
            #     x_now_sim[25:28] = [-2.0, 0.0, 0.0]
           # if t_now >= 12.5:
            # #     x_now_sim[25:28] = [-1.0, 0.0, 0.0]
            #     y_target = 0.5 * np.cos(np.pi/3*(t_now-12.5))
            #     z_target = 2 + 0.8 * np.sin(np.pi/3*(t_now-12.5))
            #     vy_target = -0.5*np.pi/3*np.sin(np.pi/3*(t_now-12.5))
            #     vz_target = 0.8*np.pi/3*np.cos(np.pi/3*(t_now-12.5))
                #target_xyz = np.array([[1.0, y_target, z_target]]).T
            #     target_vxyz = np.array([[0.0,vy_target, vz_target]]).T
                # target_rpy = np.array([[target_roll, target_pitch, 0.0]]).T
                # target_vxyz = (target_xyz - target_pxyz)/ ts_sim
                # target_wxyz = (target_rpy - target_prpy)/ ts_sim
                # target_pxyz = target_xyz
                # target_prpy = target_rpy
                #if 5*np.sin(np.pi/2*(t_now-6))>=0:
                #x_now_sim[16:22] = [20*np.sin(np.pi/2*(t_now-6)), 0.0, 5*np.sin(np.pi/2*(t_now-6)), 0.0, 0.0, 0.0]
                # else:
                #x_now_sim[19:25] = [0.0, 3.0, 0.0, 0.0, 0.0, 0.0]
            #     #x_now_sim[16:22] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # joint angles state
            # if t_now >= 8:
            #     #x_now_sim[16:22] = [14.0, 0.0, 7.0, 0.0, 0.0, 0.0]
            # if t_now >= 10:
            # #      # joint angles command
            #     x_now_sim[16:22] = [14.0, 0.0, 7.0, 0.0, 0.0, 0.0]
            # if t_now >= 12:
            #     x_now_sim[16:22] = [14.0, 0.0, 7.0, 0.0, 0.0, 0.0]
        # Compute reference trajectory from target pose
        xr, ur = nmpc.compute_trajectory(nmpc.acados_init_p[4:33], x_now, u_cmd, target_xyz, target_rpy, target_vxyz, target_wrpy, target_force=target_force, target_torque=target_torque)
        if args.plot_type == 2:
            if nx > 13:
                xr[:, 13:] = 0.0
            if args.arch == 'bi':
                ur[:, 2:] = 0.0
            elif args.arch == 'tri':
                ur[:, 3:] = 0.0
            elif args.arch == 'qd':
                ur[:, 4:] = 0.0

        # -------- Set SQP mode --------
        if is_sqp_change and t_sqp_start > t_sqp_end:
            if t_now >= t_sqp_start:
                ocp_solver.solver_options["nlp_solver_type"] = "SQP"

            if t_now >= t_sqp_end:
                ocp_solver.solver_options["nlp_solver_type"] = "SQP_RTI"

        # -------- Update solver --------
        comp_time_start = time.time()
        # -------- NMPC Controller --------    
        if t_ctl >= ts_ctrl:
            t_ctl = 0.0 
            # 0 ~ N-1
            for j in range(ocp_solver.N):
                yr = np.concatenate((xr[j, :], ur[j, :]))
                ocp_solver.set(j, "yref", yr)
                quaternion_r = xr[j, 6:10]
                nmpc.acados_init_p[0:4] = quaternion_r
                    # if t_now >= 7.0:
                ocp_solver.set(j, "p", nmpc.acados_init_p)  # For nonlinear quaternion error

            # N
            yr = xr[ocp_solver.N, :]
            ocp_solver.set(ocp_solver.N, "yref", yr)  # Final state of x, no u
            quaternion_r = xr[ocp_solver.N, 6:10]
            # print(yr)
            # print(ur[ocp_solver.N-1, :])
            # print("x_now: ", x_now)
            # print("u_cmd: ", u_cmd)
            # print("-----")
            nmpc.acados_init_p[0:4] = quaternion_r
            ocp_solver.set(ocp_solver.N, "p", nmpc.acados_init_p)  # For nonlinear quaternion error
            # external wrench estimation update
            sum_momentum, N = nmpc.get_comp_of_external_wrench(nmpc.acados_init_p[4:33], x_now)
            force_cmd, torque_cmd = nmpc.get_command_wrench(nmpc.acados_init_p[4:33], x_now, u_cmd)
            target_wrench = np.concatenate((force_cmd, torque_cmd))
            external_wrench_I_term += (target_wrench - N + est_external_wrench) * ts_ctrl
            ko = np.matrix(np.diag([8.0, 8.0, 8.0, 4.5, 4.5, 6.0]))
            est_external_wrench = np.squeeze(np.asarray(ko @ (sum_momentum-external_wrench_I_term)))
            # When the drone is on the ground, ignore the support force.
            est_external_wrench_plt = est_external_wrench.copy()
            I_matrix = nmpc.get_I_matrix(nmpc.acados_init_p[4:33], x_now)
            print("I matrix: ", I_matrix)
            if i < 200 * 4:
                est_external_wrench_plt[2] = 0.0
            # Compute control feedback and take the first action
            try:
                u_cmd = ocp_solver.solve_for_x0(x_now)
            except Exception as e:
                print(f"Round {i}: acados ocp_solver returned status {ocp_solver.status}. Exiting.")
                break
           # force, torque = nmpc.get_command_wrench(nmpc.acados_init_p[4:30], x_now_sim, u_cmd)  
            # print(force, torque)
            # print( nmpc.get_a_matrix(nmpc.acados_init_p[4:30], x_now_sim))
            # print(x_now_sim[13:16])
            # print(u_cmd[4:7])
            
            # if target_xyz[2,0] > 0 and take_off == False:
            #     take_off = True
            # if take_off == False:
            #     z_acc = 0.0
            # else:
            #     if i - take_off_i < 400:
            #         z_acc = z_acc * (i - take_off_i) / 400
            

        # --------- Admittance controller ----------
        # Ma(xd_ddot-xr_ddot) + Ca(xd_dot-xr_dot) + Ka(xd - xr) = Fext
        # We design Ma, Ca and Ka to realize the second order system in joints
        # Now xr_ddot = 0, xr_ddot = 0
        # Fext is get from the external force acting on the end effector

            # if t_now >= 10:
            #     Ma = 5
            #     Ca = 2*0.35*np.sqrt(75)
            #     Ka = 15

            #     xd_ddot = (x_now_sim[25] - Ka * (xd - xref) - Ca * xd_dot) / Ma
            #     #yd_ddot = (-x_now_sim[25]*np.sin(yaw) + x_now_sim[26]*np.cos(yaw)  - Ka * (yd - yref) - Ca * yd_dot) / Ma
            #     yd_ddot = 0.0
            #     xd += xd_dot * ts_sim
            #     yd += yd_dot * ts_sim
            #     pd = np.array([[xd, 0.0, 0.0]]).T
                
            #     xd_dot += xd_ddot * ts_sim
            #     yd_dot += yd_ddot * ts_sim 
            #     # Differential inverse kinematics

            #     print('xd: ', xd)
            #     theta = np.arccos(xd / (2 * 0.6))
            #     print("theta: ", theta)
            #     u_cmd[4:7] = [np.pi/2 - theta,2 * theta, -theta]




            #     #nmpc.cal_end_effector_position(nmpc.acados_init_p[4:30], x_now)
            #     #u_cmd[4:7] += np.squeeze(np.asarray(delta_q))
            #     # print("d_v: ", d_v)
            #     #print("xd: ", xd, "xf: ", xref)
            #     #print(pe_world, "x", x_now_sim[0], "y", x_now_sim[1])






        # --------- Update simulation ----------
   
        sim_solver.set("x", x_now_sim)
        sim_solver.set("u", u_cmd)
        sim_solver.set("p", nmpc.acados_init_p)
        #sim_solver.set("p", params=sim_nmpc.acados_init_p)

        status = sim_solver.solve()
        if status != 0:
            raise Exception(f"acados integrator returned status {status} in closed loop instance {i}")

        x_now_sim = sim_solver.get("x")

        ####################wall constraints######################
        # if t_now >= 8.0:
        #     pe_cog_sim, pe_world_sim = nmpc.get_end_effector_position(nmpc.acados_init_p[4:30], x_now_sim)  
        #     fric_ratio = 0.3
        #     x_cons = 1.99 + pe_world_sim[1]*0.2
         
        #     if pe_world_sim[0] >= x_cons:
        #         contact_force = 800*(x_cons-pe_world_sim[0]) + min(150.0*(0.0-x_now_sim[3]), 0.0)
        #         friction_force = fric_ratio * contact_force
        #         friction_force = 0.0
        #         y_fric = friction_force * (-x_now_sim[4] / (x_now_sim[4]**2 + x_now_sim[5]**2))
        #         z_fric = friction_force * (-x_now_sim[5] / (x_now_sim[4]**2 + x_now_sim[5]**2))
        #     else:
        #         contact_force = 0.0
        #         friction_force = 0.0
        #         y_fric = 0.0
        #         z_fric = 0.0

        #     x_now_sim[25:28] = [contact_force, y_fric, z_fric]
        ############################################################

        # --------- Add state constaints ----------
        # We add a wall here, the equation is y + x - 1 = 0. The drone could enter the area where y + x - 1 > 0
        # p = np.array([x_now_sim[0], x_now_sim[1], x_now_sim[2]]) # position vector
        # v = np.array([x_now_sim[3], x_now_sim[4], x_now_sim[5]]) # velocity vector
        # n = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2, 0.0]) # normal vector
        # # if (v @ n) < 0.0:
        # #     if p[0]+p[1]-1 > 0.0:
        # #         a = (p[0]+p[1]-1)/np.sqrt(2)
        # #         x_now_sim[0] += n[0]*a
        # #         x_now_sim[1] += n[1]*a
        # #         b = -(v @ n) * n
        # #         x_now_sim[3] += b[0]
        # #         x_now_sim[4] += b[1]
        
        if x_now_sim[5] < 0.0:
            if x_now_sim[2] <= 0.0:
                x_now_sim[2] = 0.0
                x_now_sim[5] = 0.0

      
        # if x_now_sim[13] > np.pi/2:
        #     x_now_sim[13] = np.pi/2
        # elif x_now_sim[13] < -np.pi/2:
        #     x_now_sim[13] = -np.pi/2
        # if x_now_sim[14] > np.pi/2:
        #     x_now_sim[14] = np.pi/2
        # elif x_now_sim[14] < -np.pi/2:
        #     x_now_sim[14] = -np.pi/2
        # if x_now_sim[15] > np.pi/2:
        #     x_now_sim[15] = np.pi/2
        # elif x_now_sim[15] < -np.pi/2:
        #     x_now_sim[15] = -np.pi/2
    


        # Save current simulation data for later comparison
        x_history.append(x_now_sim.copy())
        u_history.append(u_cmd.copy())
        r_now = [target_xyz[0,0], target_xyz[1,0], target_xyz[2,0], 
        target_rpy[0,0], target_rpy[1,0], target_rpy[2,0], 
        nmpc.acados_init_p[30], nmpc.acados_init_p[31], nmpc.acados_init_p[32],
        # nmpc.acados_init_p[33], nmpc.acados_init_p[34], nmpc.acados_init_p[35],
        # nmpc.acados_init_p[36], nmpc.acados_init_p[37], nmpc.acados_init_p[38],
        ur[ocp_solver.N-1, 4], ur[ocp_solver.N-1, 5], ur[ocp_solver.N-1, 6],
        ur[ocp_solver.N-1, 7], est_external_wrench_plt[4], est_external_wrench_plt[5]]
  
        # --------- Update visualizer ----------
        viz.update(i, x_now_sim, r_now, u_cmd)  # Note: The recording frequency of u_cmd is the same as ts_sim

    # ========== Visualize ==========
    if not args.no_viz:
        if args.plot_type == 0:
            viz.visualize(
                ocp_solver.acados_ocp.model.name,
                sim_solver.model_name,
                ts_ctrl,
                ts_sim,
                t_total_sim,
                t_servo_ctrl=t_servo_ctrl,
                t_servo_sim=t_servo_sim
            )
        elif args.plot_type == 1:
            viz.visualize_less(
                ts_sim,
                t_total_sim
            )
        elif args.plot_type == 2:
            viz.visualize_rpy(
                ocp_solver.acados_ocp.model.name,
                ts_sim,
                t_total_sim
            )

    if args.save_data:
        file_path = args.file_path

        np.savez(
            file_path + f"nmpc_{type(nmpc).__name__}_model_{type(sim_nmpc).__name__}.npz",
            x=np.array(x_history),
            u=np.array(u_history)
        )
    return np.array(x_history), np.array(u_history)


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Run the simulation of different NMPC models.")
    parser.add_argument(
        "model",
        type=int,
        help="The NMPC model to be simulated. "
             "Options: 0 (basic model), 1 (servo), "
             "2 (thrust), 3(servo+thrust), "
             "21 (servo+dist), 22 (servo+thrust+dist), "
             "91(no_servo_new_cost), 92(servo_old_cost), "
             "93(servo_diff), 94(servo+drag+dist), "
             "95 (servo+thrust+drag), 96 (servo+drag_param+dist).",
    )

    parser.add_argument(
        "-sim",
        "--sim_model",
        type=int,
        default=0,
        help="The simulation model. "
             "Options: 0 (default: servo+thrust), "
             "1 (servo+thrust+drag).",
    )

    parser.add_argument(
        "-p",
        "--plot_type",
        type=int,
        default=0,
        help="The type of plot. "
             "Options: 0 (default: full), 1 (less), 2 (only rpy)."
    )

    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default='qd',
        help="The robot's architecture. Options: bi, tri, qd (default)."
    )

    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable visualization after simulation. Note that this is different from the plot_type option, "
             "because plot_type also decides the simulation parameters."
    )

    parser.add_argument(
        "-s",
        "--save_data",
        action="store_true",
        help="Save simulation x and u data to file"
    )

    parser.add_argument(
        "--file_path",
        type=str,
        default=f"../../../../test/data/",
        help="Path to save the data file"
    )

    args = parser.parse_args()
    main(args)
