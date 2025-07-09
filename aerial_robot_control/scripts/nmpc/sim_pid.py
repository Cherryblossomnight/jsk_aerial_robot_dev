import sys, os
from copy import deepcopy
import time
import numpy as np
import argparse
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/tilt_bi")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/tilt_tri")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/tilt_qd")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/hydrus")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/archive")

from pid_viz import Visualizer

# Quadrotor
import tilt_qd.phys_param_beetle_omni as phys_omni
import archive.phys_param_beetle_art as phys_art
import hydrus.phys_param_hydrus as phys_hydrus
# - Naive models
from archive.tilt_qd_no_servo_ac_cost import NMPCTiltQdNoServoAcCost
from tilt_qd.tilt_qd_no_servo import NMPCTiltQdNoServo


from hydrus.hydrus_thrust import HydrusThrust
# - Consider the servo delay with its model
from tilt_qd.tilt_qd_servo import NMPCTiltQdServo
from tilt_qd.tilt_qd_servo_dist import NMPCTiltQdServoDist
from archive.tilt_qd_servo_drag_w_dist import NMPCTiltQdServoDragDist
from archive.tilt_qd_servo_w_cog_end_dist import NMPCTiltQdServoWCogEndDist

from archive.tilt_qd_servo_old_cost import NMPCTiltQdServoOldCost
from tilt_qd.tilt_qd_servo_diff import NMPCTiltQdServoDiff

# - Consider the thrust delay with its model
from tilt_qd.tilt_qd_thrust import NMPCTiltQdThrust

# - Consider the servo & thrust delay with its models
from tilt_qd.tilt_qd_servo_thrust import NMPCTiltQdServoThrust
from tilt_qd.tilt_qd_servo_thrust_dist import NMPCTiltQdServoThrustDist
from archive.tilt_qd_servo_thrust_drag import NMPCTiltQdServoThrustDrag

# Birotor
from tilt_bi.tilt_bi_servo import NMPCTiltBiServo
from tilt_bi.tilt_bi_2ord_servo import NMPCTiltBi2OrdServo

# Trirotor
from tilt_tri.tilt_tri_servo import NMPCTiltTriServo
from tilt_tri.tilt_tri_servo_dist import NMPCTiltTriServoDist


def main(args):
    # ========== Init ==========
    # ---------- Controller ----------
    if args.arch == 'qd':

        if args.model == 0:
            nmpc = HydrusThrust(phys=phys_hydrus)
        elif args.model == 1:
            nmpc = NMPCTiltQdServo(phys=phys_art)
        elif args.model == 2:
            nmpc = NMPCTiltQdThrust(phys=phys_art)
        elif args.model == 3:
            nmpc = NMPCTiltQdServoThrust(phys=phys_art)

        elif args.model == 21:
            nmpc = NMPCTiltQdServoDist(phys=phys_omni)
        elif args.model == 22:
            nmpc = NMPCTiltQdServoThrustDist(phys=phys_omni)

        # Archived methods
        elif args.model == 91:
            nmpc = NMPCTiltQdNoServoAcCost()
        elif args.model == 92:
            nmpc = NMPCTiltQdServoOldCost()
        elif args.model == 93:
            nmpc = NMPCTiltQdServoDiff()
            alpha_integ = np.zeros(4)
        elif args.model == 94:
            nmpc = NMPCTiltQdServoDragDist()
        elif args.model == 95:
            nmpc = NMPCTiltQdServoThrustDrag()
        elif args.model == 96:
            nmpc = NMPCTiltQdServoWCogEndDist()
        else:
            raise ValueError(f"Invalid control model {args.model}.")

    elif args.arch == 'bi':

        if args.model == 0:
            nmpc = NMPCTiltBiServo()
        elif args.model == 1:
            nmpc = NMPCTiltBi2OrdServo()
        else:
            raise ValueError(f"Invalid model {args.model}.")

    elif args.arch == 'tri':

        if args.model == 0:
            nmpc = NMPCTiltTriServo()
        elif args.model == 1:
            nmpc = NMPCTiltTriServoDist()
        else:
            raise ValueError(f"Invalid model {args.model}.")

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
    u_init = np.zeros(nu)
    u_init[0:4] = [0.0, 0.0, 0.0, 0.0]  # thrust
    u_init[4:7] = [np.pi/2, np.pi/2, np.pi/2]   # joint angles command

    for stage in range(ocp_solver.N + 1):
        ocp_solver.set(stage, "x", x_init)
    for stage in range(ocp_solver.N):
        ocp_solver.set(stage, "u", u_init)

    # ---------- Simulator ----------
    if args.arch == 'qd':
        sim_nmpc = HydrusThrust(phys=phys_hydrus)
        # sim_phy = phys_omni if 20 < args.model < 30 else phys_art
        # if args.sim_model == 0:
        #     sim_nmpc = NMPCTiltQdServoThrust(phys=sim_phy)  # Consider both the servo delay and the thrust delay
        # elif args.sim_model == 1:
        #     sim_nmpc = NMPCTiltQdServoThrustDrag(phys=sim_phy)  # Also consider drag in wrench formulation
        # else:
        #     raise ValueError(f"Invalid sim model {args.sim_model}.")

    elif args.arch == 'bi':

        if args.sim_model == 0:
            sim_nmpc = NMPCTiltBiServo()
        # elif args.sim_model == 1:
        #     sim_nmpc = NMPCTiltBi2OrdServo()   # This model is wrong
        else:
            raise ValueError(f"Invalid sim model {args.sim_model}.")

    elif args.arch == 'tri':

        sim_nmpc = NMPCTiltTriServo()

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

    t_total_sim = 15.0
    if args.plot_type == 1:
        t_total_sim = 4.0
    if args.plot_type == 2:
        t_total_sim = 3.0

    N_sim = int(t_total_sim / ts_sim)
 

    # Sim solver
    sim_solver = sim_nmpc.create_acados_sim_solver(ts_sim, is_build=True)
    nx_sim = sim_solver.acados_sim.dims.nx
    nr = 15
    r_init = [0.3, 0.6, 1.0, 0.0, 0.0, 0.0, np.pi/2, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # State Initialization
    x_init_sim = np.zeros(nx_sim)
    x_init_sim[6] = 1.0  # qw
    x_init_sim[13:16] = [np.pi/2, np.pi/2, np.pi/2]   # joint angles state
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
    y_target = 0.5
    z_target = 2.0
    t_ctrl = 6.0
    target_vxyz = np.zeros(3)
    target_wxyz = np.zeros(3)
    target_pxyz = np.array([[0.0, 0.5, 2.0]]).T
    target_prpy = np.array([[0.0, 0.0, 0.0]]).T
    xd_ddot = 0.0
    xd_dot = 0.0
    xd = 0.90466333

    xref = xd
    yd_ddot = 0.0
    yd_dot = 0.0
    yd = 0.52230762
    yref = yd
    cmd = False
    pe_cog_sim = np.zeros(3)
    pe_world_sim = np.zeros(3)
    lam = 1.0
    last_err = 0.0
    last_q = u_cmd[4:7]
    sim_start = False
    for i in range(N_sim):
        # --------- Update time ---------
        t_now = i * ts_sim
        t_ctl += ts_sim
        # --------- Add state constaints ---------
       

        # --------- Update state estimation ---------
        # Assemble state from simulation and disturbance estimation 
        if nmpc.include_cog_dist_model:
            x_now = np.zeros(nx)
            if nmpc.include_end_effector_dist_model:
                x_now[: nx - 9] = deepcopy(x_now_sim[: nx - 9])
            else:
                x_now[: nx - 6] = deepcopy(x_now_sim[: nx - 6])
        else:
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
        # target_vxyz = np.array([[0.0, 0.0, 0.0]]).T
        # target_wxyz = np.array([[0.0, 0.0, 0.0]]).T
        if args.plot_type == 2:
            #target_xyz = np.array([[1.0, 1.5, 2.0]]).T
            target_rpy = np.array([[0.0, 0.0, 0.0]]).T

        if t_total_sim > 2.0:
            if 2.0 <= t_now :
                #target_xyz = np.array([[1.0, 1.5, 2.0]]).T

                roll = 30.0 / 180.0 * np.pi
                pitch = 60.0 / 180.0 * np.pi
                yaw = 30.0 / 180.0 * np.pi
           # if t_now >= 4:
                #target_xyz = np.array([[1.0, 1.5, 2.0]]).T
                
               # x_now_sim[16:22] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                # u_cmd[4:7] = [np.pi/2, 0, 0] 
            # if 3.0 <= t_now < 5.5:
            #     assert t_sqp_end <= 3.0
            #     target_xyz = np.array([[1.0, 1.0, 1.0]]).T
            #     target_rpy = np.array([[0.0, 0.0, 0.0]]).T
            if t_now >= 6:
            #     target_xyz = np.array([[1.0, 1.0, 1.0]]).T
               
                if not cmd:
                    u_cmd[4:7] = [np.pi/3, np.pi/3, -np.pi/6]
                    cmd = True
                #target_xyz = np.array([[1.0, 1.5, 2]]).T
                #target_rpy = np.array([[0.0, 0.0, 0.0]]).T
            if t_now >= 7:
                target_rpy = np.array([[roll, pitch, -x_now_sim[14]-x_now_sim[15]]]).T
            #     roll = 30.0 / 180.0 * np.pi
            #     pitch = 0.0 / 180.0 * np.pi
            #     yaw = 0.0 / 180.0 * np.pi
            #     target_rpy = np.array([[roll, pitch, yaw]]).T

            if t_now >= 10:
                target_xyz = np.array([[0.0, 0.0, 2.0]]).T
            #     assert t_sqp_end <= 3.0
                # if np.sqrt((y_target-x_now_sim[1])**2 + (z_target-x_now_sim[2])**2) < 0.1:
                #     t_ctrl += ts_sim
                #y_target = 0.5 * np.cos(np.pi/3*(t_now-6))
                #z_target = 2 + 0.8 * np.sin(np.pi/3*(t_now-6))

                x_now_sim[25:28] = [5.0, 0.0, 0.0]
            # if t_now >= 14:
            #     x_now_sim[25:28] = [-3.0, 0.0, 0.0]
            # if t_now >= 18:
            #     x_now_sim[25:28] = [-2.0, 0.0, 0.0]
            # if t_now >= 22:
            #     x_now_sim[25:28] = [-1.0, 0.0, 0.0]
                # target_xyz = np.array([[0.0, y_target, z_target]]).T
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
        xr, ur = reference_generator.compute_trajectory(target_xyz, target_rpy)

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
        # -------- PID Controller --------
        # if t_ctl >= ts_ctrl:
        #     t_ctl = 0.0
        #     z_acc = 24.0*(target_xyz[2,0]-x_now[2])+10.0*(0.0-x_now[5]) + 9.798*3.4
        #     quat = [x_now[7], x_now[8], x_now[9], x_now[6]]
        #     roll, pitch, yaw = euler_from_quaternion(quat)
        #     x_acc = 1.0*(target_xyz[0,0]-x_now[0]) + 1.6*(0.0-x_now[3])
        #     y_acc = 1.0*(target_xyz[1,0]-x_now[1]) + 1.6*(0.0-x_now[4])
        #     target_roll = -y_acc*np.cos(yaw) + x_acc*np.sin(yaw) / 9.798
        #     target_pitch = y_acc*np.sin(yaw) + x_acc*np.cos(yaw) / 9.798
        #     roll_acc = 9.0*(target_roll-roll) + 5.0*(0.0-x_now[10])
        #     pitch_acc = 9.0*(target_pitch-pitch) + 5.0*(0.0-x_now[11])
        #     yaw_acc = 9.0*(target_rpy[2,0]-yaw) + 6.0*(0.0-x_now[12])
        #     mat = nmpc.get_alloc_matrix(nmpc.acados_init_p[4:30], x_now)
    
        #     delta_u = np.linalg.pinv(mat) @ np.array([z_acc, roll_acc, pitch_acc, yaw_acc])
        #     u_cmd[0:4] = np.squeeze(np.asarray(delta_u)) 
        # -------- Impedance Controller --------
        
        if t_ctl >= ts_ctrl:
            t_ctl = 0.0
            m = 3.4
            zeta = 2.0
            # if est_external_wrench[0] > 3 or est_external_wrench[0] < -3:
            #     mdx = np.abs(1 * est_external_wrench[0])
            # else:
            mdx = 10.0
            # if est_external_wrench[1] > 5:
            #     mdy = 1 * est_external_wrench[1]
            # else:
            mdy = 5.0
            #if est_external_wrench[2] > 5:
                #mdz = 1 * est_external_wrench[2]
            #else:
            mdz = 5.0
            # calcaluate the translation acceleration
            x_acc = ( - 1/m) * est_external_wrench[0] + 2.0*(target_xyz[0,0]-x_now[0]) + 2*0.7*np.sqrt(2.0)*(0.0-x_now[3])
            y_acc = (1/mdy - 1/m) * est_external_wrench[1] + 2.0*(target_xyz[1,0]-x_now[1]) + 2*0.7*np.sqrt(2.0)*(0.0-x_now[4])
            z_acc = (1/mdz - 1/m) * est_external_wrench[2] + 8.0*(target_xyz[2,0]-x_now[2]) + 2*1.0*np.sqrt(8.0)*(0.0-x_now[5]) + 9.798
            # x_acc = 1.0*(target_xyz[0,0]-x_now[0]) + 1.6*(0.0-x_now[3])
            # y_acc = 1.0*(target_xyz[1,0]-x_now[1]) + 1.6*(0.0-x_now[4])
            # z_acc = 8.0*(target_xyz[2,0]-x_now[2]) + 5.0*(0.0-x_now[5]) + 9.798
            
            if target_xyz[2,0] > 0 and take_off == False:
                take_off = True
                take_off_i = i
            if take_off == False:
                z_acc = 0.0
            else:
                if i - take_off_i < 400:
                    z_acc = z_acc * (i - take_off_i) / 400
            quat = [x_now[7], x_now[8], x_now[9], x_now[6]]
            roll, pitch, yaw = euler_from_quaternion(quat)
            rot_cog = quaternion_matrix(quat)[:3, :3]
            target_roll = np.arctan2((-y_acc*np.cos(yaw) + x_acc*np.sin(yaw)), z_acc)
            target_pitch = np.arctan2((y_acc*np.sin(yaw) + x_acc*np.cos(yaw)), z_acc)
            real_x_acc = (np.tan(roll)*np.sin(yaw) + np.tan(pitch)*np.cos(yaw)) * z_acc
            real_y_acc = (-np.tan(roll)*np.cos(yaw) + np.tan(pitch)*np.sin(yaw)) * z_acc
            I = nmpc.get_I_matrix(nmpc.acados_init_p[4:30], x_now)
            I_d = np.matrix([[2, 0.0, 0.0],
                             [0.0, 2, 0.0],
                             [0.0, 0.0, 4]])
            I_d = I
            Kp = np.matrix([[20.0, 0.0, 0.0],
                            [0.0, 20.0, 0.0],
                            [0.0, 0.0, 4.0]])
            Kd = np.matrix([[2*0.5*np.sqrt(20*2), 0.0, 0.0],
                            [0.0, 2*0.5*np.sqrt(20*2), 0.0],
                            [0.0, 0.0, 2*0.5*np.sqrt(4*4)]])
            ko = np.matrix(np.diag([6.0, 6.0, 6.0, 4.5, 4.5, 4.5]))
            delta_x = np.array([target_roll-roll, target_pitch-pitch, target_rpy[2,0]-yaw])
            delta_v = np.array([0.0-x_now[10], 0.0-x_now[11], 0.0-x_now[12]])
    
            c_vector = nmpc.get_c_vector(nmpc.acados_init_p[4:30], x_now)
            # calcaluate the command torque
            torque = ((I * I_d.I - np.eye(3)) @ est_external_wrench[3:6]).reshape(3, 1) + (Kp @ delta_x + Kd @ delta_v).reshape(3, 1) 
            #torque = (Kp @ delta_x + Kd @ delta_v).reshape(3, 1) 
            # print((I * I_d.I - np.eye(3)) @ x_now_sim[19:22])
            # print((Kp @ delta_x + Kd @ delta_v))
            # print("aa")
            torque = np.squeeze(np.asarray(torque)) 
            torque_w = rot_cog @ torque
            sum_momentum, N = nmpc.get_comp_of_external_wrench(nmpc.acados_init_p[4:30], x_now)
            # # When drone is on the ground, the N_vector(Gravity and Coriolis force) should be zero.
            # # Now, the dynamic equation is Mv_cot + Cv + g = tao_ext + tao_cmd + N, Here N is the support force.
            # # So, the N_vector now is (Cv+g-N) = 0)
            target_wrench = np.array([real_x_acc*m, real_y_acc*m, z_acc*m, torque_w[0], torque_w[1], torque_w[2]])
            external_wrench_I_term += (target_wrench - N + est_external_wrench) * ts_ctrl
            est_external_wrench = np.squeeze(np.asarray(ko @ (sum_momentum-external_wrench_I_term)))
            # When the drone is on the ground, ignore the support force.
            est_external_wrench_plt = est_external_wrench.copy()
            if i < 200 * 4:
                est_external_wrench_plt[2] = 0.0
            # print("target_wrench: ", target_wrench) 
            # print("est_external_wrench: ", est_external_wrench)
            mat = nmpc.get_alloc_matrix(nmpc.acados_init_p[4:30], x_now)
            delta_u = np.linalg.pinv(mat) @ np.array([z_acc*m, torque[0], torque[1], torque[2]])
            pe_cog, pe_world = nmpc.get_end_effector_position(nmpc.acados_init_p[4:30], x_now)
            # print("yaw: ", yaw)
            # print("joint", x_now[13:16])
            # print("euler_acc: ", euler_acc)
            # print("delta_u: ", delta_u)
            # print("mat: ", np.linalg.pinv(mat))
            u_cmd[0:4] = np.squeeze(np.asarray(delta_u)) 

        # --------- Admittance controller ----------
        # Ma(xd_ddot-xr_ddot) + Ca(xd_dot-xr_dot) + Ka(xd - xr) = Fext
        # We design Ma, Ca and Ka to realize the second order system in joints
        # Now xr_ddot = 0, xr_ddot = 0
        # Fext is get from the external force acting on the end effector

            if t_now >= 10:
                Ma = 5
                Ca = 2*0.35*np.sqrt(750)
                Ka = 150

                xd_ddot = (x_now_sim[25]*np.cos(yaw) + x_now_sim[26]*np.sin(yaw) - Ka * (xd - xref) - Ca * xd_dot) / Ma
                yd_ddot = (-x_now_sim[25]*np.sin(yaw) + x_now_sim[26]*np.cos(yaw)  - Ka * (yd - yref) - Ca * yd_dot) / Ma
                xd += xd_dot * ts_sim
                yd += yd_dot * ts_sim
                pd = np.array([[xd, yd, 0.0]]).T
                
                xd_dot += xd_ddot * ts_sim
                yd_dot += yd_ddot * ts_sim 
                # Differential inverse kinematics
                x_sim = x_now
                x_sim[13:16] = last_q
                pe_cog_tar = [xd, yd, pe_cog[2]]
                print("pe_cog_tar: ",pe_cog_tar)
                if not sim_start:
                    pe_cog_sim = pe_cog
                    sim_start = True
                    print("pe_cog_sim: ",pe_cog_sim)
                print("x_sim[13:16]: ", x_sim[13:16])
                x_sim[13:16] = nmpc.inv_kinematics(nmpc.acados_init_p[4:30], x_sim, pe_cog_tar)
                #print("yaw: ", yaw)
                # x_sim[13:16] += np.squeeze(np.asarray(delta_q))
                pe_cog_sim, pe_world_sim = nmpc.get_end_effector_position(nmpc.acados_init_p[4:30], x_sim)  
                # iter = 0 
                # while True:
                #     if iter == 0:
                #         print("x_sim: ", x_sim[13:16])
                #         print("pe_cog_sim: ", pe_cog_sim,"pe_world_tar: ", pe_cog_tar,np.sqrt((pe_cog_sim[0]-pe_cog_tar[0])**2 + (pe_cog_sim[1]-pe_cog_tar[1])**2))
                #     if np.sqrt((pe_cog_sim[0]-pe_cog_tar[0])**2 + (pe_cog_sim[1]-pe_cog_tar[1])**2) < 1e-3:
                #         print("break")
                #         #u_cmd[4:7] = x_sim[13:16]
                #         break
                #     jacobian = nmpc.get_end_Jacobian(nmpc.acados_init_p[4:30], x_sim)
                  
                   
                #     # print("dv:", d_v)
                #     # print("dq:",delta_q.T)
                #     #print("x_sim: ", x_sim[13:16])
                #     pe_cog_sim, pe_world_sim = nmpc.get_end_effector_position(nmpc.acados_init_p[4:30], x_now)  
               
                #     #if last_err < np.sqrt((pe_world_sim[0]-pe_world_tar[0])**2 + (pe_world_sim[1]-pe_world_tar[1])**2):
                #         #last_err = np.sqrt((pe_world_sim[0]-pe_world_tar[0])**2 + (pe_world_sim[1]-pe_world_tar[1])**2)
                   
                #     #     delta_q = np.linalg.pinv(jacobian) @ (lam * d_v)
                #     #     x_sim[13:16] += np.squeeze(np.asarray(delta_q))
                #     #     pe_cog_sim, pe_world_sim = nmpc.get_end_effector_position(nmpc.acados_init_p[4:30], x_now)  
                #     # lam = 1.0
                #     #print(jacobian)
                #    #print(np.linalg.pinv(jacobian))
                #     #print(d_v)
                #     #print(delta_q)
                #     #print("---")
                #     #print("pe_world_tar: ", pe_world_tar,"pe_world_sim: ", pe_world_sim," ",np.sqrt((pe_world_sim[0]-pe_world_tar[0])**2 + (pe_world_sim[1]-pe_world_tar[1])**2))
                #     #print(np.sqrt((pe_world_sim[0]-pe_world_tar[0])**2 + (pe_world_sim[1]-pe_world_tar[1])**2))
                #     iter += 1 
                #     #print("pe_cog_sim: ", pe_cog_sim,"pe_cog_tar: ", pe_cog_tar,np.sqrt((pe_cog_sim[0]-pe_cog_tar[0])**2 + (pe_cog_sim[1]-pe_cog_tar[1])**2))
                #     if iter > 50:
                #         print("break iter")
                        
                #         break
                print("pe_world: ", pe_world,"x_now: ", x_now_sim[0:3])
                print("x_sim: ", x_sim[13:16])
                print("yaw", yaw)
                print(np.sqrt((pe_cog_sim[0]-pe_cog_tar[0])**2 + (pe_cog_sim[1]-pe_cog_tar[1])**2))
                last_q = x_sim[13:16]
                u_cmd[4:7] = x_sim[13:16]
                #nmpc.cal_end_effector_position(nmpc.acados_init_p[4:30], x_now)
                #u_cmd[4:7] += np.squeeze(np.asarray(delta_q))
                # print("d_v: ", d_v)
                #print("xd: ", xd, "xf: ", xref)
                #print(pe_world, "x", x_now_sim[0], "y", x_now_sim[1])






        # --------- Update simulation ----------
      
        sim_solver.set("x", x_now_sim)
        sim_solver.set("u", u_cmd)

        status = sim_solver.solve()
        if status != 0:
            raise Exception(f"acados integrator returned status {status} in closed loop instance {i}")

        x_now_sim = sim_solver.get("x")
        # --------- Add state constaints ----------
        # We add a wall here, the equation is y + x - 1 = 0. The drone could enter the area where y + x - 1 > 0
        p = np.array([x_now_sim[0], x_now_sim[1], x_now_sim[2]]) # position vector
        v = np.array([x_now_sim[3], x_now_sim[4], x_now_sim[5]]) # velocity vector
        n = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2, 0.0]) # normal vector
        # if (v @ n) < 0.0:
        #     if p[0]+p[1]-1 > 0.0:
        #         a = (p[0]+p[1]-1)/np.sqrt(2)
        #         x_now_sim[0] += n[0]*a
        #         x_now_sim[1] += n[1]*a
        #         b = -(v @ n) * n
        #         x_now_sim[3] += b[0]
        #         x_now_sim[4] += b[1]
        
        if x_now_sim[5] < 0.0:
            if x_now_sim[2] <= 0.0:
                x_now_sim[2] = 0.0
                x_now_sim[5] = 0.0
        if x_now_sim[13] > np.pi/2:
            x_now_sim[13] = np.pi/2
        elif x_now_sim[13] < -np.pi/2:
            x_now_sim[13] = -np.pi/2
        if x_now_sim[14] > np.pi/2:
            x_now_sim[14] = np.pi/2
        elif x_now_sim[14] < -np.pi/2:
            x_now_sim[14] = -np.pi/2
        if x_now_sim[15] > np.pi/2:
            x_now_sim[15] = np.pi/2
        elif x_now_sim[15] < -np.pi/2:
            x_now_sim[15] = -np.pi/2
    


        # Save current simulation data for later comparison
        x_history.append(x_now_sim.copy())
        u_history.append(u_cmd.copy())
        r_now = [target_xyz[0,0], target_xyz[1,0], target_xyz[2,0], target_roll, target_pitch, target_rpy[2,0], u_cmd[4], u_cmd[5], u_cmd[6], est_external_wrench_plt[0], est_external_wrench_plt[1], est_external_wrench_plt[2], est_external_wrench_plt[3], est_external_wrench_plt[4], est_external_wrench_plt[5]]
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
