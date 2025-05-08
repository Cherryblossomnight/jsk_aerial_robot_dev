import sys, os
from copy import deepcopy
import time
import numpy as np
import argparse
from tf.transformations import euler_from_quaternion

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
    u_init[0:4] = [8.3283, 8.3283, 8.3283, 8.3283]  # thrust
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
    nr = 9
    r_init = [0.3, 0.6, 1.0, 0.0, 0.0, 0.0, np.pi/2, np.pi/2, np.pi/2]
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
    for i in range(N_sim):
        # --------- Update time ---------
        t_now = i * ts_sim
        t_ctl += ts_sim
        # --------- Add state constaints ---------
       

        # --------- Update state estimation ---------
        # Assemble state from simulation and disturbance estimation 
        if nmpc.include_cog_dist_model:
            x_now = np.zeros(nx)
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

        if args.plot_type == 2:
            target_xyz = np.array([[0.0, 0.0, 2.0]]).T
            target_rpy = np.array([[0.0, 0.0, 0.0]]).T

        if t_total_sim > 2.0:
            if 2.0 <= t_now :
                target_xyz = np.array([[0.0, 0.0, 2.0]]).T

                roll = 30.0 / 180.0 * np.pi
                pitch = 60.0 / 180.0 * np.pi
                yaw = 30.0 / 180.0 * np.pi
            if t_now >= 4:
                target_xyz = np.array([[1.0, 0.0, 2.0]]).T
                target_rpy = np.array([[roll, pitch, 0.0]]).T
                x_now_sim[16:22] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                # u_cmd[4:7] = [np.pi/2, 0, 0] 
            # if 3.0 <= t_now < 5.5:
            #     assert t_sqp_end <= 3.0
            #     target_xyz = np.array([[1.0, 1.0, 1.0]]).T
            #     target_rpy = np.array([[0.0, 0.0, 0.0]]).T
            # if t_now >= 5.5:
            #     target_xyz = np.array([[1.0, 1.0, 1.0]]).T

            #     roll = 30.0 / 180.0 * np.pi
            #     pitch = 0.0 / 180.0 * np.pi
            #     yaw = 0.0 / 180.0 * np.pi
            #     target_rpy = np.array([[roll, pitch, yaw]]).T

            if t_now >= 6:
                assert t_sqp_end <= 3.0
                target_xyz = np.array([[1.0, 0.0, 2.0]]).T
                target_rpy = np.array([[0.0, 0.0, 0.0]]).T
                x_now_sim[16:22] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # joint angles state
            # if t_now >= 8:
            #     x_now_sim[16:22] = [0.0, 3.0, 0.0, 0.0, 0.0, 0.0]
            if t_now >= 8:
                 # joint angles command
                x_now_sim[16:22] = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if t_now >= 10:
                x_now_sim[16:22] = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
            md = 5.0

            x_acc = (1/md - 1/m) * x_now_sim[16] + 1.0*(target_xyz[0,0]-x_now[0]) + 1.6*(0.0-x_now[3])
            y_acc = (1/md - 1/m) * x_now_sim[17] + 1.0*(target_xyz[1,0]-x_now[1]) + 1.6*(0.0-x_now[4])
            z_acc = (1/md - 1/m) * x_now_sim[18] + 8.0*(target_xyz[2,0]-x_now[2]) + 5.0*(0.0-x_now[5]) + 9.798
            if t_now < 2.0:
                z_acc = z_acc * i / 400

            quat = [x_now[7], x_now[8], x_now[9], x_now[6]]
            roll, pitch, yaw = euler_from_quaternion(quat)
            target_roll = (-y_acc*np.cos(yaw) + x_acc*np.sin(yaw)) / 9.798
            target_pitch = (y_acc*np.sin(yaw) + x_acc*np.cos(yaw)) / 9.798
            I = nmpc.get_I_matrix(nmpc.acados_init_p[4:30], x_now)
            I_d = np.matrix([[2, 0.0, 0.0],
                             [0.0, 2, 0.0],
                             [0.0, 0.0, 4]])
            I_d = I
            Kp = np.matrix([[12.0, 0.0, 0.0],
                            [0.0, 12.0, 0.0],
                            [0.0, 0.0, 3.0]])
            Kd = np.matrix([[7.0, 0.0, 0.0],
                            [0.0, 7.0, 0.0],
                            [0.0, 0.0, 2.0]])
            delta_x = np.array([target_roll-roll, target_pitch-pitch, target_rpy[2,0]-yaw])
            delta_v = np.array([0.0-x_now[10], 0.0-x_now[11], 0.0-x_now[12]])
    
            c_vector = nmpc.get_c_vector(nmpc.acados_init_p[4:30], x_now)
            torque = ((I * I_d.I - np.eye(3)) @ x_now_sim[19:22]).reshape(3, 1) + (Kp @ delta_x + Kd @ delta_v).reshape(3, 1) 
            # print((I * I_d.I - np.eye(3)) @ x_now_sim[19:22])
            # print((Kp @ delta_x + Kd @ delta_v))
            # print("aa")
            torque = np.squeeze(np.asarray(torque)) 
            mat = nmpc.get_alloc_matrix(nmpc.acados_init_p[4:30], x_now)
           
            delta_u = np.linalg.pinv(mat) @ np.array([z_acc*m, torque[0], torque[1], torque[2]])
            # print("target_roll: ", target_roll)
            # print("target_pitch: ", target_pitch)
            # print("euler_acc: ", euler_acc)
            # print("delta_u: ", delta_u)
            # print("mat: ", np.linalg.pinv(mat))
            u_cmd[0:4] = np.squeeze(np.asarray(delta_u)) 

        # --------- Update simulation ----------
      
        sim_solver.set("x", x_now_sim)
        sim_solver.set("u", u_cmd)

        status = sim_solver.solve()
        if status != 0:
            raise Exception(f"acados integrator returned status {status} in closed loop instance {i}")

        x_now_sim = sim_solver.get("x")
        # --------- Add state constaints ----------
        if x_now_sim[5] < 0.0:
            if x_now_sim[2] <= 0.0:
                x_now_sim[2] = 0.0
                x_now_sim[5] = 0.0
        if x_now_sim[3] > 0.0:
            if x_now_sim[0] >= 0.8:
                x_now_sim[0] = 0.8
                x_now_sim[3] = 0.0


        # Save current simulation data for later comparison
        x_history.append(x_now_sim.copy())
        u_history.append(u_cmd.copy())
        r_now = [target_xyz[0,0], target_xyz[1,0], target_xyz[2,0], target_roll, target_pitch, target_rpy[2,0], u_cmd[4], u_cmd[5], u_cmd[6]]
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
