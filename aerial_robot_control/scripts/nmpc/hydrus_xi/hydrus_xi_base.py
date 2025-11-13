import os, sys
from abc import abstractmethod
import numpy as np
from acados_template import AcadosModel, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import casadi as ca
from tf_conversions import transformations as tf

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rh_base import RecedingHorizonBase
from hydrus_xi.hydrus_xi_reference_generator import HydrusXiReferenceGenerator


class HydrusXiBase(RecedingHorizonBase):
    """
    Base class for all multilinked aerial robots link hydrus and dragon.
    Inherits from RecedingHorizonBase which also lays foundations for MHE classes.

    :param str model_name: Name of the model defined in controller file.
    :param bool overwrite: Flag to overwrite existing c generated code for the OCP solver. Default: False
    """

    def __init__(self, overwrite: bool = False):
        #     The child classes only have specifications which define the controller specifications and need to set the following flags:
        # check if the model name is set
        # - model_name: Name of the model defined in controller file.
        if not hasattr(self, "model_name"):
            raise AttributeError("Model name not set. Please set the model_name attribute in the child class.")
        # - phys: Physical parameters of the robot.
        if not hasattr(self, "phys"):
            raise AttributeError("Physical parameters not set. Please set the phys attribute in the child class.")
        # - tilt: Flag to include tiltable rotors. If not included, the quadrotor is assumed to be a fixed quadrotor.
        if not hasattr(self, "tilt"):
            raise AttributeError("Tilt flag not set. Please set the tilt attribute in the child class.")
        # - include_servo_model: Flag to include the servo model based on the angle alpha (a) between frame E (end of arm) and R (rotor). If not included, angle control is assumed to be equal to angle state.
        if not hasattr(self, "include_servo_model"):
            raise AttributeError(
                "Servo model flag not set. Please set the include_servo_model attribute in the child class.")
        # - include_servo_derivative: Flag to include the continuous time-derivative of the servo angle as control input(!) instead of numeric differentation.
        if not hasattr(self, "include_servo_derivative"):
            self.include_servo_derivative = False
        # - include_thrust_model: Flag to include dynamics from rotor and use thrust as state. If not included, thrust control is assumed to be equal to thrust state.
        if not hasattr(self, "include_thrust_model"):
            raise AttributeError(
                "Thrust model flag not set. Please set the include_thrust_model attribute in the child class.")

        # Disturbance on each rotor individually was investigated into but didn't properly work, therefore only disturbance on CoG implemented.
        # include_cog_dist_parameter are for I term, which accounts for model error. include_cog_dist_model are for disturbances.
        # - include_cog_dist_parameter: Flag to include disturbance on the CoG into the acados model parameters.
        if not hasattr(self, "include_cog_dist_parameter"):
            raise AttributeError(
                "CoG disturbance parameter flag not set. Please set the include_cog_dist_parameter attribute in the child class.")

        # These two variables are only for impedance control
        # - include_cog_dist_model: Flag to include disturbance on the CoG into the acados model states.
        if not hasattr(self, "include_cog_dist_model"):
            self.include_cog_dist_model = False
        # - include_impedance: Flag to include virtual mass and inertia to calculate impedance cost. Doesn't add any functionality for the model.
        if not hasattr(self, "include_impedance"):
            self.include_impedance = False

        self.acados_init_p = None  # initial value for parameters in acados. Mainly for physical parameters.

        # Call RecedingHorizon constructor coming as NMPC method
        super().__init__("impedance", overwrite)

        # Create Reference Generator object
        self._reference_generator = self._create_reference_generator()

    def get_reference_generator(self) -> HydrusXiReferenceGenerator:
        return self._reference_generator

    def create_acados_model(self) -> AcadosModel:
        """
        Define generic state-space, acados model parameters, control inputs for kinematics of a quadrotor.
        Calculate transformation matrix from robot's architecture to compute internal wrench.
        Assemble acados model based on given cost function.
        """
        if self.include_servo_derivative and not self.include_servo_model: raise ValueError(
            "Servo derivative can only work with servo angle defined as state through the 'include_servo_model' flag.")

        # Standard state-space (Note: store in self to access for cost function in child controller class)
        self.x = ca.SX.sym("x")  # Position
        self.y = ca.SX.sym("y")
        self.z = ca.SX.sym("z");
        self.p = ca.vertcat(self.x, self.y, self.z)
        self.vx = ca.SX.sym("vx")  # Linear velocity
        self.vy = ca.SX.sym("vy")
        self.vz = ca.SX.sym("vz");
        self.v = ca.vertcat(self.vx, self.vy, self.vz)
        self.qw = ca.SX.sym("qw")  # Quaternions
        self.qx = ca.SX.sym("qx")
        self.qy = ca.SX.sym("qy")
        self.qz = ca.SX.sym("qz");
        self.q = ca.vertcat(self.qw, self.qx, self.qy, self.qz)
        self.wx = ca.SX.sym("wx")  # Angular velocity
        self.wy = ca.SX.sym("wy")
        self.wz = ca.SX.sym("wz");
        self.w = ca.vertcat(self.wx, self.wy, self.wz)
        self.states = ca.vertcat(self.p, self.v, self.q, self.w)

        # - Extend state-space by dynamics of servo angles (actual)
        # Differentiate between actual angles and control angles
        # Note: If servo angle is not used as control input the model for omnidirectional Quadrotor
        # has been observed to be unstable (see https://arxiv.org/abs/2405.09871).
        if self.include_servo_model:
            self.j1s = ca.SX.sym("j1s")
            self.j2s = ca.SX.sym("j2s")
            self.j3s = ca.SX.sym("j3s")
            self.j_s = ca.vertcat(self.j1s, self.j2s, self.j3s)
            self.states = ca.vertcat(self.states, self.j_s)
        
        if self.include_servo_dynamic:
            self.w1s = ca.SX.sym("w1s")
            self.w2s = ca.SX.sym("w2s")
            self.w3s = ca.SX.sym("w3s")
            self.w_s = ca.vertcat(self.w1s, self.w2s, self.w3s)
            self.states = ca.vertcat(self.states, self.w_s)

        # - Extend state-space by dynamics of rotor (actual)
        # Differentiate between actual thrust and control thrust
        if self.include_thrust_model:
            self.ft1s = ca.SX.sym("ft1s")
            self.ft2s = ca.SX.sym("ft2s")
            self.ft3s = ca.SX.sym("ft3s")
            self.ft4s = ca.SX.sym("ft4s")
            self.ft_s = ca.vertcat(self.ft1s, self.ft2s, self.ft3s, self.ft4s)
            self.states = ca.vertcat(self.states, self.ft_s)

        # - Extend state-space by disturbance on CoG (actual)
        # Differentiate between actual disturbance set as state and set as parameter
        if self.include_cog_dist_model:
            # Force disturbance applied to CoG in World frame
            self.fds_w = ca.SX.sym("fds_w", 3)
            # Torque disturbance applied to CoG in Body frame
            self.tau_ds_b = ca.SX.sym("tau_ds_b", 3)

            self.states = ca.vertcat(self.states, self.fds_w, self.tau_ds_b)
        else:
            self.fds_w = ca.vertcat(0.0, 0.0, 0.0)
            self.tau_ds_b = ca.vertcat(0.0, 0.0, 0.0)

        if self.include_end_effector_dist_model:
            self.fde_w = ca.SX.sym("fde_w", 3)
            self.states = ca.vertcat(self.states, self.fde_w)
        else:
            self.fde_w = ca.vertcat(0.0, 0.0, 0.0)

        # Control inputs
        # - Forces from thrust at each rotor
        self.ft1c = ca.SX.sym("ft1c")
        self.ft2c = ca.SX.sym("ft2c")
        self.ft3c = ca.SX.sym("ft3c")
        self.ft4c = ca.SX.sym("ft4c")
        self.ft_c = ca.vertcat(self.ft1c, self.ft2c, self.ft3c, self.ft4c)
        controls = ca.vertcat(self.ft1c, self.ft2c, self.ft3c, self.ft4c)
        # - Servo angle for tiltable rotors (actuated)
        if self.include_servo_model:       
            self.j1c = ca.SX.sym("j1c")# Joint angles
            self.j2c = ca.SX.sym("j2c")
            self.j3c = ca.SX.sym("j3c")
            # Either use the time-derivative of the servo angle as control input directly
            self.j_c = ca.vertcat(self.j1c, self.j2c, self.j3c)
            #controls = ca.vertcat(controls, self.j_c)


        # Model parameters
        self.qwr = ca.SX.sym("qwr")  # Reference for quaternions
        self.qxr = ca.SX.sym("qxr")
        self.qyr = ca.SX.sym("qyr")
        self.qzr = ca.SX.sym("qzr")
        parameters = ca.vertcat(self.qwr, self.qxr, self.qyr, self.qzr)

        # added on 2025-3-28: make physical parameters available in the model
        self.l = ca.SX.sym("l")
        self.m1 = ca.SX.sym("m1")
        self.m2 = ca.SX.sym("m2")
        self.m3 = ca.SX.sym("m3")
        self.m4 = ca.SX.sym("m4")
        self.m = ca.SX.sym("m")
        mass = ca.vertcat(self.m1, self.m2, self.m3, self.m4)
        self.gravity = ca.SX.sym("gravity")

        I1xx = ca.SX.sym("I1xx")
        I1yy = ca.SX.sym("I1yy")
        I1zz = ca.SX.sym("I1zz")
        I2xx = ca.SX.sym("I2xx")
        I2yy = ca.SX.sym("I2yy")
        I2zz = ca.SX.sym("I2zz")
        I3xx = ca.SX.sym("I3xx")
        I3yy = ca.SX.sym("I3yy")
        I3zz = ca.SX.sym("I3zz")
        I4xx = ca.SX.sym("I4xx")
        I4yy = ca.SX.sym("I4yy")
        I4zz = ca.SX.sym("I4zz")

        # m11 = I2zz+I3zz+I4zz+self.m*self.l**2*(15/4+3*ca.cos(self.j2s)+ca.cos(self.j3s)+ca.cos(self.j2s+self.j3s))
        # m12 = I3zz+I4zz+self.m*self.l**2*(3/2+3*ca.cos(self.j2s)/2+ca.cos(self.j3s)+ca.cos(self.j2s+self.j3s)/2)
        # m13 = I4zz+self.m*self.l**2*(1/4+ca.cos(self.j3s)/2+ca.cos(self.j2s+self.j3s)/2)
        # m22 = I3zz+I4zz+self.m*self.l**2*(3/2+ca.cos(self.j3s))
        # m23 = I4zz+self.m*self.l**2*(1/4+ca.cos(self.j3s)/2)
        # m33 = I4zz+self.m*self.l**2*(1/4)

        # self.II = ca.vertcat(
        #         ca.horzcat(m11, m12, m13), ca.horzcat(m12, m22, m23), ca.horzcat(m13, m23, m33)
        #     ) 
        # self.II_inv = ca.inv(self.II)

      
        self.kq_d_kt = ca.SX.sym("kq_d_kt")

        self.dr1 = ca.SX.sym("dr1")
        self.dr2 = ca.SX.sym("dr2")
        self.dr3 = ca.SX.sym("dr3")
        self.dr4 = ca.SX.sym("dr4")

        t_rotor = ca.SX.sym("t_rotor")
        t_servo = ca.SX.sym("t_servo")

        phy_params = ca.vertcat(self.l, mass, self.m, self.gravity, I1xx, I1yy, I1zz, I2xx, I2yy, I2zz, I3xx, I3yy, I3zz, I4xx, I4yy, I4zz, self.kq_d_kt,
                                self.dr1, self.dr2, self.dr3, self.dr4, t_rotor, t_servo)
        parameters = ca.vertcat(parameters, phy_params)

        # - Extend model parameters by CoG disturbance
        if self.include_cog_dist_parameter:
            # Force disturbance applied to CoG in World frame
            self.fdp_w = ca.SX.sym("fdp_w", 3)
            # Torque disturbance applied to CoG in Body frame
            self.tau_dp_b = ca.SX.sym("tau_dp_b", 3)

            parameters = ca.vertcat(parameters, self.fdp_w, self.tau_dp_b)
        else:
            self.fdp_w = ca.vertcat(0.0, 0.0, 0.0)
            self.tau_dp_b = ca.vertcat(0.0, 0.0, 0.0)

        # - Extend model parameters by virtual mass and inertia for impedance cost function
        if self.include_impedance:
            if not self.include_cog_dist_model or not self.include_cog_dist_parameter: raise ValueError(
                "Impedance cost can only be calculated if disturbance flags are activated.")

            self.mpx = ca.SX.sym("mpx")  # Virtual mass (p = position)
            self.mpy = ca.SX.sym("mpy")
            self.mpz = ca.SX.sym("mpz")
            mp = ca.vertcat(self.mpx, self.mpy, self.mpz)

            self.mqx = ca.SX.sym("mqx")  # Virtual inertia (q = quaternion)
            self.mqy = ca.SX.sym("mqy")
            self.mqz = ca.SX.sym("mqz")
            mq = ca.vertcat(self.mqx, self.mqy, self.mqz)

            parameters = ca.vertcat(parameters, mp, mq)
        parameters = ca.vertcat(parameters, self.j_c)
        # Transformation matrices between coordinate systems World, Body, End-of-arm, Rotor using quaternions
        # - Root to CoG
        rot_r2c = ca.vertcat(
                ca.horzcat(ca.cos(self.j1s), -ca.sin(self.j1s), 0), ca.horzcat(ca.sin(self.j1s), ca.cos(self.j1s), 0), ca.horzcat(0, 0, 1)
            ) 
        rot_1_2 = ca.vertcat(
                ca.horzcat(ca.cos(self.j1s), -ca.sin(self.j1s), 0), ca.horzcat(ca.sin(self.j1s), ca.cos(self.j1s), 0), ca.horzcat(0, 0, 1)
            )
        rot_2_3 = ca.vertcat(
                ca.horzcat(ca.cos(self.j2s), -ca.sin(self.j2s), 0), ca.horzcat(ca.sin(self.j2s), ca.cos(self.j2s), 0), ca.horzcat(0, 0, 1)
            )
        rot_3_4 = ca.vertcat(
                ca.horzcat(ca.cos(self.j3s), -ca.sin(self.j3s), 0), ca.horzcat(ca.sin(self.j3s), ca.cos(self.j3s), 0), ca.horzcat(0, 0, 1)
            )
        self.l_vec = ca.vertcat(self.l, 0, 0)
        l_r1 = self.l_vec/2
        l_r2 = self.l_vec + ca.mtimes(rot_1_2, self.l_vec/2)
        l_r3 = self.l_vec + ca.mtimes(rot_1_2, self.l_vec) + ca.mtimes(ca.mtimes(rot_2_3, rot_1_2), self.l_vec/2)
        l_r4 = self.l_vec + ca.mtimes(rot_1_2, self.l_vec) + ca.mtimes(ca.mtimes(rot_2_3, rot_1_2), self.l_vec) + ca.mtimes(ca.mtimes(rot_3_4, ca.mtimes(rot_2_3, rot_1_2)), self.l_vec/2)

        tran_r2c = (self.m1*l_r1 + self.m2*l_r2 + self.m3*l_r3 + self.m4*l_r4) / self.m

        self.rot_c_1 =  ca.vertcat(
                ca.horzcat(ca.cos(self.j1s), ca.sin(self.j1s), 0), ca.horzcat(-ca.sin(self.j1s), ca.cos(self.j1s), 0), ca.horzcat(0, 0, 1)
            )

        self.rot_c_2 =  ca.vertcat(
                ca.horzcat(1, 0, 0), ca.horzcat(0, 1, 0), ca.horzcat(0, 0, 1)
            )

        self.rot_c_3 = ca.vertcat(
                ca.horzcat(ca.cos(self.j2s), -ca.sin(self.j2s), 0), ca.horzcat(ca.sin(self.j2s), ca.cos(self.j2s), 0), ca.horzcat(0, 0, 1)
            )

        self.rot_c_4 = ca.vertcat(
                ca.horzcat(ca.cos(self.j2s+self.j3s), -ca.sin(self.j2s+self.j3s), 0), ca.horzcat(ca.sin(self.j2s+self.j3s), ca.cos(self.j2s+self.j3s), 0), ca.horzcat(0, 0, 1)
            )

        self.tran_c_1 = ca.mtimes(self.rot_c_1,l_r1-tran_r2c)

        self.tran_c_2 = ca.mtimes(self.rot_c_1,l_r2-tran_r2c)

        self.tran_c_3 = ca.mtimes(self.rot_c_1,l_r3-tran_r2c)

        self.tran_c_4 = ca.mtimes(self.rot_c_1,l_r4-tran_r2c)

        self.tran_c_e = self.tran_c_4 + ca.mtimes(self.rot_c_4, self.l_vec/2)

        # - World to Body
        row_1 = ca.horzcat(
            ca.SX(1 - 2 * self.qy ** 2 - 2 * self.qz ** 2), ca.SX(2 * self.qx * self.qy - 2 * self.qw * self.qz),
            ca.SX(2 * self.qx * self.qz + 2 * self.qw * self.qy)
        )
        row_2 = ca.horzcat(
            ca.SX(2 * self.qx * self.qy + 2 * self.qw * self.qz), ca.SX(1 - 2 * self.qx ** 2 - 2 * self.qz ** 2),
            ca.SX(2 * self.qy * self.qz - 2 * self.qw * self.qx)
        )
        row_3 = ca.horzcat(
            ca.SX(2 * self.qx * self.qz - 2 * self.qw * self.qy), ca.SX(2 * self.qy * self.qz + 2 * self.qw * self.qx),
            ca.SX(1 - 2 * self.qx ** 2 - 2 * self.qy ** 2)
        )
        self.rot_wb = ca.vertcat(row_1, row_2, row_3)
        # - Calculate Jacobians
        J_v1 = ca.vertcat(
                ca.horzcat(-self.l/2*ca.sin(self.j1s), 0, 0), 
                ca.horzcat(self.l/2*ca.cos(self.j1s), 0, 0), 
                ca.horzcat(0, 0, 0)) 
        
        J_v2 = ca.vertcat(
                ca.horzcat(-self.l*ca.sin(self.j1s)-self.l/2*ca.sin(self.j1s+self.j2s), -self.l/2*ca.sin(self.j1s+self.j2s), 0), 
                ca.horzcat(self.l*ca.cos(self.j1s)+self.l/2*ca.cos(self.j1s+self.j2s), self.l/2*ca.cos(self.j1s+self.j2s), 0), 
                ca.horzcat(0, 0, 0)) 

        J_v3 = ca.vertcat(
                ca.horzcat(-self.l*ca.sin(self.j1s)-self.l*ca.sin(self.j1s+self.j2s)-self.l/2*ca.sin(self.j1s+self.j2s+self.j3s), -self.l*ca.sin(self.j1s+self.j2s)-self.l/2*ca.sin(self.j1s+self.j2s+self.j3s), -self.l/2*ca.sin(self.j1s+self.j2s+self.j3s)), 
                ca.horzcat(self.l*ca.cos(self.j1s)+self.l*ca.cos(self.j1s+self.j2s)+self.l/2*ca.cos(self.j1s+self.j2s+self.j3s), self.l*ca.cos(self.j1s+self.j2s)+self.l/2*ca.cos(self.j1s+self.j2s+self.j3s), self.l/2*ca.cos(self.j1s+self.j2s+self.j3s)), 
                ca.horzcat(0, 0, 0)) 

        J_ve = ca.vertcat(
                ca.horzcat(-self.l*ca.sin(self.j1s)-self.l*ca.sin(self.j1s+self.j2s)-self.l*ca.sin(self.j1s+self.j2s+self.j3s), -self.l*ca.sin(self.j1s+self.j2s)-self.l*ca.sin(self.j1s+self.j2s+self.j3s), -self.l*ca.sin(self.j1s+self.j2s+self.j3s)), 
                ca.horzcat(self.l*ca.cos(self.j1s)+self.l*ca.cos(self.j1s+self.j2s)+self.l*ca.cos(self.j1s+self.j2s+self.j3s), self.l*ca.cos(self.j1s+self.j2s)+self.l*ca.cos(self.j1s+self.j2s+self.j3s), self.l*ca.cos(self.j1s+self.j2s+self.j3s)), 
                ca.horzcat(0, 0, 0)) 

        self.J_ve_world = ca.mtimes(self.rot_wb, ca.mtimes(rot_r2c.T, J_ve))

        self.J_ve_cog = ca.mtimes(rot_r2c.T, J_ve)

        J_w1 = ca.vertcat(
                ca.horzcat(0, 0, 0), 
                ca.horzcat(0, 0, 0), 
                ca.horzcat(1, 0, 0)) 
        
        J_w2 = ca.vertcat(
                ca.horzcat(0, 0, 0), 
                ca.horzcat(0, 0, 0), 
                ca.horzcat(1, 1, 0))

        J_w3 = ca.vertcat(
                ca.horzcat(0, 0, 0), 
                ca.horzcat(0, 0, 0), 
                ca.horzcat(1, 1, 1))

        # End_effctor force acting on every joint
        tao_j = ca.mtimes(self.J_ve_world,self.fde_w)

        # - Body to End-of-arm

        # - End-of-arm to Rotor
        # Take tilt rotation with angle alpha (a) of R frame to E frame into account
        self.tilt_angle = np.pi * 20 / 180
        if self.tilt:
            # If servo dynamics are modeled, use angle state.
            # Else use angle control which is then assumed to be equal to the angle state at all times.
            if self.include_servo_model:
                self.g1s = ca.SX.sym("g1s")# Joint angles
                self.g2s = ca.SX.sym("g2s")
                self.g3s = ca.SX.sym("g3s")
                self.g4s = ca.SX.sym("g4s")
            # Either use the time-derivative of the servo angle as control input directly
                self.g_s = ca.vertcat(self.g1s, self.g2s, self.g3s, self.g4s)
                self.states = ca.vertcat(self.states, self.g_s)

                self.g1c = ca.SX.sym("g1c")# Joint angles
                self.g2c = ca.SX.sym("g2c")
                self.g3c = ca.SX.sym("g3c")
                self.g4c = ca.SX.sym("g4c")
            # Either use the time-derivative of the servo angle as control input directly
                self.g_c = ca.vertcat(self.g1c, self.g2c, self.g3c, self.g4c)
                controls = ca.vertcat(controls, self.g_c)
            # clockwisely rotate tilt_angle
            rot_tilt = ca.vertcat(
                ca.horzcat(ca.cos(self.tilt_angle), 0, -ca.sin(self.tilt_angle)),
                ca.horzcat(0, 1, 0),
                ca.horzcat(ca.sin(self.tilt_angle), 0, ca.cos(self.tilt_angle))
            )
            rot_y1 = ca.vertcat(
                ca.horzcat(ca.cos(self.g1s), -ca.sin(self.g1s), 0), ca.horzcat(ca.sin(self.g1s), ca.cos(self.g1s), 0), ca.horzcat(0, 0, 1)
            )
            rot_y2 = ca.vertcat(
                ca.horzcat(ca.cos(self.g2s), -ca.sin(self.g2s), 0), ca.horzcat(ca.sin(self.g2s), ca.cos(self.g2s), 0), ca.horzcat(0, 0, 1)
            )
            rot_y3 = ca.vertcat(
                ca.horzcat(ca.cos(self.g3s), -ca.sin(self.g3s), 0), ca.horzcat(ca.sin(self.g3s), ca.cos(self.g3s), 0), ca.horzcat(0, 0, 1)
            )
            rot_y4 = ca.vertcat(
                ca.horzcat(ca.cos(self.g4s), -ca.sin(self.g4s), 0), ca.horzcat(ca.sin(self.g4s), ca.cos(self.g4s), 0), ca.horzcat(0, 0, 1)
            )
            self.rot_e1r1 = ca.mtimes(rot_y1, rot_tilt)
            self.rot_e2r2 = ca.mtimes(rot_y2, rot_tilt)
            self.rot_e3r3 = ca.mtimes(rot_y3, rot_tilt)
            self.rot_e4r4 = ca.mtimes(rot_y4, rot_tilt)

        else:
            self.rot_e1r1 = ca.SX.eye(3);
            self.rot_e2r2 = ca.SX.eye(3);
            self.rot_e3r3 = ca.SX.eye(3);
            self.rot_e4r4 = ca.SX.eye(3)

        # Wrench in Rotor frame
        # If rotor dynamics are modeled, explicitly use thrust state as force.
        # Else use thrust control which is then assumed to be equal to the thrust state at all times.
        if self.include_thrust_model:
            ft1 = self.ft1s;
            ft2 = self.ft2s;
            ft3 = self.ft3s;
            ft4 = self.ft4s
        else:
            ft1 = self.ft1c;
            ft2 = self.ft2c;
            ft3 = self.ft3c;
            ft4 = self.ft4c

        self.ft_r1 = ca.vertcat(0, 0, ft1)
        self.ft_r2 = ca.vertcat(0, 0, ft2)
        self.ft_r3 = ca.vertcat(0, 0, ft3)
        self.ft_r4 = ca.vertcat(0, 0, ft4)

        tau_r1 = ca.vertcat(0, 0, -self.dr1 * ft1 * self.kq_d_kt)
        tau_r2 = ca.vertcat(0, 0, -self.dr2 * ft2 * self.kq_d_kt)
        tau_r3 = ca.vertcat(0, 0, -self.dr3 * ft3 * self.kq_d_kt)
        tau_r4 = ca.vertcat(0, 0, -self.dr4 * ft4 * self.kq_d_kt)

        # Wrench in Body frame
        self.fu_b = (
                ca.mtimes(self.rot_c_1, ca.mtimes(self.rot_e1r1, self.ft_r1))
                + ca.mtimes(self.rot_c_2, ca.mtimes(self.rot_e2r2, self.ft_r2))
                + ca.mtimes(self.rot_c_3, ca.mtimes(self.rot_e3r3, self.ft_r3))
                + ca.mtimes(self.rot_c_4, ca.mtimes(self.rot_e4r4, self.ft_r4))
        )

        self.tau_u_b = (
                ca.mtimes(self.rot_c_1, ca.mtimes(self.rot_e1r1, tau_r1))
                + ca.mtimes(self.rot_c_2, ca.mtimes(self.rot_e2r2, tau_r2))
                + ca.mtimes(self.rot_c_3, ca.mtimes(self.rot_e3r3, tau_r3))
                + ca.mtimes(self.rot_c_4, ca.mtimes(self.rot_e4r4, tau_r4))
                + ca.cross(self.tran_c_1, ca.mtimes(self.rot_c_1, ca.mtimes(self.rot_e1r1, self.ft_r1)))
                + ca.cross(self.tran_c_2, ca.mtimes(self.rot_c_2, ca.mtimes(self.rot_e2r2, self.ft_r2)))
                + ca.cross(self.tran_c_3, ca.mtimes(self.rot_c_3, ca.mtimes(self.rot_e3r3, self.ft_r3)))
                + ca.cross(self.tran_c_4, ca.mtimes(self.rot_c_4, ca.mtimes(self.rot_e4r4, self.ft_r4)))
        )


        # Compute Inertia
        I1 = ca.diag(ca.vertcat(I1xx, I1yy, I1zz))
        I2 = ca.diag(ca.vertcat(I2xx, I2yy, I2zz))
        I3 = ca.diag(ca.vertcat(I3xx, I3yy, I3zz))
        I4 = ca.diag(ca.vertcat(I4xx, I4yy, I4zz))
        self.I = ca.mtimes(self.rot_c_1, ca.mtimes(I1, self.rot_c_1.T)) + ca.mtimes(self.rot_c_2, ca.mtimes(I2, self.rot_c_2.T)) + \
            ca.mtimes(self.rot_c_3, ca.mtimes(I3, self.rot_c_3.T)) + ca.mtimes(self.rot_c_4, ca.mtimes(I4, self.rot_c_4.T)) + \
            ca.mtimes(ca.SX.eye(3), ca.mtimes(self.tran_c_1.T, self.tran_c_1)) - ca.mtimes(self.tran_c_1, self.tran_c_1.T) + \
            ca.mtimes(ca.SX.eye(3), ca.mtimes(self.tran_c_2.T, self.tran_c_2)) - ca.mtimes(self.tran_c_2, self.tran_c_2.T) + \
            ca.mtimes(ca.SX.eye(3), ca.mtimes(self.tran_c_3.T, self.tran_c_3)) - ca.mtimes(self.tran_c_3, self.tran_c_3.T) + \
            ca.mtimes(ca.SX.eye(3), ca.mtimes(self.tran_c_4.T, self.tran_c_4)) - ca.mtimes(self.tran_c_4, self.tran_c_4.T)

        I_inv = ca.inv(self.I)
        g_w = ca.vertcat(0, 0, -self.gravity)  # World frame

        tau_de_b = ca.cross(self.tran_c_e, ca.mtimes(self.rot_wb.T, self.fde_w))  
        
        # Dynamic model (Time-derivative of states)
        ds = ca.vertcat(
            self.v,
            (ca.mtimes(self.rot_wb, self.fu_b) + self.fds_w + self.fdp_w + self.fde_w) / self.m + g_w,
            (-self.wx * self.qx - self.wy * self.qy - self.wz * self.qz) / 2,
            (self.wx * self.qw + self.wz * self.qy - self.wy * self.qz) / 2,
            (self.wy * self.qw - self.wz * self.qx + self.wx * self.qz) / 2,
            (self.wz * self.qw + self.wy * self.qx - self.wx * self.qy) / 2,
            ca.mtimes(I_inv, (-ca.cross(self.w, ca.mtimes(self.I, self.w)) + self.tau_u_b + self.tau_ds_b + self.tau_dp_b + tau_de_b)),
        )

#         f = ca.Function("f", [self.j1s, self.j2s, self.j3s, self.qx, self.qy,self.qz,self.qw,l, m1, m2, m3, m4, m, I1xx, I1yy, I1zz, I2xx, I2yy, I2zz, I3xx, I3yy, I3zz,I4xx, I4yy, I4zz,self.ft1s, self.ft2s,self.ft3s,self.ft4s,gravity], [ds])
#         result = f(pi/2, pi/2, pi/2, 0,0,0,1,0.6, 0.1, 0.1, 0.1, 0.1, 0.4, 0.001, 0.008, 0.008,0.001, 0.008, 0.008 ,0.001, 0.008, 0.008,0.001, 0.008, 0.008, 1, 1, 1, 1,9.81)
# # Evaluate the expression
#         print(result)  
#         model = AcadosModel()

#         return model

        # - Extend model by servo first-order dynamics
        # Assumption if not included: a_c = a_s
        # Either use continuous time-derivate as control variable
        if self.include_servo_derivative:
            ds = ca.vertcat(ds,
                            self.ad_c
                            )
        # Or use numerical differentation
        # Kp_servo = ca.diag(ca.vertcat(550.0, 450.0, 500.0))
        # Kd_servo = ca.diag(ca.vertcat(100.0, 80.0, 15.0))
        # if self.include_servo_model:
        #     if not self.include_servo_dynamic:
        #         ds = ca.vertcat(ds,
        #                     (self.j_c - self.j_s) / t_servo  # Time constant of servo motor
        #                     )
        #     else:
        #         ds = ca.vertcat(ds,
        #                     self.w_s,
        #                     ca.mtimes(self.II_inv, ca.mtimes(Kp_servo,(self.j_c-self.j_s))-ca.mtimes(Kd_servo, self.w_s)+tao_j)  # Time constant of servo motor,
        #                       # Time constant of servo motor
        #                     )
        if self.include_servo_model:
             ds = ca.vertcat(ds,
                             (self.j_c - self.j_s) / t_servo, # Time constant of servo motor
                            )
        if self.tilt:
             ds = ca.vertcat(ds,
                             (self.g_c - self.g_s) / t_servo, # Time constant of servo motor
                            )

        # - Extend model by thrust first-order dynamics
        # Assumption if not included: f_tc = f_ts
        if self.include_thrust_model:
            ds = ca.vertcat(ds,
                            (self.ft_c - self.ft_s) / t_rotor  # Time constant of rotor
                            )

        # - Extend model by disturbances simply to match state dimensions
        if self.include_cog_dist_model:
            ds = ca.vertcat(ds,
                            ca.vertcat(0.0, 0.0, 0.0),
                            ca.vertcat(0.0, 0.0, 0.0),
                            )
        if self.include_end_effector_dist_model:
            ds = ca.vertcat(ds,
                            ca.vertcat(0.0, 0.0, 0.0),
                            )


        # Assemble acados function
        f = ca.Function("f", [self.states, controls], [ds], ["state", "control_input"], ["ds"], {"allow_free": True})

        # Implicit dynamics
        # Note: Used only mainly because of acados template
        x_dot = ca.SX.sym("x_dot", self.states.size())  # Combined state vector
        f_impl = x_dot - f(self.states, controls)

        # Get terms of cost function
        if self.include_impedance:
            # Compute linear acceleration (in World frame) and angular acceleration (in Body frame) for impedance cost
            # TODO clarify note and fix if necessary        
            # Note that this part should be f_d_i and no f_d_i_para, since the impedance should not respond to the I Term force.
            lin_acc_w = (ca.mtimes(self.rot_wb, fu_b) + self.fds_w + self.fdp_w) / mass + g_w
            ang_acc_b = ca.mtimes(I_inv,
                                  (-ca.cross(self.w, ca.mtimes(I, self.w)) + tau_u_b + self.tau_ds_b + self.tau_dp_b))
            state_y, state_y_e, control_y = self.get_cost_function(lin_acc_w=lin_acc_w, ang_acc_b=ang_acc_b)
        else:
            state_y, state_y_e, control_y = self.get_cost_function()

        # Assemble acados model
        model = AcadosModel()
        model.name = self.model_name
        model.f_expl_expr = f(self.states, controls)  # CasADi expression for the explicit dynamics
        model.f_impl_expr = f_impl  # CasADi expression for the implicit dynamics
        model.x = self.states
        model.xdot = x_dot
        model.u = controls
        model.p = parameters
        model.cost_y_expr = ca.vertcat(state_y, control_y)  # NONLINEAR_LS
        model.cost_y_expr_e = state_y_e

        self.f_ref = ca.vertcat(8.32, 8.32, 8.32, 8.32)
        self.g_ref = ca.vertcat(np.pi, 0.0, np.pi, 0.0)


        
        return model

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def get_cost_function(self, lin_acc_w=None, ang_acc_b=None):
        pass

    def compute_trajectory(self, params, x_now, u_cmd, target_xyz, target_rpy, target_vxyz=np.array([[0.0, 0.0, 0.0]]).T, target_wrpy=np.array([[0.0, 0.0, 0.0]]).T, target_joint_angles=np.array([[np.pi/2, np.pi/2, np.pi/2]]).T, target_gimbal_angles=np.array([[np.pi, 0.0, np.pi, 0.0]]).T):
        """
        Convert current target pose to a reference trajectory over the entire horizon.
        Compute target quaternions and control reference from a target rotation and then 
        get assembled reference trajectories from controller file.

        :param target_xyz: Target position
        :param target_rpy: Target orientation (roll, pitch, yaw)
        :return xr: Reference for the state x
        :return ur: Reference for the input u
        """
        roll = target_rpy[0]
        pitch = target_rpy[1]
        yaw = target_rpy[2]

        q = tf.quaternion_from_euler(roll, pitch, yaw, axes="sxyz")
        target_qwxyz = np.array([[q[3], q[0], q[1], q[2]]]).T

        # Convert [0,0,gravity] to Body frame
        q_inv = tf.quaternion_inverse(q)
        rot = tf.quaternion_matrix(q_inv)
        fg_w = np.array([0, 0, self.m * self.gravity, 0])    # World frame
        fg_b = rot @ fg_w                                       # Body frame
        target_force = np.array([0, 0, self.m * self.gravity]).T
        target_torque = np.array([[0.0, 0.0, 0.0]]).T
        model = super().get_acados_model()
         
        f_opt = ca.SX.sym("f_opt", 4)  
        g_opt = ca.SX.sym("g_opt", 4) 
        opt = ca.vertcat(f_opt, g_opt)
        wrench = ca.vertcat(ca.mtimes(self.rot_wb, self.fu_b), self.tau_u_b)
 
        least_squares = ca.mtimes((ca.mtimes(self.rot_wb, self.fu_b) - target_force).T, (ca.mtimes(self.rot_wb, self.fu_b) - target_force))+\
                        ca.mtimes((self.tau_u_b - target_torque).T, (self.tau_u_b - target_torque)) + \
                        10*ca.mtimes((self.ft_c - self.f_ref).T, (self.ft_c - self.f_ref)) + \
                        10*ca.mtimes((self.g_s - self.g_ref).T, (self.g_s - self.g_ref))
        func = ca.Function('wrench_fun', [model.p[4:30], model.x, model.u], [least_squares])
        
     

        x_fixed = ca.vertcat(x_now[0:16], g_opt)
        u_fixed = ca.vertcat(f_opt, u_cmd[4:8])
        func_new = func(model.p[4:30], x_fixed, u_fixed)



        nlp = {'x': opt, 'p': model.p[4:30], 'f': func_new}
        opts = {
        'ipopt.print_level': 0,    
        'print_time': 0,            
        'ipopt.sb': 'yes'          
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        lbx = [0, 0, 0, 0, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]
        ubx = [30, 30, 30, 30, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]
        sol = solver(lbx=lbx, ubx=ubx, p=params)
      
        result = np.squeeze(np.array(sol['x'].full()))
        #print("x_opt:", result)
        #print("cost:", np.squeeze(np.array(sol['f'].full())) )
        ft_ref = result[0:4]
        target_gimbal_angles = result[4:8] 
        self.f_ref = ca.vertcat(ft_ref[0], ft_ref[1], ft_ref[2], ft_ref[3])
        self.g_ref = ca.vertcat(target_gimbal_angles[0], target_gimbal_angles[1], target_gimbal_angles[2], target_gimbal_angles[3])
       
        # A faster method if alloc_mat is dynamic:  x, _, _, _ = np.linalg.lstsq(alloc_mat, target_wrench, rcond=None)
        # target_force = self.alloc_mat_pinv @ target_wrench
        
        # # Compute reference values for thrust
        # # Set either state or control input based on model properties, i.e., based on include flags
        # ft1_ref = np.sqrt(target_force[0, 0] ** 2 + target_force[1, 0] ** 2)
        # ft2_ref = np.sqrt(target_force[2, 0] ** 2 + target_force[3, 0] ** 2)
        # ft3_ref = np.sqrt(target_force[4, 0] ** 2 + target_force[5, 0] ** 2)
        # ft4_ref = np.sqrt(target_force[6, 0] ** 2 + target_force[7, 0] ** 2)
        # ft_ref = [ft1_ref, ft2_ref, ft3_ref, ft4_ref]

        # # Compute reference values for servo angles
        # # Set either state or control input based on model properties, i.e., based on include flags
        # a1_ref = np.arctan2(target_force[0, 0], target_force[1, 0])
        # a2_ref = np.arctan2(target_force[2, 0], target_force[3, 0])
        # a3_ref = np.arctan2(target_force[4, 0], target_force[5, 0])
        # a4_ref = np.arctan2(target_force[6, 0], target_force[7, 0])
        # a_ref = [a1_ref, a2_ref, a3_ref, a4_ref]
            
        # Assemble reference trajectories in controller file since their definition is 
        # closely related to the cost function
        # xr, ur = self.nmpc.get_reference(target_xyz, target_qwxyz, ft_ref, a_ref)
       
        xr, ur = self.get_reference(target_xyz, target_qwxyz, target_vxyz, target_wrpy, target_joint_angles, target_gimbal_angles, ft_ref)
        return xr, ur

    def cal_wrench(self, params, x_now, u_cmd, wrench_tgt):
        model = super().get_acados_model()
        g_opt = ca.SX.sym("g_opt", 4)  
        wrench = ca.vertcat(ca.mtimes(self.rot_wb, self.fu_b), self.tau_u_b)
        least_squares = ca.mtimes((wrench - wrench_tgt).T, (wrench - wrench_tgt))
        func = ca.Function('wrench_fun', [model.p[4:30], model.x, model.u], [least_squares])
        
        x_fixed = ca.vertcat(x_now[0:16], g_opt)
        func_new = func(model.p[4:30], x_fixed, u_cmd)

        nlp = {'x': g_opt, 'p': model.p[4:30], 'f': func_new}
        opts = {
        'ipopt.print_level': 0,    
        'print_time': 0,            
        'ipopt.sb': 'yes'          
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(lbg=0, ubg=0, p=params)
        return np.squeeze(np.array(sol['x'].full()))

    def get_alloc_matrix(self, params, x_now):
        model = super().get_acados_model()
        fz = ca.vertcat(0, 0, 1)
        tz1 = ca.vertcat(0, 0, -self.dr1 * self.kq_d_kt)
        tz2 = ca.vertcat(0, 0, -self.dr2 * self.kq_d_kt)
        tz3 = ca.vertcat(0, 0, -self.dr3 * self.kq_d_kt)
        tz4 = ca.vertcat(0, 0, -self.dr4 * self.kq_d_kt)
        fu_b = ca.horzcat(
                ca.vertcat(ca.mtimes(self.rot_c_1, ca.mtimes(self.rot_e1r1, fz))),
                ca.vertcat(ca.mtimes(self.rot_c_2, ca.mtimes(self.rot_e2r2, fz))),
                ca.vertcat(ca.mtimes(self.rot_c_3, ca.mtimes(self.rot_e3r3, fz))),
                ca.vertcat(ca.mtimes(self.rot_c_4, ca.mtimes(self.rot_e4r4, fz))))
        fu_w = ca.mtimes(self.rot_wb, fu_b)
        tau_b = ca.horzcat(
                ca.vertcat(ca.cross(self.tran_c_1, ca.mtimes(self.rot_c_1, ca.mtimes(self.rot_e1r1, fz))) + ca.mtimes(self.rot_c_1, ca.mtimes(self.rot_e1r1, tz1))),
                ca.vertcat(ca.cross(self.tran_c_2, ca.mtimes(self.rot_c_2, ca.mtimes(self.rot_e2r2, fz))) + ca.mtimes(self.rot_c_2, ca.mtimes(self.rot_e2r2, tz2))),
                ca.vertcat(ca.cross(self.tran_c_3, ca.mtimes(self.rot_c_3, ca.mtimes(self.rot_e3r3, fz))) + ca.mtimes(self.rot_c_3, ca.mtimes(self.rot_e3r3, tz3))),
                ca.vertcat(ca.cross(self.tran_c_4, ca.mtimes(self.rot_c_4, ca.mtimes(self.rot_e4r4, fz))) + ca.mtimes(self.rot_c_4, ca.mtimes(self.rot_e4r4, tz4))))   
        alloc_mat = ca.vertcat(fu_w, tau_b)
        alloc_fun = ca.Function('alloc_fun', [model.p[4:30], model.x], [alloc_mat])
        return np.matrix(alloc_fun(params, x_now).full())
    
    def get_command_wrench(self, params, x_now, u_cmd):
        model = super().get_acados_model()
        force_fun = ca.Function('force_fun', [model.p[4:30], model.x, model.u], [ca.mtimes(self.rot_c_1, ca.mtimes(self.rot_e1r1, self.ft_r1))])
        force = np.squeeze(np.array(force_fun(params, x_now, u_cmd).full()))
        torque_fun = ca.Function('torque_fun', [model.p[4:30], model.x, model.u], [ca.mtimes(self.rot_e1r1, self.ft_r1)])
        torque = np.squeeze(np.array(torque_fun(params, x_now, u_cmd).full()))
        #  + ca.cross(self.tran_c_1, ca.mtimes(self.rot_c_1, ca.mtimes(self.rot_e1r1, self.ft_r1)))
        #         + ca.cross(self.tran_c_2, ca.mtimes(self.rot_c_2, ca.mtimes(self.rot_e2r2, self.ft_r2)))
        #         + ca.cross(self.tran_c_3, ca.mtimes(self.rot_c_3, ca.mtimes(self.rot_e3r3, self.ft_r3)))
        #         + ca.cross(self.tran_c_4, ca.mtimes(self.rot_c_4, ca.mtimes(self.rot_e4r4, self.ft_r4)))
        return force, torque
    def get_a_matrix(self, params, x_now):
        model = super().get_acados_model()
        I_matrix_fun = ca.Function('I_matrix_fun', [model.p[4:30], model.x], [self.rot_c_1])
        return np.matrix(I_matrix_fun(params, x_now).full())
    
    def get_c_vector(self, params, x_now):
        model = super().get_acados_model()
        c_vector = ca.cross(self.w, ca.mtimes(self.I, self.w))
        c_vector_fun = ca.Function('c_vector_fun', [model.p[4:30], model.x], [c_vector])
        return np.squeeze(np.array(c_vector_fun(params, x_now).full()))
    
    def get_I_matrix(self, params, x_now):
        model = super().get_acados_model()
        I_matrix_fun = ca.Function('I_matrix_fun', [model.p[4:30], model.x], [self.I])
        return np.matrix(I_matrix_fun(params, x_now).full())

    def get_end_Jacobian(self, params, x_now):
        model = super().get_acados_model()
        jacobian_fun = ca.Function('c_vector_fun', [model.p[4:30], model.x], [self.J_ve_cog])
        return np.matrix(jacobian_fun(params, x_now).full())
    
    def get_comp_of_external_wrench(self, params, x_now):
        model = super().get_acados_model()
        sum_momentum = ca.vertcat(self.m * self.v, ca.mtimes(self.I, self.w))
        N = ca.vertcat(self.m * ca.vertcat(0, 0, self.gravity), ca.cross(self.w, ca.mtimes(self.I, self.w)))
        sum_momentum_fun = ca.Function('sum_momentum_fun', [model.p[4:30], model.x], [sum_momentum])
        N_fun = ca.Function('N_fun', [model.p[4:30], model.x], [N])
        return np.squeeze(np.array(sum_momentum_fun(params, x_now).full())), np.squeeze(np.array(N_fun(params, x_now).full())) 

    def get_end_effector_position(self, params, x_now):
        model = super().get_acados_model()
        pe_cog = self.tran_c_4 + ca.mtimes(self.rot_c_4, self.l_vec/2)
        pe_world = ca.mtimes(self.rot_wb, pe_cog) + self.p
        pe_cog_fun = ca.Function('pe_cog_fun', [model.p[4:30], model.x], [pe_cog])
        pe_world_fun = ca.Function('pe_world_fun', [model.p[4:30], model.x], [pe_world])
        return np.squeeze(np.array(pe_cog_fun(params, x_now).full())), np.squeeze(np.array(pe_world_fun(params, x_now).full())) 

    def inv_kinematics(self, params, x_now, pe_cog_tar):
        model = super().get_acados_model()
        pe_cog = self.tran_c_4 + ca.mtimes(self.rot_c_4, self.l_vec/2)
        pe_cog_fun = ca.Function('pe_cog_fun', [model.p[4:30], model.x], [pe_cog])
        cost = ca.sumsqr(pe_cog - pe_cog_tar)
        nlp = {'x': ca.vertcat(self.j1s, self.j2s, self.j3s),
               'f': cost,
               'p': ca.vertcat(self.l, self.m1, self.m2, self.m3, self.m4, self.m),
               'g': ca.vertcat(self.j1s, self.j2s, self.j3s)}
        opts = {'ipopt': {'print_level': 1}, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x_now[13:16], p=params[0:6], lbg=[-np.pi/2,-np.pi/2,-np.pi/2], ubg=[np.pi/2,np.pi/2,np.pi/2])
        return np.array(sol['x'].full().flatten())
    
    def create_acados_ocp_solver(self) -> AcadosOcpSolver:
        """
        Create generic acados solver for NMPC framework of a quadrotor.
        Generate c code into source folder in aerial_robot_control to be used in workflow.
        """
        # Get OCP object
        ocp = super().get_ocp()

        # Model dimensions
        nx = ocp.model.x.size()[0]
        nu = ocp.model.u.size()[0]
        n_param = ocp.model.p.size()[0]

        # Get weights from parametrization child file
        Q, R = self.get_weights()
        # Cost function options
        # see https://docs.acados.org/python_interface/#acados_template.acados_ocp_cost.AcadosOcpCost for details
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
        ocp.cost.W_e = Q  # Weight matrix at terminal shooting node (N)

        # Set constraints
        # TODO include fixed rotor arch
        # - State box constraints bx
        # -- Index for vx, vy, vz, wx, wy, wz, j1, j2, j3
        ocp.constraints.idxbx = np.array([3, 4, 5, 10, 11, 12, 13, 14, 15])

        # -- Index for a1s, a2s, a3s, a4s
        # if self.tilt and self.include_servo_model:
        #     ocp.constraints.idxbx = np.append(ocp.constraints.idxbx, [13, 14, 15, 16])

        #     # -- Index for ft1s, ft2s, ft3s, ft4s (When included servo AND thrust, add further indices)
        #     if self.include_thrust_model:
        #         ocp.constraints.idxbx = np.append(ocp.constraints.idxbx, [17, 18, 19, 20])

        # # -- Index for ft1s, ft2s, ft3s, ft4s (When only included thrust, use the same indices)
        # elif self.include_thrust_model:
        #     ocp.constraints.idxbx = np.append(ocp.constraints.idxbx, [13, 14, 15, 16])

        # # -- Lower State Bound
        ocp.constraints.lbx = np.array(
            [self.params["v_min"],
             self.params["v_min"],
             self.params["v_min"],
             self.params["w_min"],
             self.params["w_min"],
             self.params["w_min"],
             self.params["j_min"],
             self.params["j_min"],
             self.params["j_min"]])

        # if self.tilt and self.include_servo_model:
        #     ocp.constraints.lbx = np.append(ocp.constraints.lbx,
        #                                     [self.params["a_min"],
        #                                      self.params["a_min"],
        #                                      self.params["a_min"],
        #                                      self.params["a_min"]])

        if self.include_thrust_model:
            ocp.constraints.lbx = np.append(ocp.constraints.lbx,
                                            [self.params["thrust_min"],
                                             self.params["thrust_min"],
                                             self.params["thrust_min"],
                                             self.params["thrust_min"]])

        # -- Upper State Bound
        ocp.constraints.ubx = np.array(
            [self.params["v_max"],
             self.params["v_max"],
             self.params["v_max"],
             self.params["w_max"],
             self.params["w_max"],
             self.params["w_max"],
             self.params["j_max"],
             self.params["j_max"],
             self.params["j_max"]])

        # if self.tilt and self.include_servo_model:
        #     ocp.constraints.ubx = np.append(ocp.constraints.ubx,
        #                                     [self.params["a_max"],
        #                                      self.params["a_max"],
        #                                      self.params["a_max"],
        #                                      self.params["a_max"]])

        if self.include_thrust_model:
            ocp.constraints.ubx = np.append(ocp.constraints.ubx,
                                            [self.params["thrust_max"],
                                             self.params["thrust_max"],
                                             self.params["thrust_max"],
                                             self.params["thrust_max"]])

        # - Terminal state box constraints bx_e
        # -- Index for vx, vy, vz, wx, wy, wz
        ocp.constraints.idxbx_e = np.array([3, 4, 5, 10, 11, 12, 13, 14, 15])

        # -- Index for a1s, a2s, a3s, a4s
        # if self.tilt and self.include_servo_model:
        #     ocp.constraints.idxbx_e = np.append(ocp.constraints.idxbx_e, [13, 14, 15, 16])

        #     # -- Index for ft1s, ft2s, ft3s, ft4s (When included servo AND thrust, add further indices)
        #     if self.include_thrust_model:
        #         ocp.constraints.idxbx_e = np.append(ocp.constraints.idxbx_e, [17, 18, 19, 20])

        # -- Index for ft1s, ft2s, ft3s, ft4s (When only included thrust, use the same indices)
        # elif self.include_thrust_model:
        #     ocp.constraints.idxbx_e = np.append(ocp.constraints.idxbx_e, [13, 14, 15, 16])

        # # -- Lower Terminal State Bound
        ocp.constraints.lbx_e = np.array(
            [self.params["v_min"],
             self.params["v_min"],
             self.params["v_min"],
             self.params["w_min"],
             self.params["w_min"],
             self.params["w_min"],
             -np.pi/2,
             -np.pi/2,
             -np.pi/2])

        # if self.tilt and self.include_servo_model:
        #     ocp.constraints.lbx_e = np.append(ocp.constraints.lbx_e,
        #                                       [self.params["a_min"],
        #                                        self.params["a_min"],
        #                                        self.params["a_min"],
        #                                        self.params["a_min"]])

        if self.include_thrust_model:
            ocp.constraints.lbx_e = np.append(ocp.constraints.lbx_e,
                                              [self.params["thrust_min"],
                                               self.params["thrust_min"],
                                               self.params["thrust_min"],
                                               self.params["thrust_min"]])

        # -- Upper Terminal State Bound
        ocp.constraints.ubx_e = np.array(
            [self.params["v_max"],
             self.params["v_max"],
             self.params["v_max"],
             self.params["w_max"],
             self.params["w_max"],
             self.params["w_max"],
             np.pi/2,
             np.pi/2,
             np.pi/2])

        # if self.tilt and self.include_servo_model:
        #     ocp.constraints.ubx_e = np.append(ocp.constraints.ubx_e,
        #                                       [self.params["a_max"],
        #                                        self.params["a_max"],
        #                                        self.params["a_max"],
        #                                        self.params["a_max"]])

        if self.include_thrust_model:
            ocp.constraints.ubx_e = np.append(ocp.constraints.ubx_e,
                                              [self.params["thrust_max"],
                                               self.params["thrust_max"],
                                               self.params["thrust_max"],
                                               self.params["thrust_max"]])

        # - Input box constraints bu
        # TODO Potentially a good idea to omit the input constraint when set the equivalent state
        # -- Index for ft1c, ft2c, ft3c, ft4c
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        # -- Index for a1c, a2c, a3c, a4c
        # if self.tilt:
        #     ocp.constraints.idxbu = np.append(ocp.constraints.idxbu, [4, 5, 6, 7])

        # -- Lower Input Bound
        ocp.constraints.lbu = np.array(
            [self.params["thrust_min"],
             self.params["thrust_min"],
             self.params["thrust_min"],
             self.params["thrust_min"]])

        # if self.tilt:
        #     ocp.constraints.lbu = np.append(ocp.constraints.lbu,
        #                                     [self.params["a_min"],
        #                                      self.params["a_min"],
        #                                      self.params["a_min"],
        #                                      self.params["a_min"]])

        # -- Upper Input Bound
        ocp.constraints.ubu = np.array(
            [self.params["thrust_max"],
             self.params["thrust_max"],
             self.params["thrust_max"],
             self.params["thrust_max"]])

        # if self.tilt:
        #     ocp.constraints.ubu = np.append(ocp.constraints.ubu,
        #                                     [self.params["a_max"],
        #                                      self.params["a_max"],
        #                                      self.params["a_max"],
        #                                      self.params["a_max"]])

        # Initial state and reference: Set all values such that robot is hovering
        # TODO debatable which initial states/inputs make sense -> not necessarily better than just all-zero!
        x_ref = np.zeros(nx)
        x_ref[6] = 1.0  # Quaternion qw

        # if self.tilt:
        #     # When included servo AND thrust, use further indices 
        #     if self.include_servo_model and self.include_thrust_model:
        #         x_ref[17:21] = self.phys.mass * self.phys.gravity / 4  # ft1s, ft2s, ft3s, ft4s
        #     # When only included thrust, use the same indices
        #     elif self.include_thrust_model:
        #         x_ref[13:17] = self.phys.mass * self.phys.gravity / 4  # ft1s, ft2s, ft3s, ft4s
        # else:
        #    x_ref[13:17] = self.phys.m * self.phys.gravity / 4  # ft1s, ft2s, ft3s, ft4s

        u_ref = np.zeros(nu)
        # Obeserved to be worse than zero!
        u_ref[0:4] = self.phys.m * self.phys.gravity / 4  # ft1c, ft2c, ft3c, ft4c

        # same order: phy_params = ca.vertcat(mass, gravity, inertia, kq_d_kt, dr, p1_b, p2_b, p3_b, p4_b, t_rotor, t_servo)
        self.acados_init_p = np.zeros(n_param)
        self.acados_init_p[0] = x_ref[6]  # qw
        # if len(self.phys.physical_param_list) != 24:
        #     raise ValueError("Physical parameters are not in the correct order. Please check the physical model.")
        self.acados_init_p[4:33] = np.array(self.phys.physical_param_list)
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref
        ocp.parameter_values = self.acados_init_p

        # Solver options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hpipm_mode = "BALANCE"  # "BALANCE", "SPEED_ABS", "SPEED", "ROBUST". Default: "BALANCE".
        # Start up flags:       [Seems only works for FULL_CONDENSING_QPOASES]
        # 0: no warm start; 1: warm start; 2: hot start. Default: 0
        # ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"  # explicit Runge-Kutta integrator
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver_cond_N = self.params["N_steps"]
        ocp.solver_options.tf = self.params["T_horizon"]

        # Build acados ocp into current working directory (which was created in super class)
        json_file_path = os.path.join("./" + ocp.model.name + "_acados_ocp.json")
        solver = AcadosOcpSolver(ocp, json_file=json_file_path, build=True)
        print("Generated C code for acados solver successfully to " + os.getcwd())

        return solver

    @abstractmethod
    def get_reference(self):
        pass

    def _create_reference_generator(self) -> HydrusXiReferenceGenerator:
        # Pass the model's and robot's properties to the reference generator
        return HydrusXiReferenceGenerator(self,
                                        self.tran_c_1, self.tran_c_2, self.tran_c_3, self.tran_c_4,
                                        self.phys.dr1, self.phys.dr2, self.phys.dr3, self.phys.dr4,
                                        self.phys.kq_d_kt, self.phys.m, self.phys.gravity)

    def create_acados_sim_solver(self, ts_sim: float, is_build: bool = True) -> AcadosSimSolver:
        ocp_model = super().get_acados_model()

        acados_sim = AcadosSim()
        acados_sim.model = ocp_model


        n_u = ocp_model.u.size()[0]  # 获取控制输入的维度
        n_param = ocp_model.p.size()[0]
        # same order: phy_params = ca.vertcat(mass, gravity, inertia, kq_d_kt, dr, p1_b, p2_b, p3_b, p4_b, t_rotor, t_servo)
        self.acados_init_p = np.zeros(n_param)
        self.acados_init_p[0] = 1.0  # qw
        self.acados_init_p[4:33] = np.array(self.phys.physical_param_list)
        acados_sim.parameter_values = self.acados_init_p
    
   

        acados_sim.solver_options.T = ts_sim
        return AcadosSimSolver(acados_sim, json_file=ocp_model.name + "_acados_sim.json", build=is_build)

  
