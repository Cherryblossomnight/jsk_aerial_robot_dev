import os
import yaml
import rospkg

# Read parameters from configuration file in the robot's package
rospack = rospkg.RosPack()

physical_param_path = os.path.join(rospack.get_path("hydrus_xi"), "config", "quad/PhysParamHydrusXi.yaml")
with open(physical_param_path, "r") as f:
    physical_param_dict = yaml.load(f, Loader=yaml.FullLoader)
physical_params = physical_param_dict["physical"]

l = physical_params["l"]
m = physical_params["m"]
m1 = physical_params["m1"]
m2 = physical_params["m2"]
m3 = physical_params["m3"]
m4 = physical_params["m4"]
j1 = physical_params["j1"]
j2 = physical_params["j2"]
j3 = physical_params["j3"]
gravity = physical_params["gravity"]
I1xx = physical_params["inertia_diag"][0]
I1yy = physical_params["inertia_diag"][1]
I1zz = physical_params["inertia_diag"][2]
I2xx = physical_params["inertia_diag"][0]
I2yy = physical_params["inertia_diag"][1]
I2zz = physical_params["inertia_diag"][2]
I3xx = physical_params["inertia_diag"][0]
I3yy = physical_params["inertia_diag"][1]
I3zz = physical_params["inertia_diag"][2]
I4xx = physical_params["inertia_diag"][0]
I4yy = physical_params["inertia_diag"][1]
I4zz = physical_params["inertia_diag"][2]
dr1 = physical_params["dr1"]
dr2 = physical_params["dr2"]
dr3 = physical_params["dr3"]
dr4 = physical_params["dr4"]
kq_d_kt = physical_params["kq_d_kt"]

t_servo = physical_params["t_servo"]  # Time constant of servo
t_rotor = physical_params["t_rotor"]  # Time constant of rotor

c0 = physical_params["c0"]
c1 = physical_params["c1"]
c2 = physical_params["c2"]
c3 = physical_params["c3"]
c4 = physical_params["c4"]


f1 = 0.0
f2 = 0.0
f3 = 0.0
t1 = 0.0
t2 = 0.0
t3 = 0.0
# concatenate the parameters to make a new list
physical_param_list = [
    l, m1, m2, m3, m4, m, gravity, 
    I1xx, I1yy, I1zz, I2xx, I2yy, I2zz,
    I3xx, I3yy, I3zz, I4xx, I4yy, I4zz,
    kq_d_kt,
    dr1, dr2, dr3, dr4, 
    t_rotor, t_servo,
    j1, j2, j3
]