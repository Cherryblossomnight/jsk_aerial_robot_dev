//
// Created by lijinjie on 23/11/29.
// Modified by Cherryblossomnight on 25/11/18.
//

#include "aerial_robot_control/nmpc/hydrus_xi_nmpc_controller.h"

using namespace aerial_robot_control;



void nmpc::HydrusXiNMPC::initialize(ros::NodeHandle nh, ros::NodeHandle nhp,
                                       boost::shared_ptr<aerial_robot_model::RobotModel> robot_model,
                                       boost::shared_ptr<aerial_robot_estimation::StateEstimator> estimator,
                                       boost::shared_ptr<aerial_robot_navigation::BaseNavigator> navigator,
                                       double ctrl_loop_du)
{
  BaseMPC::initialize(nh, nhp, robot_model, estimator, navigator, ctrl_loop_du);

  /* init dynamic reconfigure */
  ros::NodeHandle control_nh(nh_, "controller");
  ros::NodeHandle nmpc_nh(control_nh, "nmpc");
  nmpc_reconf_servers_.push_back(boost::make_shared<NMPCControlDynamicConfig>(nmpc_nh));
  nmpc_reconf_servers_.back()->setCallback(boost::bind(&HydrusXiNMPC::cfgNMPCCallback, this, _1, _2));

  /* set some ROS parameters */
  nmpc_nh.setParam("NN", mpc_solver_ptr_->NN_);
  nmpc_nh.setParam("NX", mpc_solver_ptr_->NX_);
  nmpc_nh.setParam("NU", mpc_solver_ptr_->NU_);

  /* timers */
  tmr_viz_ = nh_.createTimer(ros::Duration(0.05), &HydrusXiNMPC::callbackViz, this);

  /* publishers */
  pub_viz_pred_ = nh_.advertise<geometry_msgs::PoseArray>("nmpc/viz_pred", 1);
  pub_viz_ref_ = nh_.advertise<geometry_msgs::PoseArray>("nmpc/viz_ref", 1);
  pub_flight_cmd_ = nh_.advertise<spinal::FourAxisCommand>("four_axes/command", 1);
  pub_gimbal_control_ = nh_.advertise<sensor_msgs::JointState>("gimbals_ctrl", 1);
  pub_flight_config_cmd_spinal_ = nh_.advertise<spinal::FlightConfigCmd>("flight_config_cmd", 1);
  pub_x_u_ref_ = nh_.advertise<aerial_robot_msgs::PredXU>("x_u_ref", 1);
  pub_estimate_external_wrench_ = nh_.advertise<geometry_msgs::WrenchStamped>("estimated_external_wrench", 1);
  /* services */
  srv_set_control_mode_ = nh_.serviceClient<spinal::SetControlMode>("set_control_mode");

  /* subscribers */
  sub_joint_states_ = nh_.subscribe("joint_states", 5, &HydrusXiNMPC::callbackJointStates, this);
  sub_set_rpy_ = nh_.subscribe("set_rpy", 5, &HydrusXiNMPC::callbackSetRPY, this);
  sub_set_ref_x_u_ = nh_.subscribe("set_ref_x_u", 5, &HydrusXiNMPC::callbackSetRefXU, this);
  sub_set_traj_ = nh_.subscribe("set_ref_traj", 5, &HydrusXiNMPC::callbackSetRefTraj, this);
  sub_set_fixed_rotor_ = nh_.subscribe("set_fixed_rotor", 5, &HydrusXiNMPC::callbackSetFixedRotor, this);


  sub_target_external_wrench_ = nh_.subscribe("target_external_wrench", 5, &HydrusXiNMPC::callbackTargetExternalWrench, this);
  thrust_gimbal_nl_solver_ = boost::make_shared<nlopt::opt>(nlopt::LN_COBYLA, 8);
  /* init some values */
  setControlMode();

  initActuatorStates();
  initPredXU(x_u_ref_, mpc_solver_ptr_->NN_, mpc_solver_ptr_->NX_, mpc_solver_ptr_->NU_);

  //getParam<bool>(control_nh, "if_use_est_wrench_4_control", if_use_est_wrench_4_control_, false);

  //pub_disturb_wrench_ = nh_.advertise<geometry_msgs::WrenchStamped>("ext_wrench_est/value", 1);
  quat_prev_.setW(1.0);
  opt_result_.resize(8, 0.0);
  //pointer_data_->robot_model_ = robot_model_;
  robot_model_for_plan_ = boost::make_shared<aerial_robot_model::RobotModel>();
  opt_result_[0] = 9.3;
  opt_result_[1] = 9.3;
  opt_result_[2] = 9.3;
  opt_result_[3] = 9.3;
  opt_result_[4] = 3.14159;
  opt_result_[6] = 3.14159;
  init_sum_momentum_ = Eigen::VectorXd::Zero(6);
  integrate_term_ = Eigen::VectorXd::Zero(6);
  est_external_wrench_ = Eigen::VectorXd::Zero(6);

  prev_est_wrench_timestamp_ = 0;

  // wrench_estimate_thread_ = boost::thread([this]()
  //                                       {
  //                                         ros::Rate loop_rate(50.0);
  //                                         while(ros::ok())
  //                                           {
  //                                             externalWrenchEstimate();
  //                                             loop_rate.sleep();
  //                                           }
  //                                       });

  reset();
  ROS_INFO("MPC Controller initialized!");
}

// void nmpc::HydrusXiNMPC::initPlugins()
// {
//   //wrench_est_i_term_.initialize(nh_, robot_model_, estimator_, ctrl_loop_du_);

//   /* plugin: wrench estimator */
//   wrench_est_loader_ptr_ = boost::make_shared<pluginlib::ClassLoader<aerial_robot_control::WrenchEstActuatorMeasBase>>(
//       "aerial_robot_control", "aerial_robot_control::WrenchEstActuatorMeasBase");
//   try
//   {
//     // 1. read the plugin name from the parameter server
//     std::string wrench_estimator_name;
//     nh_.param("wrench_estimator_name", wrench_estimator_name, std::string("aerial_robot_control::WrenchEstNone"));

//     // 2. load the plugin
//     wrench_est_ptr_ = wrench_est_loader_ptr_->createInstance(wrench_estimator_name);
//     wrench_est_ptr_->initialize(nh_, robot_model_, estimator_, ctrl_loop_du_);
//     ROS_INFO("load wrench estimator plugin: %s", wrench_estimator_name.c_str());
//   }
//   catch (pluginlib::PluginlibException& ex)
//   {
//     ROS_ERROR("wrench_est_plugin: The plugin failed to load for some reason. Error: %s", ex.what());
//   }
// }

void nmpc::HydrusXiNMPC::activate()
{
  //initAllocMat();

  updateInertialParams();

  if (is_print_phys_params_)
    printPhysicalParams();

  // make takeoff slow
  modifyVelConstraints(-vel_limit_takeoff_, vel_limit_takeoff_);
  has_restored_vel_ = false;  // reset the flag, so that we can restore the velocity after hovering
  /* also for some commands that should be sent after takeoff */
  // enable imu sending, only works in simulation. Without this part, the IMU reading in Gazebo in incorrect.
  spinal::FlightConfigCmd flight_config_cmd;
  flight_config_cmd.cmd = spinal::FlightConfigCmd::INTEGRATION_CONTROL_ON_CMD;
  pub_flight_config_cmd_spinal_.publish(flight_config_cmd);

  BaseMPC::activate();
}

bool nmpc::HydrusXiNMPC::update()
{
  if (!BaseMPC::update())
    return false;

  this->controlCore();
  this->sendCmd();

  return true;
}

void nmpc::HydrusXiNMPC::reset()
{
  BaseMPC::reset();

  // reset x_u_ref_
  std::vector<double> xr_vec = meas2VecX(true);
  std::vector<double> u_vec(mpc_solver_ptr_->NU_, 0);
  u_vec[0] = 9.3;
  u_vec[1] = 9.3;
  u_vec[2] = 9.3;
  u_vec[3] = 9.3;
  u_vec[4] = 3.14159;
  u_vec[6] = 3.14159;
  // for (int i = 0; i < 51; i++)
  // {
  //    std::cout<<"row"<<i+1<<std::endl;
  //   for (int j = 0; j < 20; j++)
  //   {
  //     std::cout<<x_u_ref_.x.data[i*20 + j]<<" ";
  //   }
  //   std::cout<<std::endl;
  // }
  //std::cout<<"x_u_ref_"<<x_u_ref_.x.data<<std::endl;
  //  std::cout<<"size"<<x_u_ref_.x.data.size()<<std::endl;
  //     std::cout<<"size"<<x_u_ref_.u.data.size()<<std::endl;
  //    std::cout<<"size"<<xr_vec.size()<<std::endl;
  //         std::cout<<"size"<<u_vec.size()<<std::endl;
  int &NX = mpc_solver_ptr_->NX_, &NU = mpc_solver_ptr_->NU_, &NN = mpc_solver_ptr_->NN_;
  for (int i = 0; i < mpc_solver_ptr_->NN_; i++)
  {
    std::copy(xr_vec.begin(), xr_vec.begin() + NX, x_u_ref_.x.data.begin() + NX * i);
    std::copy(u_vec.begin(), u_vec.begin() + NU, x_u_ref_.u.data.begin() + NU * i);
  }
  std::copy(xr_vec.begin(), xr_vec.begin() + NX, x_u_ref_.x.data.begin() + NX * NN);

  // reset mpc solver
  mpc_solver_ptr_->resetXrUrByX0U0(xr_vec, u_vec);

  std::vector<double> x_vec = meas2VecX();

  mpc_solver_ptr_->resetSolverByX0U0(x_vec, u_vec);

  /* reset control input */
  flight_cmd_.base_thrust = std::vector<float>(motor_num_, 0.0);

  gimbal_ctrl_cmd_.name.clear();
  gimbal_ctrl_cmd_.position.clear();
  for (int i = 0; i < gimbal_num_; i++)
  {
    gimbal_ctrl_cmd_.name.emplace_back("gimbal" + std::to_string(i + 1));
    if (i % 2 == 0)
      gimbal_ctrl_cmd_.position.push_back(3.14159);
    else
      gimbal_ctrl_cmd_.position.push_back(0.0);
  }

  pub_gimbal_control_.publish(gimbal_ctrl_cmd_);

}

void nmpc::HydrusXiNMPC::initGeneralParams()
{
  ros::NodeHandle control_nh(nh_, "controller");
  ros::NodeHandle nmpc_nh(control_nh, "nmpc");
  ros::NodeHandle physical_nh(nh_, "physical");
  ros::NodeHandle alloc_nh(control_nh, "alloc");

  getParam<int>(alloc_nh, "type", alloc_type_, 0);
  getParam<double>(alloc_nh, "ft_thresh", ft_thresh_, 0.5);
  if (ft_thresh_ <= 0.0)
    throw std::runtime_error(
        "ft_thresh must be greater than zero! Please set a positive value for ft_thresh in the parameter server.");

  getParam<int>(physical_nh, "num_joints", joint_num_, 0);
  getParam<int>(physical_nh, "num_gimbals", gimbal_num_, 0);
  getParam<double>(physical_nh, "t_servo", t_servo_, 0.01);
  getParam<int>(physical_nh, "num_rotors", motor_num_, 0);
  getParam<double>(physical_nh, "t_rotor", t_rotor_, 0.01);

  getParam<double>(nmpc_nh, "T_samp", t_nmpc_samp_, 0.01);
  getParam<double>(nmpc_nh, "T_step", t_nmpc_step_, 0.01);
  getParam<double>(nmpc_nh, "T_horizon", t_nmpc_horizon_, 0.5);

  if (t_nmpc_samp_ != 1 / ctrl_loop_du_)
    throw std::runtime_error(
        "The NMPC sampling time T_samp is not equal to the control loop time! Please set T_step to 1/ctrl_loop_du_ in "
        "the config.");

  getParam<bool>(nmpc_nh, "is_attitude_ctrl", is_attitude_ctrl_, true);
  getParam<bool>(nmpc_nh, "is_body_rate_ctrl", is_body_rate_ctrl_, false);
  getParam<bool>(nmpc_nh, "is_print_phys_params", is_print_phys_params_, false);
  getParam<bool>(nmpc_nh, "is_debug", is_debug_, false);
  is_debug_ = true;

  if (is_debug_)
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
}

void nmpc::HydrusXiNMPC::initNMPCCostW()
{
  ros::NodeHandle control_nh(nh_, "controller");
  ros::NodeHandle nmpc_nh(control_nh, "nmpc");

  /* control parameters with dynamic reconfigure */
  double Qp_xy, Qp_z, Qv_xy, Qv_z, Qq_xy, Qq_z, Qw_xy, Qw_z, Qj, Qg, Rt, Rg;
  getParam<double>(nmpc_nh, "Qp_xy", Qp_xy, 600);
  getParam<double>(nmpc_nh, "Qp_z", Qp_z, 600);
  getParam<double>(nmpc_nh, "Qv_xy", Qv_xy, 10);
  getParam<double>(nmpc_nh, "Qv_z", Qv_z, 10);
  getParam<double>(nmpc_nh, "Qq_xy", Qq_xy, 1200);
  getParam<double>(nmpc_nh, "Qq_z", Qq_z, 600);
  getParam<double>(nmpc_nh, "Qw_xy", Qw_xy, 15);
  getParam<double>(nmpc_nh, "Qw_z", Qw_z, 10);
  getParam<double>(nmpc_nh, "Qj", Qj, 0.0);
  getParam<double>(nmpc_nh, "Qg", Qg, 20);
  getParam<double>(nmpc_nh, "Rt", Rt, 0.5);
  getParam<double>(nmpc_nh, "Rg", Rg, 0.01);

  // diagonal matrix
  mpc_solver_ptr_->setCostWDiagElement(0, Qp_xy);
  mpc_solver_ptr_->setCostWDiagElement(1, Qp_xy);
  mpc_solver_ptr_->setCostWDiagElement(2, Qp_z);
  mpc_solver_ptr_->setCostWDiagElement(3, Qv_xy);
  mpc_solver_ptr_->setCostWDiagElement(4, Qv_xy);
  mpc_solver_ptr_->setCostWDiagElement(5, Qv_z);
  mpc_solver_ptr_->setCostWDiagElement(6, 0);
  mpc_solver_ptr_->setCostWDiagElement(7, Qq_xy);
  mpc_solver_ptr_->setCostWDiagElement(8, Qq_xy);
  mpc_solver_ptr_->setCostWDiagElement(9, Qq_z);
  mpc_solver_ptr_->setCostWDiagElement(10, Qw_xy);
  mpc_solver_ptr_->setCostWDiagElement(11, Qw_xy);
  mpc_solver_ptr_->setCostWDiagElement(12, Qw_z);
  for (int i = 13; i < 13 + joint_num_; ++i)
    mpc_solver_ptr_->setCostWDiagElement(i, Qj);
  for (int i = 13 + joint_num_; i < 13 + joint_num_ + gimbal_num_; ++i)
    mpc_solver_ptr_->setCostWDiagElement(i, Qg);
  for (int i = mpc_solver_ptr_->NX_; i < mpc_solver_ptr_->NX_ + motor_num_; ++i)
    mpc_solver_ptr_->setCostWDiagElement(i, Rt, false);
  for (int i = mpc_solver_ptr_->NX_ + motor_num_; i < mpc_solver_ptr_->NX_ + motor_num_ + gimbal_num_; ++i)
    mpc_solver_ptr_->setCostWDiagElement(i, Rg, false);
}

void nmpc::HydrusXiNMPC::initNMPCConstraints()
{
  ros::NodeHandle control_nh(nh_, "controller");
  ros::NodeHandle nmpc_nh(control_nh, "nmpc");

  double body_rate_max, body_rate_min;
  getParam<double>(nmpc_nh, "w_max", body_rate_max, 6.0);
  getParam<double>(nmpc_nh, "w_min", body_rate_min, -6.0);
  getParam<double>(nmpc_nh, "v_max", vel_max_, 1.0);
  getParam<double>(nmpc_nh, "v_min", vel_min_, -1.0);
  getParam<double>(nmpc_nh, "thrust_max", thrust_ctrl_max_, 0.0);
  getParam<double>(nmpc_nh, "thrust_min", thrust_ctrl_min_, 0.0);
  getParam<double>(nmpc_nh, "j_max", joint_angle_max_, 1.5708);
  getParam<double>(nmpc_nh, "j_min", joint_angle_min_, -1.5708);
  getParam<double>(nmpc_nh, "g_max", gimbal_angle_max_, 6.2832);
  getParam<double>(nmpc_nh, "g_min", gimbal_angle_min_, -6.2832);


  //  TODO: this should be set in flight_navigation; don't know why set 0.2 results solver failure
  getParam<double>(control_nh, "vel_limit_takeoff", vel_limit_takeoff_, 1.0);  // m/s

  // lbx and ubx
  std::vector<int> idxbx = mpc_solver_ptr_->getConstraintsIdxbx();
  std::vector<int> idxbx_desired = { 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

  // if (idxbx.size() != idxbx_desired.size() || !std::equal(idxbx.begin(), idxbx.end(), idxbx_desired.begin()))
  // {
  //   ROS_ERROR("idxbx is not equal to idxbx_desired, we cannot set constraints lbx and ubx!");
  // }

  std::vector<double> lbx = { vel_min_, vel_min_, vel_min_, body_rate_min, body_rate_min, body_rate_min };
  std::vector<double> ubx = { vel_max_, vel_max_, vel_max_, body_rate_max, body_rate_max, body_rate_max };
  lbx.resize(6 + joint_num_ + gimbal_num_);
  ubx.resize(6 + joint_num_ + gimbal_num_);
  for (int i = 0; i < joint_num_; i++)
  {
    lbx[6 + i] = joint_angle_min_;
    ubx[6 + i] = joint_angle_max_;
  }
  for (int i = joint_num_; i < joint_num_ + gimbal_num_; i++)
  {
    lbx[6 + i] = gimbal_angle_min_;
    ubx[6 + i] = gimbal_angle_max_;
  }
  mpc_solver_ptr_->setConstraintsLbx(lbx);
  mpc_solver_ptr_->setConstraintsUbx(ubx);

  // lbxe and ubxe
  std::vector<int> idxbxe = mpc_solver_ptr_->getConstraintsIdxbxe();
  std::vector<int> idxbxe_desired = idxbx_desired;
  if (idxbxe.size() != idxbxe_desired.size() || !std::equal(idxbxe.begin(), idxbxe.end(), idxbxe_desired.begin()))
  {
    ROS_ERROR("idxbx_end is not equal to idxbx_end_desired, we cannot set constraints lbxe and ubxe!");
  }
  mpc_solver_ptr_->setConstraintsLbxe(lbx);
  mpc_solver_ptr_->setConstraintsUbxe(ubx);

  // lbu and ubu
  std::vector<int> idxbu = mpc_solver_ptr_->getConstraintsIdxbu();
  std::vector<int> idxbu_desired(motor_num_ + gimbal_num_);
  for (int i = 0; i < motor_num_; i++)
  {
    idxbu_desired[i] = i;
  }
  for (int i = 0; i < gimbal_num_; i++)
  {
    idxbu_desired[motor_num_ + i] = motor_num_ + i;
  }
  if (idxbu.size() != idxbu_desired.size() || !std::equal(idxbu.begin(), idxbu.end(), idxbu_desired.begin()))
  {
    ROS_ERROR("idxbu is not equal to idxbu_desired, we cannot set constraints lbu and ubu!");
  }

  std::vector<double> lbu(motor_num_ + gimbal_num_, 0.0);
  std::vector<double> ubu(motor_num_ + gimbal_num_, 0.0);
  for (int i = 0; i < motor_num_; i++)
  {
    lbu[i] = thrust_ctrl_min_;
    ubu[i] = thrust_ctrl_max_;
  }
  for (int i = 0; i < gimbal_num_; i++)
  {
    lbu[motor_num_ + i] = gimbal_angle_min_;
    ubu[motor_num_ + i] = gimbal_angle_max_;
  }
  mpc_solver_ptr_->setConstraintsLbu(lbu);
  mpc_solver_ptr_->setConstraintsUbu(ubu);
}

void nmpc::HydrusXiNMPC::setControlMode()
{
  bool res = ros::service::waitForService("set_control_mode", ros::Duration(5));
  if (!res)
  {
    ROS_ERROR("cannot find service named set_control_mode");
  }
  ros::Duration(2.0).sleep();
  spinal::SetControlMode set_control_mode_srv;
  set_control_mode_srv.request.is_attitude = is_attitude_ctrl_;
  set_control_mode_srv.request.is_body_rate = is_body_rate_ctrl_;
  while (!srv_set_control_mode_.call(set_control_mode_srv))
    ROS_WARN_THROTTLE(1,
                      "Waiting for set_control_mode service.... If you always see this message, the robot cannot fly.");

  ROS_INFO("Set control mode: attitude = %d and body rate = %d", set_control_mode_srv.request.is_attitude,
           set_control_mode_srv.request.is_body_rate);
}

void nmpc::HydrusXiNMPC::initAllocMat()
{
  /* get physical param */
  int rotor_num = robot_model_->getRotorNum();  // For tilt-rotor, rotor_num = servo_num
  const auto& rotor_p = robot_model_->getRotorsOriginFromCog<Eigen::Vector3d>();
  const map<int, int> rotor_dr = robot_model_->getRotorDirection();
  double kq_d_kt = abs(robot_model_->getMFRate());  // PAY ATTENTION: should be positive value

  /* alloc mat */
  alloc_mat_.resize(0, 0);
  alloc_mat_pinv_.resize(0, 0);

  // construct alloc_mat_
  alloc_mat_ = Eigen::MatrixXd::Zero(6, 2 * rotor_num);

  for (int i = 0; i < rotor_num; i++)
  {
    Eigen::Vector3d p_b = rotor_p[i];
    int dr = rotor_dr.find(i + 1)->second;  // PAY ATTENTION: the rotor index starts from 1!!!!!!!!!!!!!!!!!!!!!

    double sqrt_p_xy = sqrt(p_b.x() * p_b.x() + p_b.y() * p_b.y());

    // - force
    alloc_mat_(0, 2 * i) = p_b.y() / sqrt_p_xy;
    alloc_mat_(1, 2 * i) = -p_b.x() / sqrt_p_xy;
    alloc_mat_(2, 2 * i + 1) = 1;

    // - torque
    alloc_mat_(3, 2 * i) = -dr * kq_d_kt * p_b.y() / sqrt_p_xy + p_b.x() * p_b.z() / sqrt_p_xy;
    alloc_mat_(4, 2 * i) = dr * kq_d_kt * p_b.x() / sqrt_p_xy + p_b.y() * p_b.z() / sqrt_p_xy;
    alloc_mat_(5, 2 * i) = -p_b.x() * p_b.x() / sqrt_p_xy - p_b.y() * p_b.y() / sqrt_p_xy;

    alloc_mat_(3, 2 * i + 1) = p_b.y();
    alloc_mat_(4, 2 * i + 1) = -p_b.x();
    alloc_mat_(5, 2 * i + 1) = -dr * kq_d_kt;
  }

  alloc_mat_pinv_ = aerial_robot_model::pseudoinverse(alloc_mat_);
}

/* Note: The difference between this function and prepareNMPCParams() is:
 * this function set idx for different physical parameters.
 */
void nmpc::HydrusXiNMPC::initNMPCParams()
{
  /* construct acados parameters */
  std::vector<double> acados_p(mpc_solver_ptr_->NP_, 0.0);

  acados_p[0] = 1.0;  // qw
  idx_p_quat_end_ = 3;
  int idx;
  // TODO: this condition is temporary for drones that don't pass in phys param (bi, tri, fix-qd)
  if (mpc_solver_ptr_->NP_ > 4 + 6)  // 4 for quaternion, 6 for disturbances
  {
    ROS_INFO("Set physical parameters for NMPC solver");

    std::vector<double> phys_p = PhysToNMPCParams();
    std::copy(phys_p.begin(), phys_p.end(), acados_p.begin() + idx_p_quat_end_ + 1);
    idx_p_phys_end_ = idx_p_quat_end_ + phys_p.size();
  }
  else
  {
    idx_p_phys_end_ = idx_p_quat_end_;
  }
  for (int i = 0; i < acados_p.size(); i++)
  {
 std::cout << "acados_p[" << i << "]=" << acados_p[i] << std::endl;
  }

  KDL::RigidBodyInertia rigid_body =  robot_model_->getInertiaMap().at("link1");
  Eigen::MatrixXd inertia_mtx = aerial_robot_model::kdlToEigen(rigid_body.getRotationalInertia());
  std::cout<<"inertia_mtx"<<inertia_mtx<<std::endl;
  rigid_body =  robot_model_->getInertiaMap().at("gimbal_link1");
  inertia_mtx = aerial_robot_model::kdlToEigen(rigid_body.getRotationalInertia());
  std::cout<<"inertia_mtx"<<inertia_mtx<<std::endl;
  /* set acados parameters */
  mpc_solver_ptr_->setParameters(acados_p);
}

void nmpc::HydrusXiNMPC::updateInertialParams()
{
  mass_.resize(5);
  mass_[0] = robot_model_->getInertiaMap().at("link1").getMass() + robot_model_->getInertiaMap().at("gimbal_link1").getMass();
  mass_[1] = robot_model_->getInertiaMap().at("link2").getMass() + robot_model_->getInertiaMap().at("gimbal_link2").getMass();
  mass_[2] = robot_model_->getInertiaMap().at("link3").getMass() + robot_model_->getInertiaMap().at("gimbal_link3").getMass();
  mass_[3] = robot_model_->getInertiaMap().at("link4").getMass() + robot_model_->getInertiaMap().at("gimbal_link4").getMass();
  mass_[4] = robot_model_->getMass();
  gravity_const_ = robot_model_->getGravity()[2];

  Eigen::Matrix3d inertia_mtx = robot_model_->getInertia<Eigen::Matrix3d>();
  inertia_.resize(5);
  for (auto &row : inertia_) 
    row.resize(3, 0.0);
  inertia_[0][0] = inertia_mtx(0,0);    
  inertia_[0][1] = inertia_mtx(1,1);   
  inertia_[0][2] = inertia_mtx(2,2);     

  KDL::RigidBodyInertia rigid_body =  robot_model_->getInertiaMap().at("link1");
  inertia_mtx = aerial_robot_model::kdlToEigen(rigid_body.getRotationalInertia());
  
  inertia_[1][0] = 0.001020;    
  inertia_[1][1] = 0.008076; 
  inertia_[1][2] = 0.007129;     

  rigid_body =  robot_model_->getInertiaMap().at("link2");
  inertia_mtx = aerial_robot_model::kdlToEigen(rigid_body.getRotationalInertia());
  inertia_[2][0] = 0.000808;
  inertia_[2][1] = 0.012747;
  inertia_[2][2] = 0.012027;

  rigid_body =  robot_model_->getInertiaMap().at("link3");
  inertia_mtx = aerial_robot_model::kdlToEigen(rigid_body.getRotationalInertia());
  inertia_[3][0] = 0.000808;
  inertia_[3][1] = 0.012747;
  inertia_[3][2] = 0.012027;

  rigid_body =  robot_model_->getInertiaMap().at("link4");
  inertia_mtx = aerial_robot_model::kdlToEigen(rigid_body.getRotationalInertia());
  inertia_[4][0] = 0.000785;
  inertia_[4][1] = 0.007046;
  inertia_[4][2] = 0.006334;
}
void nmpc::HydrusXiNMPC::modifyVelConstraints(double vel_min, double vel_max) const
{
  // Hardcoded: the vel idx is 3,4,5, which are the first three elements. TODO: consider to make it more general

  std::vector<double> lbx = mpc_solver_ptr_->getConstraintsLbx();
  lbx[0] = vel_min;
  lbx[1] = vel_min;
  lbx[2] = vel_min;
  mpc_solver_ptr_->setConstraintsLbx(lbx);

  std::vector<double> lbxe = mpc_solver_ptr_->getConstraintsLbxe();
  lbxe[0] = vel_min;
  lbxe[1] = vel_min;
  lbxe[2] = vel_min;
  mpc_solver_ptr_->setConstraintsLbxe(lbxe);

  std::vector<double> ubx = mpc_solver_ptr_->getConstraintsUbx();
  ubx[0] = vel_max;
  ubx[1] = vel_max;
  ubx[2] = vel_max;
  mpc_solver_ptr_->setConstraintsUbx(ubx);

  std::vector<double> ubxe = mpc_solver_ptr_->getConstraintsUbxe();
  ubxe[0] = vel_max;
  ubxe[1] = vel_max;
  ubxe[2] = vel_max;
  mpc_solver_ptr_->setConstraintsUbxe(ubxe);

  ROS_INFO("Velocity constraints modified: [%f, %f, %f] for lbx and [%f, %f, %f] for ubx", lbx[0], lbx[1], lbx[2],
           ubx[0], ubx[1], ubx[2]);
}

std::vector<double> nmpc::HydrusXiNMPC::PhysToNMPCParams() const
{
  int rotor_num = robot_model_->getRotorNum();  // For tilt-rotor, rotor_num = servo_num
  const auto& rotor_p = robot_model_->getRotorsOriginFromCog<Eigen::Vector3d>();
  const map<int, int> rotor_dr = robot_model_->getRotorDirection();
  double kq_d_kt = abs(robot_model_->getMFRate());  // PAY ATTENTION: should be positive value

  std::vector<double> phys_p(29, 0);
  // order: mass, gravity, Ixx, Iyy, Izz, kq_d_kt, dr1, p1_b, dr2, p2_b, dr3, p3_b, dr4, p4_b, t_rotor, t_servo
  // ee_p, ee_qwxyz
  phys_p[0] = 0.6;
  phys_p[1] = mass_[0];
  phys_p[2] = mass_[1];
  phys_p[3] = mass_[2];
  phys_p[4] = mass_[3];
  phys_p[5] = mass_[4];
  phys_p[6] = gravity_const_;

  phys_p[7] = inertia_[1][0];
  phys_p[8] = inertia_[1][1];
  phys_p[9] = inertia_[1][2];

  phys_p[10] = inertia_[2][0];
  phys_p[11] = inertia_[2][1];
  phys_p[12] = inertia_[2][2];

  phys_p[13] = inertia_[3][0];
  phys_p[14] = inertia_[3][1];
  phys_p[15] = inertia_[3][2];

  phys_p[16] = inertia_[4][0];
  phys_p[17] = inertia_[4][1];
  phys_p[18] = inertia_[4][2];

  phys_p[19] = kq_d_kt;
  int idx = 20;
  for (int i = 0; i < rotor_num; i++)
  {
    phys_p[idx] = rotor_dr.find(i + 1)->second;
    idx++;
  }

  phys_p[idx] = t_rotor_;
  idx++;
  phys_p[idx] = t_servo_;
  idx++;
  phys_p[idx] = joint_angles_[0];
  idx++;
  phys_p[idx] = joint_angles_[1];
  idx++;
  phys_p[idx] = joint_angles_[2];
  idx++;

  std::vector<double> contact_frame_p = { 0.0, 0.0, 0.0 };
  std::vector<double> contact_frame_q = { 1.0, 0.0, 0.0, 0.0 };
  if (traj_child_frame_id_ == "cog")
  {
  }
  else if (traj_child_frame_id_ == "ee")
  {
    if (robot_model_->hasFrame("ee_contact"))
      robot_model_->getCoGtoFramePosQuat("ee_contact", contact_frame_p, contact_frame_q);
    else
      ROS_WARN_THROTTLE(5, "No frame named ee_contact in the robot model! The end-effector pose will be set to CoG.");
  }
  else
  {
    ROS_WARN_THROTTLE(5, "Unsupported traj_child_frame_id_! The end-effector pose will be set to CoG.");
  }

  // std::copy(contact_frame_p.begin(), contact_frame_p.end(), phys_p.begin() + idx);
  // idx += static_cast<int>(contact_frame_p.size());
  // std::copy(contact_frame_q.begin(), contact_frame_q.end(), phys_p.begin() + idx);
  // idx += static_cast<int>(contact_frame_q.size());
  return phys_p;
}

void nmpc::HydrusXiNMPC::controlCore()
{
  // restore velocity constraints after hovering
  if (navigator_->getNaviState() == aerial_robot_navigation::HOVER_STATE and has_restored_vel_ == false)
  {
    modifyVelConstraints(vel_min_, vel_max_);
    has_restored_vel_ = true;
  }

  prepareNMPCRef();

  prepareNMPCParams();

  /* prepare initial value */
  std::vector<double> bx0 = meas2VecX();

  /* solve */
  try
  {
    mpc_solver_ptr_->solve(bx0, is_debug_);

  }
 catch (mpc_solver::AcadosSolveException& e)
  {
   ROS_FATAL("NMPC solver failed. Details: %s", e.what());
  }
  // The result is stored in mpc_solver_ptr_->uo_

  /* get result */
  // - thrust
  externalWrenchEstimate();
  for (int i = 0; i < motor_num_; i++)
  {
    flight_cmd_.base_thrust[i] = (float)getCommand(i);
  }

  // - servo angle
  gimbal_ctrl_cmd_.header.stamp = ros::Time::now();
  gimbal_ctrl_cmd_.name.clear();
  gimbal_ctrl_cmd_.position.clear();
  for (int i = 0; i < gimbal_num_; i++)
  {
    gimbal_ctrl_cmd_.name.emplace_back("gimbal" + std::to_string(i + 1));
    gimbal_ctrl_cmd_.position.push_back(getCommand(motor_num_ + i));
  }
}

void nmpc::HydrusXiNMPC::sendCmd()
{
  /* publish */
  if (motor_num_ > 0)
    pub_flight_cmd_.publish(flight_cmd_);
  if (gimbal_num_ > 0)
    pub_gimbal_control_.publish(gimbal_ctrl_cmd_);
}

void nmpc::HydrusXiNMPC::prepareNMPCRef()
{
  // TODO: wrap to a state machine
  if (!is_traj_tracking_)
  {
    setPointRefFromNavigator(true);
    return;
  }

  /* if in trajectory tracking mode, the ref is set by callbackSetRefXU.
   * So here we check if the traj info is still received. If not, we turn off the tracking mode */
  double t_interval_sec = (ros::Time::now() - x_u_ref_.header.stamp).toSec();

  double traj_switch_time = 0.1;  // second
  double min_no_traj_time = 0.5;  // second

  if (t_interval_sec <= traj_switch_time)
    return;

  // - for the switch between two trajectories, such as from single point tracking to traj
  if (traj_switch_time < t_interval_sec && t_interval_sec <= min_no_traj_time)
  {
    last_traj_msg_.points.clear();  // every time end one traj, clear the traj msg
    return;
  }

  // - for the general case that no traj msg is received for a long time
  is_traj_tracking_ = false;
  traj_child_frame_id_ = "cog";
  ROS_INFO_STREAM(
      "No traj msg for 0.5s. Trajectory tracking mode is off! Return to the hovering! The child frame is set to "
      << traj_child_frame_id_ << ".");

  tf::Vector3 current_pos = estimator_->getPos(Frame::COG, estimate_mode_);
  tf::Vector3 current_rpy = estimator_->getEuler(Frame::COG, estimate_mode_);
  // navigator_->setTargetPosX(static_cast<float>(current_pos.x()));
  // navigator_->setTargetPosY(static_cast<float>(current_pos.y()));
  // navigator_->setTargetPosZ(static_cast<float>(current_pos.z()));
  // navigator_->setTargetVelX(0.0);
  // navigator_->setTargetVelY(0.0);
  // navigator_->setTargetVelZ(0.0);
  // navigator_->setTargetRoll(0.0);
  // navigator_->setTargetPitch(0.0);
  // navigator_->setTargetYaw(static_cast<float>(current_rpy.z()));
  // navigator_->setTargetOmegaX(0.0);
  // navigator_->setTargetOmegaY(0.0);
  // navigator_->setTargetOmegaZ(0.0);

  setPointRefFromNavigator(false);
}

void nmpc::HydrusXiNMPC::prepareNMPCParams()
{
  updateInertialParams();

  // TODO: this condition is temporary for drones that don't pass in phys param (bi, tri, fix-qd)
  if (mpc_solver_ptr_->NP_ > 4 + 6)
  {
    std::vector<double> phys_p = PhysToNMPCParams();
    mpc_solver_ptr_->setParameters(phys_p, idx_p_quat_end_ + 1);
  }
}

void nmpc::HydrusXiNMPC::setPointRefFromNavigator(bool is_shifted_not_set_all)
{
  tf::Vector3 target_cog_pos_in_w = navigator_->getTargetPos();
  tf::Vector3 target_cog_vel_in_w = navigator_->getTargetVel();
  tf::Vector3 target_cog_rpy = navigator_->getTargetRPY();
  tf::Quaternion target_cog_quat;
  target_cog_quat.setRPY(target_cog_rpy.x(), target_cog_rpy.y(), target_cog_rpy.z());
  tf::Vector3 target_cog_omega = navigator_->getTargetOmega();
  tf::Vector3 target_cog_ext_force_in_w = tf::Vector3(target_external_wrench_.wrench.force.x, target_external_wrench_.wrench.force.y, target_external_wrench_.wrench.force.z);
  tf::Vector3 target_cog_ext_torque_in_b = tf::Vector3(target_external_wrench_.wrench.torque.x, target_external_wrench_.wrench.torque.y, target_external_wrench_.wrench.torque.z);
 
  ros::Duration dt = ros::Time::now() - time_last_target_external_wrench_;
  if (dt.toSec() > 1.0)
  {
    target_external_wrench_.wrench = geometry_msgs::Wrench();
  }
 // std::cout<<"target_cog_vel:"<<target_cog_vel_in_w.x()<<","<<target_cog_vel_in_w.y()<<","<<target_cog_vel_in_w.z()<<std::endl;
// std::cout<<"target_cog_quat:"<<target_cog_quat.w()<<","<<target_cog_quat.x()<<","<<target_cog_quat.y()<<","<<target_cog_quat.z()<<std::endl;
  if (is_shifted_not_set_all)
    setXrUrRef(target_cog_pos_in_w, target_cog_vel_in_w, tf::Vector3(0, 0, 0), target_cog_quat, target_cog_omega,
              tf::Vector3(0, 0, 0), target_cog_ext_force_in_w, target_cog_ext_torque_in_b, -1);
  else
    setXrUrRef(target_cog_pos_in_w, target_cog_vel_in_w, tf::Vector3(0, 0, 0), target_cog_quat, target_cog_omega,
              tf::Vector3(0, 0, 0), target_cog_ext_force_in_w, target_cog_ext_torque_in_b, -1);

  rosXU2VecXU(x_u_ref_, mpc_solver_ptr_->xr_, mpc_solver_ptr_->ur_);
  mpc_solver_ptr_->setReference(mpc_solver_ptr_->xr_, mpc_solver_ptr_->ur_, true);
}

/**
 * @brief calXrUrRef: calculate the reference state and control input
 * @param ref_pos_i
 * @param ref_vel_i
 * @param ref_acc_i - the acceleration is in the inertial frame, no including the gravity
 * @param ref_quat_ib
 * @param ref_omega_b
 * @param ref_ang_acc_b
 * @param horizon_idx - set -1 for adding the target point to the end of the reference trajectory; 0 ~ NN for adding
 * the target point to the horizon_idx interval; -2 for adding the target point to all points
 */
void nmpc::HydrusXiNMPC::setXrUrRef(const tf::Vector3& ref_pos_i, const tf::Vector3& ref_vel_i,
                                       const tf::Vector3& ref_acc_i, const tf::Quaternion& ref_quat_ib,
                                       const tf::Vector3& ref_omega_b, const tf::Vector3& ref_ang_acc_b,
                                       const tf::Vector3& ref_ext_force_i, const tf::Vector3& ref_ext_torque_b,
                                       const int& horizon_idx)
{
  int& NX = mpc_solver_ptr_->NX_;
  int& NU = mpc_solver_ptr_->NU_;
  int& NN = mpc_solver_ptr_->NN_;

  /* calculate the reference wrench in the body frame */
  Eigen::VectorXd acc_with_g_i(3);
  acc_with_g_i(0) = ref_acc_i.x();
  acc_with_g_i(1) = ref_acc_i.y();
  acc_with_g_i(2) = ref_acc_i.z() + gravity_const_;  // add gravity

  // coordinate transformation
  tf::Quaternion q_bi = ref_quat_ib.inverse();
  Eigen::Matrix3d rot_bi;
  tf::matrixTFToEigen(tf::Transform(q_bi).getBasis(), rot_bi);
  Eigen::VectorXd ref_acc_b = rot_bi * acc_with_g_i;

  Eigen::VectorXd ref_wrench_b(6);
  ref_wrench_b(0) = ref_acc_b(0) * mass_[4] - ref_ext_force_i.x();
  ref_wrench_b(1) = ref_acc_b(1) * mass_[4] - ref_ext_force_i.y();
  ref_wrench_b(2) = ref_acc_b(2) * mass_[4] - ref_ext_force_i.z();
  ref_wrench_b(3) = ref_ang_acc_b.x() * inertia_[0][0] - ref_ext_torque_b.x();
  ref_wrench_b(4) = ref_ang_acc_b.y() * inertia_[0][1] - ref_ext_torque_b.y();
  ref_wrench_b(5) = ref_ang_acc_b.z() * inertia_[0][2] - ref_ext_torque_b.z();

  std::cout <<"  "<<ref_ext_force_i.x()<<","<<ref_ext_force_i.y()<<","<<ref_ext_force_i.z()<<std::endl;

  /* calculate X U from ref, aka. control allocation */
  std::vector<double> x(NX);
  std::vector<double> u(NU);
  allocateToXU(ref_pos_i, ref_vel_i, ref_quat_ib, ref_omega_b, ref_wrench_b, x, u);

  /* set values */
  if (horizon_idx == -1)
  {
    // Aim: gently add the target point to the end of the reference trajectory
    // - x: NN + 1, u: NN
    // - for 0 ~ NN-2 x and u, shift
    // - copy x to x: NN-1 and NN, copy u to u: NN-1
    for (int i = 0; i < NN - 1; i++)
    {
      // shift one step
      std::copy(x_u_ref_.x.data.begin() + NX * (i + 1), x_u_ref_.x.data.begin() + NX * (i + 2),
                x_u_ref_.x.data.begin() + NX * i);
      std::copy(x_u_ref_.u.data.begin() + NU * (i + 1), x_u_ref_.u.data.begin() + NU * (i + 2),
                x_u_ref_.u.data.begin() + NU * i);
    }

    // std::cout<<"ref_x:"<<std::endl;
    // for(int i=0;i<NX;i++)
    //   std::cout<<x[i]<<",";
    std::cout<<std::endl;
    std::copy(x.begin(), x.begin() + NX, x_u_ref_.x.data.begin() + NX * (NN - 1));
    std::copy(u.begin(), u.begin() + NU, x_u_ref_.u.data.begin() + NU * (NN - 1));

    std::copy(x.begin(), x.begin() + NX, x_u_ref_.x.data.begin() + NX * NN);

    return;
  }

  if (horizon_idx == -2)
  {
    // Aim: set the target point to all points in the horizon
    for (int i = 0; i < NN; i++)
    {
      std::copy(x.begin(), x.begin() + NX, x_u_ref_.x.data.begin() + NX * i);
      std::copy(u.begin(), u.begin() + NU, x_u_ref_.u.data.begin() + NU * i);
    }
    std::copy(x.begin(), x.begin() + NX, x_u_ref_.x.data.begin() + NX * NN);

    return;
  }

  if (horizon_idx < 0 || horizon_idx > NN)
  {
    ROS_WARN("horizon_idx is out of range! CalXrUrRef failed!");
    return;
  }

  std::copy(x.begin(), x.begin() + NX, x_u_ref_.x.data.begin() + NX * horizon_idx);
  if (horizon_idx < NN)
    std::copy(u.begin(), u.begin() + NU, x_u_ref_.u.data.begin() + NU * horizon_idx);
  std::cout<<"set ref at horizon idx "<<std::endl;

}


double leastSquare(const std::vector<double> &x, std::vector<double> &grad, void *robot_model_ptr)
{
  // cnt++;
  aerial_robot_model::RobotModel* robot_model = reinterpret_cast<aerial_robot_model::RobotModel*>(robot_model_ptr);
  //boost::shared_ptr<aerial_robot_model::RobotModel> robot_model_for_plan = boost::make_shared<aerial_robot_model::RobotModel>();
 // KDL::JntArray joint_positions_former = pointer_data.robot_model_->getJointPositions(); // real;
  Eigen::VectorXd opt_ref = Eigen::VectorXd::Zero(8);
  opt_ref[0] = 9.3;
  opt_ref[1] = 9.3;
  opt_ref[2] = 9.3;
  opt_ref[3] = 9.3;
  opt_ref[4] = 3.14159;
  opt_ref[6] = 3.14159;
  

  // std::cout<<"x1"<<std::endl;
  KDL::JntArray joint_positions = robot_model->getJointPositions(); 
  Eigen::VectorXd u_vec = Eigen::VectorXd::Zero(8);
  for (int i; i < joint_positions.data.size(); i++)
  {
   // std::cout<<"x["<<i<<"]="<<joint_positions(i)<<", ";
  }
  Eigen::VectorXd thrust_cmds = robot_model->getThrustCmd();
  for(int i = 0; i < 4; i++)
  { 
    u_vec(i) = thrust_cmds(i);
    u_vec(i+4) = joint_positions(3 * i);
    joint_positions(3 * i) = x.at(i+4);
  }

  robot_model->updateRobotModel(joint_positions); 
// std::cout<<"x2"<<std::endl;
  for (int i; i < joint_positions.data.size(); i++)
  {
   // std::cout<<"x["<<i<<"]="<<joint_positions2(i)<<", ";
  }
  Eigen::MatrixXd A = robot_model->calcWrenchMatrixOnCoG();
  Eigen::MatrixXd R = robot_model->getRotationMatrix();
   // std::cout<<"update1"<<A<<std::endl;
  Eigen::VectorXd thrusts = Eigen::VectorXd::Zero(robot_model->getRotorNum());
  for(int i = 0; i < robot_model->getRotorNum(); i++)
    thrusts(i) = x.at(i);
  Eigen::VectorXd achieved_wrench = A * thrusts;
  //std::cout<<"R" <<R<<std::endl;
  achieved_wrench.segment(0, 3) = R * achieved_wrench.segment(0, 3); // to world frame
  Eigen::VectorXd desired_wrench = robot_model->getDesiredWrench();
//std::cout<<"b" <<achieved_wrench.transpose()<<std::endl;
  //std::cout<<"achieved_wrench:"<<achieved_wrench.transpose()<<std::endl;
  for (int i; i < x.size(); i++)
  {
   // std::cout<<"x["<<i<<"]="<<x.at(i)<<", ";
  }
  //std::cout<<ros::Time::now().toSec()<<" "<<std::endl;
  //std::cout<<std::endl;
  double cost = 1*(achieved_wrench.segment(0, 3) - desired_wrench.segment(0, 3)).squaredNorm() +
                5*(achieved_wrench.segment(3, 3) - desired_wrench.segment(3, 3)).squaredNorm();
  //std::cout<<"cost:"<<cost<<std::endl;
  //cost += 0.1* (u_vec.segment(0, 4) - opt_ref.segment(0, 4)).squaredNorm() +
         // 0.1* (u_vec.segment(4, 4) - opt_ref.segment(4, 4)).squaredNorm();
  for (int i = 0; i < 4; i++)
  {
    //cost += 0.1*(x.at(i) - u_vec(i)) * (x.at(i) - u_vec(i)) + 0.1*(x.at(i+4) - u_vec(i+4)) * (x.at(i+4) - u_vec(i+4));
  }
   for (int i = 0; i < 4; i++)
  {
    cost += 0.1*(x.at(i) - opt_ref(i)) * (x.at(i) - opt_ref(i)) + 0.1*(x.at(i+4) - opt_ref(i+4)) * (x.at(i+4) - opt_ref(i+4));
  }
  //std::cout<<"cost:"<<cost<<std::endl;
  //robot_model->updateRobotModel(joint_positions_former);
  //std::cout<<"update2"<<std::endl;
  // if(!robot_model->stabilityCheck(planner->getPlanVerbose()))
  //   {
  //     invalid_cnt ++;
  //     std::stringstream ss;
  //     for(const auto& angle: x) ss << angle << ", ";
  //     if(planner->getPlanVerbose()) ROS_WARN_STREAM("nlopt, robot stability is invalid with gimbals: " << ss.str() << " (cnt: " << invalid_cnt << ")");
  //     return 0;
  //   }

  // invalid_cnt = 0;

  // Eigen::VectorXd force_v = robot_model->getStaticThrust();
  // double average_force = force_v.sum() / force_v.size();
  // double variant = 0;

  // for(int i = 0; i < force_v.size(); i++)
  //   variant += ((force_v(i) - average_force) * (force_v(i) - average_force));

  // variant = sqrt(variant / force_v.size());

  return cost;
}

void nmpc::HydrusXiNMPC::allocateToXU(const tf::Vector3& ref_pos_i, const tf::Vector3& ref_vel_i,
                                         const tf::Quaternion& ref_quat_ib, const tf::Vector3& ref_omega_b,
                                         const VectorXd& ref_wrench_b, vector<double>& x, vector<double>& u)
{
  x.at(0) = ref_pos_i.x();
  x.at(1) = ref_pos_i.y();
  x.at(2) = ref_pos_i.z();
  x.at(3) = ref_vel_i.x();
  x.at(4) = ref_vel_i.y();
  x.at(5) = ref_vel_i.z();
  x.at(6) = ref_quat_ib.w();
  x.at(7) = ref_quat_ib.x();
  x.at(8) = ref_quat_ib.y();
  x.at(9) = ref_quat_ib.z();
  x.at(10) = ref_omega_b.x();
  x.at(11) = ref_omega_b.y();
  x.at(12) = ref_omega_b.z();
  x.at(13) = joint_angles_[0];
  x.at(14) = joint_angles_[1]; 
  x.at(15) = joint_angles_[2];


  // ========= 0) if one rotor is fixed, do it and finish. ======
  if (is_set_fix_rotor_)
  {
    if (ros::Time::now() - fix_rotor_msg_.header.stamp > ros::Duration(0.1))
    {
      ROS_INFO_THROTTLE(1, "No FixRotor msg for 0.1s. Recover to the normal allocation state.");
      is_set_fix_rotor_ = false;
    }

    allocateToXUwOneFixedRotor(fix_rotor_msg_.rotor_id, fix_rotor_msg_.fix_ft, fix_rotor_msg_.fix_alpha, ref_wrench_b,
                               x, u);
    return;
  }
  // =============================================================

  // 1) do one allocation
  //Eigen::VectorXd x_lambda = alloc_mat_pinv_ * ref_wrench_b;
  KDL::JntArray joint_positions = robot_model_->getJointPositions();
  robot_model_for_plan_->updateRobotModel(joint_positions);
  robot_model_for_plan_->setDesiredWrench(ref_wrench_b);
  double max_f = 0.0;
  thrust_gimbal_nl_solver_->set_min_objective(leastSquare, robot_model_for_plan_.get());
  thrust_gimbal_nl_solver_->set_xtol_rel(1e-4); //1e-4
  thrust_gimbal_nl_solver_->set_maxeval(1000);
  std::vector<double> opt_lower_bounds_(8);
  std::vector<double> opt_upper_bounds_(8);
  for (int i = 0; i < 4; i++)
  {
    opt_lower_bounds_[i] = 0.0;
    opt_upper_bounds_[i] = 25.0;
  }
  for (int i = 4; i < 8; i++)
  {
    opt_lower_bounds_[i] = -6.28318; 
    opt_upper_bounds_[i] = 6.28318;
  }
  thrust_gimbal_nl_solver_->set_lower_bounds(opt_lower_bounds_);
  thrust_gimbal_nl_solver_->set_upper_bounds(opt_upper_bounds_);
  tf::Matrix3x3 cog(ref_quat_ib);
  Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
      R(i, j) = cog[i][j];
  }
  robot_model_for_plan_->setRotationMatrix(R);
  Eigen::VectorXd thrust_cmds = Eigen::VectorXd::Zero(4);
  for (int i = 0; i < 4; i++)
    thrust_cmds[i] = getCommand(i);

  robot_model_for_plan_->setThrustCmd(thrust_cmds);

  if (count_%2 == 0)
  {
    try
    {
      std::cout<<ros::Time::now().toSec()<<" Start nlopt optimization!"<<std::endl;
      nlopt::result result = thrust_gimbal_nl_solver_->optimize(opt_result_, max_f);

      // cnt = 0;
      // invalid_cnt = 0;
    }
    catch(std::exception &e)
    {
      std::cout << "nlopt failed: " << e.what() << std::endl;
    }
      std::cout<<ros::Time::now().toSec()<<" End nlopt optimization!"<<std::endl;
    if (count_%20 == 0)
    {
      for (int i = 0; i < 8; i++)
    {
      std::cout << "opt_result_[" << i << "]=" << opt_result_[i] << ", ";
    }
    }
    //std::cout <<"count: "<<count_<< std::endl;
    
    Eigen::MatrixXd A = robot_model_for_plan_->calcWrenchMatrixOnCoG();
    Eigen::MatrixXd B = robot_model_->calcWrenchMatrixOnCoG();
   // std::cout<<"update1"<<A<<std::endl;
    Eigen::VectorXd thrusts = Eigen::VectorXd::Zero(4);
    for(int i = 0; i < 4; i++)
      thrusts(i) = opt_result_[i];
    Eigen::VectorXd desired_wrench = A * thrusts;
    //Eigen::VectorXd desired_wrench = A * thrusts;
    desired_wrench.segment(0, 3) = R * desired_wrench.segment(0, 3); // to world frame
    //Eigen::VectorXd desired_wrench = robot_model->getDesiredWrench();

    std::cout<<"desired_wrench:"<<desired_wrench.transpose()<<std::endl;
    //count_ = 0;
  }
  Eigen::MatrixXd A = robot_model_for_plan_->calcWrenchMatrixOnCoG();
  Eigen::VectorXd thrusts = Eigen::VectorXd::Zero(4);
  for(int i = 0; i < 4; i++)
    thrusts(i) = opt_result_[i];
  Eigen::VectorXd desired_wrench = A * thrusts;
  target_wrench_cog_ = desired_wrench;
  count_++;
  std::vector<double> ft_ref_vec(motor_num_);
  std::vector<double> g_ref_vec(gimbal_num_);
 // std::cout<<"aaa"<<target_external_wrench_<<std::endl;

  // if (motor_num_ != joint_num_)
  // {
  //   ROS_ERROR("motor_num_ is not equal to joint_num_! Cannot allocate to X and U!");
  //   throw std::runtime_error("motor_num_ is not equal to joint_num_! Cannot allocate to X and U!");
  // }

  //  try
  //   {
  //     //nlopt::result result = vectoring_nl_solver_->optimize(opt_gimbal_angles_, max_f);

  //     double roll,pitch,yaw;
  //     robot_model_for_plan_->getCogDesireOrientation<KDL::Rotation>().GetRPY(roll, pitch, yaw);

  //     if(prev_opt_gimbal_angles_.size() == 0) prev_opt_gimbal_angles_ = opt_gimbal_angles_;

  //     if(plan_verbose_)
  //       {
  //         std::cout << "nlopt: " << std::setprecision(7)
  //                   << ros::Time::now().toSec() - start_time  <<  "[sec], cnt: " << cnt;
  //         std::cout << ", found optimal gimbal angles: ";
  //         for(auto it: opt_gimbal_angles_) std::cout << std::setprecision(5) << it << " ";
  //         std::cout << ", max min yaw: " << max_min_yaw_;
  //         std::cout << ", fc t min: " << robot_model_for_plan_->getFeasibleControlTMin();
  //         std::cout << ", atttidue: [" << roll << ", " << pitch;
  //         std::cout << "], force: [" << robot_model_for_plan_->getStaticThrust().transpose();
  //         std::cout << "]" << std::endl;
  //       }


  //     cnt = 0;
  //     invalid_cnt = 0;
  //   }
  // catch(std::exception &e)
  //   {
  //     std::cout << "nlopt failed: " << e.what() << std::endl;
  //   }

  g_ref_vec[0] = 0.0;
  g_ref_vec[1] = 0.0;
  g_ref_vec[2] = 0.0;
  g_ref_vec[3] = 0.0;
  ft_ref_vec[0] = 8.30;
  ft_ref_vec[1] = 10.30;
  ft_ref_vec[2] = 10.30;
  ft_ref_vec[3] = 8.30;
  for (int i = 0; i < motor_num_; i++)
  {
    // ft_ref_vec[i] = 9.30;
   
    //u.at(i) = ft_ref_vec[i];
    u.at(i) = opt_result_[i];
    //x.at(16 + i) = ensureOneServoContinuity(g_ref_vec[i], i);
    //x.at(16 + i) = g_ref_vec[i];
    x.at(16 + i) = opt_result_[i+4];
    u.at(i + 4) = g_ref_vec[i];
  }

  aerial_robot_msgs::PredXU x_u_ref_msg;
  x_u_ref_msg.header.stamp = ros::Time::now();
  for (int i = 0; i < x.size(); i++)
    x_u_ref_msg.x.data.push_back(x[i]);
  for (int i = 0; i < u.size(); i++)
    x_u_ref_msg.u.data.push_back(u[i]);
  pub_x_u_ref_.publish(x_u_ref_msg);
  //std::cout<<x_u_ref_msg<<std::endl;
  if (alloc_type_ == 0)
    return;

//   // 2) check if one rotor's thrust is less than threshold and flip backwards
//   std::vector<int> rotor_idx_vec;
//   for (int i = 0; i < motor_num_; i++)
//   {
//     if (ft_ref_vec[i] > ft_thresh_)
//       continue;

//     if (a_ref_vec[i] >= -M_PI_2 && a_ref_vec[i] <= M_PI_2)
//       continue;

//     rotor_idx_vec.push_back(i);
//   }

//   if (rotor_idx_vec.empty())
//     return;

//   int rotor_idx;
//   if (rotor_idx_vec.size() > 1)
//   {
//     double max_ft = 0.0;
//     int max_rotor_idx = -1;
//     for (const auto& idx : rotor_idx_vec)
//     {
//       if (ft_ref_vec[idx] > max_ft)
//       {
//         max_ft = ft_ref_vec[idx];
//         max_rotor_idx = idx;
//       }
//     }
//     rotor_idx = max_rotor_idx;

//     ROS_WARN_THROTTLE(1.0,
//                       "More than one rotor is below threshold and flip backwards! "
//                       "Select rotor %d with thrust %.2f as the fixed rotor.",
//                       rotor_idx, max_ft);
//   }
//   else
//   {
//     rotor_idx = rotor_idx_vec.at(0);
//   }

//   // 3) if rotor_idx is not empty, maintain the thrust and modify the angle
//   double ft_stop_rotor = ft_ref_vec[rotor_idx];
//   double alpha_stop_rotor = M_PI_2 - acos(x_lambda(2 * rotor_idx) / ft_thresh_);

//   // 4) re-alloc
//   allocateToXUwOneFixedRotor(rotor_idx, ft_stop_rotor, alpha_stop_rotor, ref_wrench_b, x, u);
// }
}

void nmpc::HydrusXiNMPC::allocateToXUwOneFixedRotor(int fix_rotor_idx, double fix_ft, double fix_alpha,
                                                       const VectorXd& ref_wrench_b, vector<double>& x,
                                                       vector<double>& u)
{
  double fix_ft_x = fix_ft * sin(fix_alpha);
  double fix_ft_y = fix_ft * cos(fix_alpha);

  // 1) construct tgt_wrench from z_from_rotor
  Eigen::VectorXd z_from_rotor = Eigen::VectorXd::Zero(motor_num_ * 2);
  z_from_rotor(2 * fix_rotor_idx) = fix_ft_x;
  z_from_rotor(2 * fix_rotor_idx + 1) = fix_ft_y;
  Eigen::VectorXd tgt_wrench_from_rotor = alloc_mat_ * z_from_rotor;

  // 2) calculate alloc_mat with this rotor's contribution
  Eigen::VectorXd tgt_wrench_modified = ref_wrench_b - tgt_wrench_from_rotor;

  // 3) calculate the allocation matrix without this rotor, which is 6*6
  if (fix_rotor_idx != rotor_idx_prev_)
  {
    Eigen::MatrixXd alloc_mat_del_rotor(alloc_mat_.rows(), alloc_mat_.cols() - 2);
    int j = 0;
    for (int k = 0; k < alloc_mat_.cols(); ++k)
    {
      if (k == 2 * fix_rotor_idx || k == 2 * fix_rotor_idx + 1)
        continue;
      alloc_mat_del_rotor.col(j++) = alloc_mat_.col(k);
    }
    alloc_mat_del_rotor_inv_ = alloc_mat_del_rotor.inverse();
  }

  // 4) reconstruct the z output
  Eigen::VectorXd z_except_rotor = alloc_mat_del_rotor_inv_ * tgt_wrench_modified;

  // 5) at the place of 2*fix_rotor_idx, insert 2 numbers to z_except_rotor
  Eigen::VectorXd z_final(motor_num_ * 2);
  z_final.head(2 * fix_rotor_idx) = z_except_rotor.head(2 * fix_rotor_idx);
  z_final(2 * fix_rotor_idx) = fix_ft_x;
  z_final(2 * fix_rotor_idx + 1) = fix_ft_y;
  z_final.tail(z_except_rotor.size() - 2 * fix_rotor_idx) =
      z_except_rotor.tail(z_except_rotor.size() - 2 * fix_rotor_idx);

  // 6) reconstruct the thrust and servo angle
  // check motor_num_ == joint_num_ before this function is called
  if (motor_num_ != joint_num_)
  {
    ROS_ERROR("motor_num_ is not equal to joint_num_! Cannot allocate to X and U!");
    throw std::runtime_error("motor_num_ is not equal to joint_num_! Cannot allocate to X and U!");
  }
  for (int i = 0; i < motor_num_; i++)
  {
    const double ft = sqrt(z_final(2 * i) * z_final(2 * i) + z_final(2 * i + 1) * z_final(2 * i + 1));
    u.at(i) = ft;
    const double alpha = atan2(z_final(2 * i), z_final(2 * i + 1));
    x.at(13 + i) = ensureOneServoContinuity(alpha, i);
  }

  // if the fixed rotor is the same with previous one, no need to recalculate the allocation matrix.
  rotor_idx_prev_ = fix_rotor_idx;
}

/**
 * @brief callbackViz: publish the predicted trajectory and reference trajectory
 * @param [ros::TimerEvent&] event
 */
void nmpc::HydrusXiNMPC::callbackViz(const ros::TimerEvent& event)
{
  // from mpc_solver_ptr_->x_u_out to PoseArray
  geometry_msgs::PoseArray pred_poses;
  geometry_msgs::PoseArray ref_poses;

  int& NN = mpc_solver_ptr_->NN_;
  int& NX = mpc_solver_ptr_->NX_;

  for (int i = 0; i < NN; ++i)
  {
    geometry_msgs::Pose pred_pose;
    pred_pose.position.x = mpc_solver_ptr_->xo_[i][0];
    pred_pose.position.y = mpc_solver_ptr_->xo_[i][1];
    pred_pose.position.z = mpc_solver_ptr_->xo_[i][2];
    pred_pose.orientation.w = mpc_solver_ptr_->xo_[i][6];
    pred_pose.orientation.x = mpc_solver_ptr_->xo_[i][7];
    pred_pose.orientation.y = mpc_solver_ptr_->xo_[i][8];
    pred_pose.orientation.z = mpc_solver_ptr_->xo_[i][9];
    pred_poses.poses.push_back(pred_pose);

    geometry_msgs::Pose ref_pose;
    ref_pose.position.x = mpc_solver_ptr_->xr_[i][0];
    ref_pose.position.y = mpc_solver_ptr_->xr_[i][1];
    ref_pose.position.z = mpc_solver_ptr_->xr_[i][2];
    ref_pose.orientation.w = mpc_solver_ptr_->xr_[i][6];
    ref_pose.orientation.x = mpc_solver_ptr_->xr_[i][7];
    ref_pose.orientation.y = mpc_solver_ptr_->xr_[i][8];
    ref_pose.orientation.z = mpc_solver_ptr_->xr_[i][9];
    ref_poses.poses.push_back(ref_pose);
  }

  pred_poses.header.frame_id = "world";
  pred_poses.header.stamp = ros::Time::now();
  pub_viz_pred_.publish(pred_poses);

  ref_poses.header.frame_id = "world";
  ref_poses.header.stamp = ros::Time::now();
  pub_viz_ref_.publish(ref_poses);
}

void nmpc::HydrusXiNMPC::callbackJointStates(const sensor_msgs::JointStateConstPtr& msg)
{
  for (int i = 0; i < gimbal_num_; i++)
    gimbal_angles_[i] = msg->position[i];
  for (int i = 0; i < joint_num_; i++)
    joint_angles_[i] = msg->position[i + gimbal_num_];

}

/* TODO: this function is just for test. We may need a more general function to set all kinds of state */
void nmpc::HydrusXiNMPC::callbackSetRPY(const spinal::DesireCoordConstPtr& msg)
{
  // add a check to avoid the singular point for euler angle
  if (msg->pitch == M_PI / 2.0 or msg->pitch == -M_PI / 2.0)
  {
    ROS_WARN(
        "The pitch angle is set to PI/2 or -PI/2, which is a singular point for euler angle."
        " Please set other values for the pitch angle.");
    return;
  }

  navigator_->setTargetRoll(msg->roll);
  navigator_->setTargetPitch(msg->pitch);
  navigator_->setTargetYaw(msg->yaw);
}

/* TODO: this function should be combined with the inner planning framework */
void nmpc::HydrusXiNMPC::callbackSetRefXU(const aerial_robot_msgs::PredXUConstPtr& msg)
{
  /* failsafe check */
  if (navigator_->getNaviState() != aerial_robot_navigation::HOVER_STATE)
  {
    ROS_WARN_THROTTLE(1, "The robot has not hovered, so the reference trajectory will be ignored!");
    return;
  }

  /* switch tracking mode */
  if (!is_traj_tracking_)
  {
    is_traj_tracking_ = true;
    traj_child_frame_id_ = msg->child_frame_id;
    ROS_INFO_STREAM("Trajectory tracking mode is on! The child frame is set to " << traj_child_frame_id_ << ".");
  }

  /* receive info */
  x_u_ref_ = *msg;

  /* set reference */
  rosXU2VecXU(x_u_ref_, mpc_solver_ptr_->xr_, mpc_solver_ptr_->ur_);
  mpc_solver_ptr_->setReference(mpc_solver_ptr_->xr_, mpc_solver_ptr_->ur_, true);
}

void nmpc::HydrusXiNMPC::callbackSetRefTraj(const trajectory_msgs::MultiDOFJointTrajectoryConstPtr& msg)
{
  if (msg->points.size() != mpc_solver_ptr_->NN_ + 1)
    ROS_WARN("The length of the trajectory is not equal to the prediction horizon! Cannot use the trajectory!");

  if (navigator_->getNaviState() != aerial_robot_navigation::HOVER_STATE)
  {
    ROS_WARN_THROTTLE(1, "The robot has not hovered, so the reference trajectory will be ignored!");
    return;
  }

  /* For set-point regulation, if the traj planner sends the same traj, we can skip the calculation of allocation. */
  // check if two trajectories are the same
  int max_same_idx = 0;
  if (!last_traj_msg_.points.empty())  // check if the last trajectory is empty
  {
    for (int i = 0; i < msg->points.size(); i++)  // only check the first NN points
    {
      if (isMulDOFJointTrajPtEqual(msg->points[i], last_traj_msg_.points[i], false))  // time is not equal
        max_same_idx = i;
      else
        break;
    }
  }

  if (max_same_idx != msg->points.size() - 1 || is_set_fix_rotor_ == true)
  {
    for (int i = 0; i < mpc_solver_ptr_->NN_ + 1; i++)
    {
      const trajectory_msgs::MultiDOFJointTrajectoryPoint& point = msg->points[i];
      geometry_msgs::Vector3 pos = point.transforms[0].translation;
      geometry_msgs::Vector3 vel = point.velocities[0].linear;
      geometry_msgs::Vector3 acc = point.accelerations[0].linear;
      geometry_msgs::Quaternion quat = point.transforms[0].rotation;
      geometry_msgs::Vector3 omega = point.velocities[0].angular;
      geometry_msgs::Vector3 ang_acc = point.accelerations[0].angular;
      setXrUrRef(tf::Vector3(pos.x, pos.y, pos.z), tf::Vector3(vel.x, vel.y, vel.z), tf::Vector3(acc.x, acc.y, acc.z),
                 tf::Quaternion(quat.x, quat.y, quat.z, quat.w), tf::Vector3(omega.x, omega.y, omega.z),
                 tf::Vector3(ang_acc.x, ang_acc.y, ang_acc.z), tf::Vector3(0, 0, 0), tf::Vector3(0, 0, 0), i);
    }
  }

  x_u_ref_.header.stamp = msg->header.stamp;
  x_u_ref_.child_frame_id = msg->joint_names[0];
  callbackSetRefXU(boost::make_shared<const aerial_robot_msgs::PredXU>(x_u_ref_));

  last_traj_msg_ = *msg;
}

void nmpc::HydrusXiNMPC::callbackSetFixedRotor(const aerial_robot_msgs::FixRotorConstPtr& msg)
{
  // failsafe
  if (msg->rotor_id < 0 || msg->rotor_id >= motor_num_)
  {
    ROS_WARN_STREAM("The rotor_id " << static_cast<int>(msg->rotor_id)
                                    << " is incorrect. Note that the id starts from 0.");
    return;
  }

  if (msg->fix_ft < thrust_ctrl_min_ || msg->fix_ft > thrust_ctrl_max_)
  {
    ROS_WARN_STREAM("The fix_ft value " << msg->fix_ft << " is out of range. It should be between " << thrust_ctrl_min_
                                        << " and " << thrust_ctrl_max_ << ".");
    return;
  }

  if (msg->fix_alpha < joint_angle_min_ || msg->fix_alpha > joint_angle_max_)
  {
    ROS_WARN_STREAM("The fix_alpha value " << msg->fix_alpha << " is out of range. It should be between "
                                           << joint_angle_min_ << " and " << joint_angle_max_ << ".");
    return;
  }

  // set values
  is_set_fix_rotor_ = true;
  fix_rotor_msg_ = *msg;
}

void nmpc::HydrusXiNMPC::callbackTargetExternalWrench(const geometry_msgs::WrenchStampedConstPtr& msg)
{
  target_external_wrench_ = *msg;
  time_last_target_external_wrench_ = ros::Time::now();
}
// void nmpc::HydrusXiNMPC::callbackRealThrust(const aerial_robot_msgs::PredXUConstPtr& msg)
// {
//   /* failsafe check */
//   if (navigator_->getNaviState() != aerial_robot_navigation::HOVER_STATE)
//   {
//     ROS_WARN_THROTTLE(1, "The robot has not hovered, so the reference trajectory will be ignored!");
//     return;
//   }

//   /* switch tracking mode */
//   if (!is_traj_tracking_)
//   {
//     is_traj_tracking_ = true;
//     traj_child_frame_id_ = msg->child_frame_id;
//     ROS_INFO_STREAM("Trajectory tracking mode is on! The child frame is set to " << traj_child_frame_id_ << ".");
//   }

//   /* receive info */
//   x_u_ref_ = *msg;

//   /* set reference */
//   rosXU2VecXU(x_u_ref_, mpc_solver_ptr_->xr_, mpc_solver_ptr_->ur_);
//   mpc_solver_ptr_->setReference(mpc_solver_ptr_->xr_, mpc_solver_ptr_->ur_, true);
// }

void nmpc::HydrusXiNMPC::cfgNMPCCallback(NMPCConfig& config, uint32_t level)
{
  using Levels = aerial_robot_msgs::DynamicReconfigureLevels;
  if (config.nmpc_flag)
  {
    try
    {
      switch (level)
      {
        case Levels::RECONFIGURE_NMPC_Q_P_XY: {
          mpc_solver_ptr_->setCostWDiagElement(0, config.Qp_xy);
          mpc_solver_ptr_->setCostWDiagElement(1, config.Qp_xy);

          ROS_INFO_STREAM("change Qp_xy for NMPC '" << config.Qp_xy << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_Q_P_Z: {
          mpc_solver_ptr_->setCostWDiagElement(2, config.Qp_z);
          ROS_INFO_STREAM("change Qp_z for NMPC '" << config.Qp_z << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_Q_V_XY: {
          mpc_solver_ptr_->setCostWDiagElement(3, config.Qv_xy);
          mpc_solver_ptr_->setCostWDiagElement(4, config.Qv_xy);
          ROS_INFO_STREAM("change Qv_xy for NMPC '" << config.Qv_xy << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_Q_V_Z: {
          mpc_solver_ptr_->setCostWDiagElement(5, config.Qv_z);
          ROS_INFO_STREAM("change Qv_z for NMPC '" << config.Qv_z << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_Q_Q_XY: {
          mpc_solver_ptr_->setCostWDiagElement(7, config.Qq_xy);
          mpc_solver_ptr_->setCostWDiagElement(8, config.Qq_xy);
          ROS_INFO_STREAM("change Qq_xy for NMPC '" << config.Qq_xy << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_Q_Q_Z: {
          mpc_solver_ptr_->setCostWDiagElement(9, config.Qq_z);
          ROS_INFO_STREAM("change Qq_z for NMPC '" << config.Qq_z << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_Q_W_XY: {
          mpc_solver_ptr_->setCostWDiagElement(10, config.Qw_xy);
          mpc_solver_ptr_->setCostWDiagElement(11, config.Qw_xy);
          ROS_INFO_STREAM("change Qw_xy for NMPC '" << config.Qw_xy << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_Q_W_Z: {
          mpc_solver_ptr_->setCostWDiagElement(12, config.Qw_z);
          ROS_INFO_STREAM("change Qw_z for NMPC '" << config.Qw_z << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_Q_A: {
          for (int i = 13; i < 13 + joint_num_; ++i)
            mpc_solver_ptr_->setCostWDiagElement(i, config.Qa);
          ROS_INFO_STREAM("change Qa for NMPC '" << config.Qa << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_R_T: {
          for (int i = mpc_solver_ptr_->NX_; i < mpc_solver_ptr_->NX_ + motor_num_; ++i)
            mpc_solver_ptr_->setCostWDiagElement(i, config.Rt, false);
          ROS_INFO_STREAM("change Rt for NMPC '" << config.Rt << "'");
          break;
        }
        case Levels::RECONFIGURE_NMPC_R_AC_D: {
          for (int i = mpc_solver_ptr_->NX_ + motor_num_; i < mpc_solver_ptr_->NX_ + motor_num_ + joint_num_; ++i)
            mpc_solver_ptr_->setCostWDiagElement(i, config.Rac_d, false);
          ROS_INFO_STREAM("change Rac_d for NMPC '" << config.Rac_d << "'");
          break;
        }
        default: {
          ROS_INFO_STREAM("The setting variable is not in the list!");
          break;
        }
      }
    }
    catch (std::invalid_argument& e)
    {
      ROS_ERROR_STREAM("NMPC config failed: " << e.what());
    }
  }
}

double nmpc::HydrusXiNMPC::getCommand(int idx_u, double T_horizon) const
{
  if (T_horizon == 0)
    return mpc_solver_ptr_->uo_.at(0).at(idx_u);

  return mpc_solver_ptr_->uo_.at(0).at(idx_u) +
         T_horizon / t_nmpc_step_ * (mpc_solver_ptr_->uo_.at(1).at(idx_u) - mpc_solver_ptr_->uo_.at(0).at(idx_u));
}

std::vector<double> nmpc::HydrusXiNMPC::meas2VecX(bool is_modified_by_traj_frame)
{
  vector<double> bx0(mpc_solver_ptr_->NBX0_, 0);

  tf::Vector3 pos = estimator_->getPos(Frame::COG, estimate_mode_);
  tf::Vector3 vel = estimator_->getVel(Frame::COG, estimate_mode_);
  tf::Quaternion quat = estimator_->getQuat(Frame::COG, estimate_mode_);
  tf::Vector3 ang_vel = estimator_->getAngularVel(Frame::COG, estimate_mode_);

  // === check the sign of the quaternion, avoid the flip of the quaternion. ===
  // This is quite important because of the warm-starting of the NMPC solver. The quaternion should be continuous.
  double qe_c_w =
      quat.w() * quat_prev_.w() + quat.x() * quat_prev_.x() + quat.y() * quat_prev_.y() + quat.z() * quat_prev_.z();
  if (qe_c_w < 0)
  {
    quat = quat.operator-();
  }

  quat_prev_ = quat;

  // === for reference, we may need to convert the position and velocity to the end-effector frame ===
  if (is_modified_by_traj_frame && traj_child_frame_id_ != "cog")
  {
    if (traj_child_frame_id_ == "ee")
    {
      // convert the position and velocity from CoG to end-effector frame
      tf::Vector3 target_ee_pos, target_ee_vel, target_ee_omega;
      tf::Quaternion target_ee_quat;
      robot_model_->convertFromCoGToEEContact(pos, vel, quat, ang_vel, target_ee_pos, target_ee_vel, target_ee_quat,
                                              target_ee_omega);

      pos = target_ee_pos;
      vel = target_ee_vel;
      quat = target_ee_quat;
      ang_vel = target_ee_omega;
    }
    else
    {
      ROS_WARN("Unsupported traj_child_frame_id_. Only support cog or ee. Use cog information instead!");
    }
  }

  // === fill the vector ===
  bx0[0] = pos.x();
  bx0[1] = pos.y();
  bx0[2] = pos.z();
  bx0[3] = vel.x();
  bx0[4] = vel.y();
  bx0[5] = vel.z();
  bx0[6] = quat.w();
  bx0[7] = quat.x();
  bx0[8] = quat.y();
  bx0[9] = quat.z();
  bx0[10] = ang_vel.x();
  bx0[11] = ang_vel.y();
  bx0[12] = ang_vel.z();
  for (int i = 0; i < joint_num_; i++)
    bx0[13 + i] = joint_angles_[i];
  // bx0[16] = 3.14159;
  // bx0[17] = 0.0;
  // bx0[18] = 3.14159;
  // bx0[19] = 0.0;

  for (int i = 0; i < gimbal_num_; i++)
    bx0[16 + i] = gimbal_angles_[i];
  // for (int i = 0; i < bx0.size(); i++)
  //   std::cout<<"bx0[" << i << "]: " << bx0[i]<<std::endl;
  // for (int i = 0; i < joint_angles_.size(); i++)
  //   std::cout<<"joint_angles_[" << i << "]: " << joint_angles_[i]<<std::endl;
  return bx0;
}

void nmpc::HydrusXiNMPC::externalWrenchEstimate()
{
  if(navigator_->getNaviState() != aerial_robot_navigation::HOVER_STATE &&
     navigator_->getNaviState() != aerial_robot_navigation::LAND_STATE)
    {
      prev_est_wrench_timestamp_ = 0;
      integrate_term_ = Eigen::VectorXd::Zero(6);
      return;
    }

  Eigen::Vector3d vel_w, omega_cog; // workaround: use the filtered value
  // auto imu_handler = boost::dynamic_pointer_cast<sensor_plugin::HydrusImu>(estimator_->getImuHandler(0));
  // if (!imu_handler)
  // {
  //   ROS_ERROR("HydrusImu is null!");
  //   return;
  // }

  tf::vectorTFToEigen(estimator_->getVel(Frame::COG, estimate_mode_), vel_w);
  tf::vectorTFToEigen(estimator_->getAngularVel(Frame::COG, estimate_mode_), omega_cog);

  Eigen::Matrix3d cog_rot;
  tf::matrixTFToEigen(estimator_->getOrientation(Frame::COG, estimate_mode_), cog_rot);
  Eigen::Matrix3d inertia = robot_model_->getInertia<Eigen::Matrix3d>();
  double mass = robot_model_->getMass();

  Eigen::VectorXd sum_momentum = Eigen::VectorXd::Zero(6);
  sum_momentum.head(3) = mass * vel_w;
  sum_momentum.tail(3) = inertia * omega_cog;
  Eigen::VectorXd sum_force = Eigen::VectorXd::Zero(6);
  //std::cout<<"acc:"<<acc_w<<std::endl;
  // sum_force.head(3) = mass * acc_w;

  Eigen::MatrixXd J_t = Eigen::MatrixXd::Identity(6,6);
  J_t.topLeftCorner(3,3) = cog_rot;
  Eigen::VectorXd N = mass * robot_model_->getGravity();

  N.tail(3) = aerial_robot_model::skew(omega_cog) * (inertia * omega_cog);
  Eigen::VectorXd target_wrench_cog = getTargetWrenchCog();
  momentum_observer_matrix_ = Eigen::MatrixXd::Identity(6,6) * 3.0; //0.1 good for hydrus xi
  if(prev_est_wrench_timestamp_ == 0)
    {
      prev_est_wrench_timestamp_ = ros::Time::now().toSec();
      init_sum_momentum_ = sum_momentum; // not good
    }
  double dt = ros::Time::now().toSec() - prev_est_wrench_timestamp_;
  integrate_term_ += (J_t * target_wrench_cog - N + est_external_wrench_ - sum_force) * dt;
  est_external_wrench_ = momentum_observer_matrix_ * (sum_momentum - init_sum_momentum_ - integrate_term_);
  Eigen::VectorXd est_external_wrench_cog = est_external_wrench_;
  est_external_wrench_cog.head(3) = cog_rot.inverse() * est_external_wrench_.head(3);
  geometry_msgs::WrenchStamped wrench_msg;
  wrench_msg.header.stamp.fromSec(estimator_->getImuLatestTimeStamp());
  wrench_msg.wrench.force.x = est_external_wrench_(0);
  wrench_msg.wrench.force.y = est_external_wrench_(1);
  wrench_msg.wrench.force.z = est_external_wrench_(2);
  wrench_msg.wrench.torque.x = est_external_wrench_(3);
  wrench_msg.wrench.torque.y = est_external_wrench_(4);
  wrench_msg.wrench.torque.z = est_external_wrench_(5);
  pub_estimate_external_wrench_.publish(wrench_msg);
  prev_est_wrench_timestamp_ = ros::Time::now().toSec();
}

double nmpc::HydrusXiNMPC::ensureOneServoContinuity(double a_ref, int idx) const
{
  double a_now = gimbal_ctrl_cmd_.position[idx];
  // ensure the servo angle is continuous
  if (a_ref - a_now > M_PI)
    a_ref -= 2 * M_PI;
  else if (a_ref - a_now < -M_PI)
    a_ref += 2 * M_PI;

  return a_ref;
}

std::vector<double> nmpc::HydrusXiNMPC::ensureAllServoContinuity(std::vector<double>& a_ref_vec) const
{
  for (int i = 0; i < gimbal_num_; i++)
    a_ref_vec[i] = ensureOneServoContinuity(a_ref_vec[i], i);

  return a_ref_vec;
}

void nmpc::HydrusXiNMPC::printPhysicalParams()
{
  cout << "mass: " << robot_model_->getMass() << endl;
  cout << "gravity: " << robot_model_->getGravity() << endl;
  cout << "inertia: " << robot_model_->getInertia<Eigen::Matrix3d>() << endl;
  cout << "rotor num: " << robot_model_->getRotorNum() << endl;
  for (const auto& dir : robot_model_->getRotorDirection())
  {
    std::cout << "Key: " << dir.first << ", Value: " << dir.second << std::endl;
  }
  for (const auto& vec : robot_model_->getRotorsOriginFromCog<Eigen::Vector3d>())
  {
    std::cout << "rotor origin from cog: " << vec << std::endl;
  }
  cout << "thrust lower limit: " << robot_model_->getThrustLowerLimit() << endl;
  cout << "thrust upper limit: " << robot_model_->getThrustUpperLimit() << endl;

  cout << "kq_kt_rate" << robot_model_->getMFRate() << endl;
  cout << "abs(kq_kt_rate)" << abs(robot_model_->getMFRate()) << endl;
}

bool nmpc::HydrusXiNMPC::isMulDOFJointTrajPtEqual(const trajectory_msgs::MultiDOFJointTrajectoryPoint& a,
                                                     const trajectory_msgs::MultiDOFJointTrajectoryPoint& b,
                                                     bool if_compare_time, double epsilon)
{
  if (a.transforms.size() != b.transforms.size() || a.velocities.size() != b.velocities.size() ||
      a.accelerations.size() != b.accelerations.size())
    return false;

  for (size_t i = 0; i < a.transforms.size(); ++i)
  {
    const auto& ta = a.transforms[i];
    const auto& tb = b.transforms[i];
    if (!isAlmostEqual(ta.translation.x, tb.translation.x, epsilon) ||
        !isAlmostEqual(ta.translation.y, tb.translation.y, epsilon) ||
        !isAlmostEqual(ta.translation.z, tb.translation.z, epsilon) ||
        !isAlmostEqual(ta.rotation.x, tb.rotation.x, epsilon) ||
        !isAlmostEqual(ta.rotation.y, tb.rotation.y, epsilon) ||
        !isAlmostEqual(ta.rotation.z, tb.rotation.z, epsilon) || !isAlmostEqual(ta.rotation.w, tb.rotation.w, epsilon))
      return false;
  }

  for (size_t i = 0; i < a.velocities.size(); ++i)
  {
    const auto& va = a.velocities[i];
    const auto& vb = b.velocities[i];
    if (!isAlmostEqual(va.linear.x, vb.linear.x, epsilon) || !isAlmostEqual(va.linear.y, vb.linear.y, epsilon) ||
        !isAlmostEqual(va.linear.z, vb.linear.z, epsilon) || !isAlmostEqual(va.angular.x, vb.angular.x, epsilon) ||
        !isAlmostEqual(va.angular.y, vb.angular.y, epsilon) || !isAlmostEqual(va.angular.z, vb.angular.z, epsilon))
      return false;
  }

  for (size_t i = 0; i < a.accelerations.size(); ++i)
  {
    const auto& aa = a.accelerations[i];
    const auto& ab = b.accelerations[i];
    if (!isAlmostEqual(aa.linear.x, ab.linear.x, epsilon) || !isAlmostEqual(aa.linear.y, ab.linear.y, epsilon) ||
        !isAlmostEqual(aa.linear.z, ab.linear.z, epsilon) || !isAlmostEqual(aa.angular.x, ab.angular.x, epsilon) ||
        !isAlmostEqual(aa.angular.y, ab.angular.y, epsilon) || !isAlmostEqual(aa.angular.z, ab.angular.z, epsilon))
      return false;
  }

  // ROS duration comparison
  if (if_compare_time)
  {
    if (!isAlmostEqual(a.time_from_start.toSec(), b.time_from_start.toSec(), epsilon))
      return false;
  }

  return true;
}

/* plugin registration */
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(aerial_robot_control::nmpc::HydrusXiNMPC, aerial_robot_control::ControlBase)
