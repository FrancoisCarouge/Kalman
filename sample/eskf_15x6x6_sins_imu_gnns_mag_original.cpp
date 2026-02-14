// NOLINTBEGIN
// REMOVE

/**
 * Error-State Kalman Filter (ESKF) Sample
 * Style: C++23 (Modules, Concepts, Ranges, Strong Types)
 * Reference: Comparable to https://github.com/FrancoisCarouge/Kalman
 * Dependencies: Eigen3
 *
 * Compilation: g++ -std=c++23 -O3 -I/usr/include/eigen3 eskf_sample.cpp -o eskf
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <random>
#include <ranges>
#include <vector>

// -- Strong Types & Concepts (Library Style) --

namespace kf {

template <typename T>
concept Numeric = std::floating_point<T>;

// Strong types for clarity and safety (Style of FrancoisCarouge/Kalman)
template <int Size> struct state_vector {
  Eigen::Vector<double, Size> value;
  // static constexpr int size = Size;
};

template <int Size> struct error_state_vector {
  Eigen::Vector<double, Size> value;
  // static constexpr int size = Size;
};

template <int Size> struct covariance {
  Eigen::Matrix<double, Size, Size> value;
};

// struct timestamp {
//   double seconds;
// };

} // namespace kf

// -- Physics & Math Constants --

namespace constants {
using namespace std::numbers;
constexpr double gravity_magnitude = 9.81;
const Eigen::Vector3d gravity_vector{0.0, 0.0, gravity_magnitude};
// Earth's magnetic field (approximate normalized vector for simulation)
const Eigen::Vector3d mag_field_inertial =
    Eigen::Vector3d{0.5, 0.0, 0.866}.normalized();
} // namespace constants

// -- The ESKF Implementation --

class error_state_kalman_filter {
public:
  // Nominal State: p(3), v(3), q(4), ba(3), bg(3)
  struct nominal_state_t {
    Eigen::Vector3d p{Eigen::Vector3d::Zero()};
    Eigen::Vector3d v{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d b_a{Eigen::Vector3d::Zero()};
    Eigen::Vector3d b_g{Eigen::Vector3d::Zero()};
  };

  // Error State Indices
  static constexpr int IDX_P = 0;
  static constexpr int IDX_V = 3;
  static constexpr int IDX_TH = 6; // Theta (Angle Error)
  static constexpr int IDX_BA = 9;
  static constexpr int IDX_BG = 12;
  static constexpr int DIM_ERR = 15;

  using ErrorStateVec = Eigen::Vector<double, DIM_ERR>;
  using ErrorCovMat = Eigen::Matrix<double, DIM_ERR, DIM_ERR>;

private:
  nominal_state_t x_;
  ErrorCovMat P_;

  // Process noise covariance blocks
  Eigen::Matrix3d Q_acc_;
  Eigen::Matrix3d Q_gyr_;
  Eigen::Matrix3d Q_ba_;
  Eigen::Matrix3d Q_bg_;

public:
  error_state_kalman_filter()
      : P_{ErrorCovMat::Identity() * 1e-2},
        // Tunable process noise
        Q_acc_{Eigen::Matrix3d::Identity() *
               0.1}, // Accelerometer noise density
        Q_gyr_{Eigen::Matrix3d::Identity() * 0.01}, // Gyro noise density
        Q_ba_{Eigen::Matrix3d::Identity() * 0.001}, // Accel bias random walk
        Q_bg_{Eigen::Matrix3d::Identity() * 0.0001} // Gyro bias random walk
  {}

  void initialize(const nominal_state_t &initial_state,
                  const ErrorCovMat &initial_covariance) {
    x_ = initial_state;
    P_ = initial_covariance;
  }

  const nominal_state_t &nominal_state() const { return x_; }
  // const ErrorCovMat &error_covariance() const { return P_; }

  // -- Prediction Step (IMU) --
  void predict(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m,
               double dt) {
    // 1. Correct IMU measurements with current bias estimates
    Eigen::Vector3d a_unbiased = a_m - x_.b_a;
    Eigen::Vector3d w_unbiased = w_m - x_.b_g;

    // 2. Nominal State Kinematics
    Eigen::Matrix3d R = x_.q.toRotationMatrix();

    // Position update: p = p + v*dt + 0.5*(R*a - g)*dt^2
    Eigen::Vector3d acc_world = R * a_unbiased - constants::gravity_vector;
    x_.p += x_.v * dt + 0.5 * acc_world * dt * dt;

    // Velocity update: v = v + (R*a - g)*dt
    x_.v += acc_world * dt;

    // Attitude update: q = q * Exp(w*dt)
    // Using 0th order integration for quaternion (small angles)
    Eigen::AngleAxisd rotation_vector(w_unbiased.norm() * dt,
                                      w_unbiased.normalized());
    Eigen::Quaterniond delta_q;
    if (w_unbiased.norm() > 1e-8) {
      delta_q = Eigen::Quaterniond(rotation_vector);
    } else {
      delta_q = Eigen::Quaterniond::Identity();
    }
    x_.q = (x_.q * delta_q).normalized();

    // Biases are constant in prediction (Random Walk modeled in P)

    // 3. Error State Jacobian (F_x)
    Eigen::Matrix<double, DIM_ERR, DIM_ERR> F =
        Eigen::Matrix<double, DIM_ERR, DIM_ERR>::Identity();

    // Position blocks
    F.block<3, 3>(IDX_P, IDX_V) = Eigen::Matrix3d::Identity() * dt;

    // Velocity blocks
    // dv/dtheta = -R * [a]x
    Eigen::Matrix3d a_skew;
    a_skew << 0, -a_unbiased.z(), a_unbiased.y(), a_unbiased.z(), 0,
        -a_unbiased.x(), -a_unbiased.y(), a_unbiased.x(), 0;
    F.block<3, 3>(IDX_V, IDX_TH) = -R * a_skew * dt;
    F.block<3, 3>(IDX_V, IDX_BA) = -R * dt; // dv/dba

    // Angle blocks
    // dtheta/dtheta = Transpose(Rot(w*dt)) approx I - [w*dt]x
    // For small dt, often approximated as Identity, but let's be precise
    F.block<3, 3>(IDX_TH, IDX_TH) =
        Eigen::Matrix3d::Identity() - skew(w_unbiased * dt);
    F.block<3, 3>(IDX_TH, IDX_BG) = -Eigen::Matrix3d::Identity() * dt;

    // 4. Process Noise (Fi * Q * Fi') simplified diagonal approximation
    // We inject process noise into Velocity (acc noise), Angle (gyro noise),
    // and Biases
    Eigen::Matrix<double, DIM_ERR, DIM_ERR> Q_total =
        Eigen::Matrix<double, DIM_ERR, DIM_ERR>::Zero();
    Q_total.block<3, 3>(IDX_V, IDX_V) = Q_acc_ * dt * dt;   // V += a * dt
    Q_total.block<3, 3>(IDX_TH, IDX_TH) = Q_gyr_ * dt * dt; // Th += w * dt
    Q_total.block<3, 3>(IDX_BA, IDX_BA) = Q_ba_ * dt;
    Q_total.block<3, 3>(IDX_BG, IDX_BG) = Q_bg_ * dt;

    // 5. Propagate Covariance
    P_ = F * P_ * F.transpose() + Q_total;
  }

  // -- Correction Step: GNSS (Position) --
  void update_gps(const Eigen::Vector3d &p_measured,
                  const Eigen::Matrix3d &R_cov) {
    // Measurement model: y = p + noise
    // H matrix selects position from error state
    Eigen::Matrix<double, 3, DIM_ERR> H =
        Eigen::Matrix<double, 3, DIM_ERR>::Zero();
    H.block<3, 3>(0, IDX_P) = Eigen::Matrix3d::Identity();

    // Innovation
    Eigen::Vector3d residual = p_measured - x_.p;

    // Kalman Gain
    Eigen::Matrix<double, 3, 3> S = H * P_ * H.transpose() + R_cov;
    Eigen::Matrix<double, DIM_ERR, 3> K = P_ * H.transpose() * S.inverse();

    // Error State Update
    ErrorStateVec delta_x = K * residual;
    ErrorCovMat I_KH = ErrorCovMat::Identity() - K * H;
    P_ =
        I_KH * P_ * I_KH.transpose() + K * R_cov * K.transpose(); // Joseph form

    inject_error(delta_x);
  }

  // -- Correction Step: Magnetometer (Heading/Vector) --
  void update_mag(const Eigen::Vector3d &m_measured,
                  const Eigen::Matrix3d &R_cov) {
    // Predicted measurement in body frame: m_b = R(q)^T * m_inertial
    Eigen::Matrix3d R = x_.q.toRotationMatrix();
    Eigen::Vector3d m_predicted = R.transpose() * constants::mag_field_inertial;

    // Residual: z - h(x)
    Eigen::Vector3d residual = m_measured - m_predicted;

    // Jacobian H w.r.t angular error theta
    // m_b_new approx m_b_old + [m_b_old]x * delta_theta
    // So H = [m_b]x
    Eigen::Matrix<double, 3, DIM_ERR> H =
        Eigen::Matrix<double, 3, DIM_ERR>::Zero();
    H.block<3, 3>(0, IDX_TH) = skew(m_predicted);

    // Kalman Gain
    Eigen::Matrix<double, 3, 3> S = H * P_ * H.transpose() + R_cov;
    Eigen::Matrix<double, DIM_ERR, 3> K = P_ * H.transpose() * S.inverse();

    // Error State Update
    ErrorStateVec delta_x = K * residual;
    // Simple covariance update for demo (Joseph form preferred for stability)
    // P_ = (ErrorCovMat::Identity() - K * H) * P_;
    ErrorCovMat I_KH = ErrorCovMat::Identity() - K * H;
    P_ =
        I_KH * P_ * I_KH.transpose() + K * R_cov * K.transpose(); // Joseph form

    inject_error(delta_x);
  }

private:
  void inject_error(const ErrorStateVec &dx) {
    // 1. Position
    x_.p += dx.segment<3>(IDX_P);

    // 2. Velocity
    x_.v += dx.segment<3>(IDX_V);

    // 3. Attitude (Quaternion)
    // dq = [1, 0.5*dtheta] for small errors
    Eigen::Vector3d dtheta = dx.segment<3>(IDX_TH);
    Eigen::Quaterniond dq;
    dq.w() = 1.0;
    dq.vec() = 0.5 * dtheta;
    dq.normalize();
    x_.q = (x_.q * dq).normalized(); // Nominal * Error

    // 4. Biases
    x_.b_a += dx.segment<3>(IDX_BA);
    x_.b_g += dx.segment<3>(IDX_BG);

    // 5. Reset Error State (implied 0) and ESKF Covariance Reset
    // Technically P should be projected if error is large (G matrix),
    // but for small errors G ~ I.
    // We leave P as is.
  }

  static Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
  }
};

// -- Extensive Simulation --

struct SimulationData {
  double t;
  Eigen::Vector3d true_pos;
  Eigen::Vector3d true_vel;
  Eigen::Quaterniond true_quat;

  // Sensor readings (Noisy)
  Eigen::Vector3d imu_acc;
  Eigen::Vector3d imu_gyro;
  Eigen::Vector3d mag;
  Eigen::Vector3d gps_pos;

  // Availability flags
  bool has_gps;
  bool has_mag;
};

class Simulator {
  std::mt19937 gen_{0}; // std::random_device{}()};
  std::normal_distribution<double> dist_acc_{0.0, 0.2};  // m/s^2 noise
  std::normal_distribution<double> dist_gyr_{0.0, 0.01}; // rad/s noise
  std::normal_distribution<double> dist_gps_{0.0, 1.5};  // m noise
  std::normal_distribution<double> dist_mag_{0.0, 0.05}; // normalized noise

  // Biases (Constant for sim)
  Eigen::Vector3d true_ba_{0.1, -0.1, 0.05};
  Eigen::Vector3d true_bg_{0.01, 0.01, -0.01};

public:
  SimulationData step(double t, double dt) {
    SimulationData data;
    data.t = t;

    // Trajectory: Figure 8 in XY plane, sinusoidal Z
    double freq = 0.2;
    double A = 50.0; // Radius

    // Parametric Position
    data.true_pos = Eigen::Vector3d{A * std::sin(freq * t),
                                    A * std::sin(freq * t) *
                                        std::cos(freq * t), // Lemniscate-ish
                                    10.0 * std::sin(0.5 * t)};

    // Analytic Velocity (Derivative)
    // Simplified for demo: Finite difference for robust "truth" generation
    static Eigen::Vector3d last_pos = data.true_pos;
    if (t == 0)
      last_pos = data.true_pos;
    data.true_vel = (data.true_pos - last_pos) / dt;
    last_pos = data.true_pos;

    // Analytic Acceleration (Finite diff)
    static Eigen::Vector3d last_vel = data.true_vel;
    Eigen::Vector3d true_acc_inertial = (data.true_vel - last_vel) / dt;
    last_vel = data.true_vel;

    // Attitude: Point velocity vector forward + some roll
    if (data.true_vel.norm() > 0.1) {
      Eigen::Vector3d forward = data.true_vel.normalized();
      Eigen::Vector3d right =
          forward.cross(Eigen::Vector3d::UnitZ()).normalized();
      Eigen::Vector3d up = right.cross(forward);
      Eigen::Matrix3d R_true;
      R_true.col(0) = forward;
      R_true.col(1) = right;
      R_true.col(2) = up;
      data.true_quat = Eigen::Quaterniond(R_true);
    } else {
      data.true_quat = Eigen::Quaterniond::Identity();
    }

    // True Gyro (Finite diff of attitude not trivial, approx body rates)
    // For simulation, we just infer required omega to match R change
    // Or simpler: generate sensors from kinematics?
    // Let's use reverse: a_m = R^T(a_inertial - g)
    Eigen::Matrix3d R_t = data.true_quat.toRotationMatrix().transpose();
    Eigen::Vector3d a_body =
        R_t *
        (true_acc_inertial + constants::gravity_vector); // Proper acceleration

    // Add noise & bias to Accel
    data.imu_acc =
        a_body + true_ba_ +
        Eigen::Vector3d{dist_acc_(gen_), dist_acc_(gen_), dist_acc_(gen_)};

    // Gyro: simplified, just add noise to a synthetic omega
    // Calculating true omega from discrete quat steps
    // q_next = q * Exp(w * dt) -> Exp(w*dt) = q_inv * q_next
    static Eigen::Quaterniond last_q = data.true_quat;
    Eigen::Quaterniond dq = last_q.conjugate() * data.true_quat;
    Eigen::AngleAxisd aa(dq);
    Eigen::Vector3d true_omega = aa.axis() * aa.angle() / dt;
    last_q = data.true_quat;

    data.imu_gyro =
        true_omega + true_bg_ +
        Eigen::Vector3d{dist_gyr_(gen_), dist_gyr_(gen_), dist_gyr_(gen_)};

    // Magnetometer
    Eigen::Vector3d mag_body = R_t * constants::mag_field_inertial;
    data.mag = mag_body + Eigen::Vector3d{dist_mag_(gen_), dist_mag_(gen_),
                                          dist_mag_(gen_)};

    // GPS
    data.gps_pos =
        data.true_pos +
        Eigen::Vector3d{dist_gps_(gen_), dist_gps_(gen_), dist_gps_(gen_)};

    // Rates
    data.has_gps = (std::fmod(t, 1.0) < dt); // 1Hz
    data.has_mag = (std::fmod(t, 0.1) < dt); // 10Hz

    return data;
  }
};

// -- Main Execution --

int main() {
  std::cout
      << "Initializing Error-State Kalman Filter Simulation (C++23 Style)...\n";

  // 1. Setup Filter
  error_state_kalman_filter eskf;

  // Initialize with a slight error to prove convergence
  error_state_kalman_filter::nominal_state_t init_state;
  init_state.p = Eigen::Vector3d::Zero(); // Truth starts near zero
  init_state.v = Eigen::Vector3d::Zero();
  init_state.q = Eigen::Quaterniond::Identity();
  // Biases start at 0 (filter must estimate them)

  // High uncertainty initially
  error_state_kalman_filter::ErrorCovMat init_P =
      error_state_kalman_filter::ErrorCovMat::Identity();
  init_P.block<3, 3>(0, 0) *= 10.0; // Position uncertain
  init_P.block<3, 3>(9, 9) *= 0.1;  // Bias uncertain

  eskf.initialize(init_state, init_P);

  Simulator sim;
  double t = 0;
  double dt = 0.01; // 100Hz IMU
  double duration = 30.0;

  // Simulation Loop
  std::cout << std::format("{:<10} {:<25} {:<25} {:<15}\n", "Time(s)",
                           "Pos Error (m)", "Att Error (deg)", "Status");
  std::cout << std::string(80, '-') << "\n";

  while (t < duration) {
    auto data = sim.step(t, dt);

    // 1. Prediction (High rate IMU)
    eskf.predict(data.imu_acc, data.imu_gyro, dt);

    // 2. Corrections
    if (data.has_gps) {
      Eigen::Matrix3d R_gps =
          Eigen::Matrix3d::Identity() * 2.25; // 1.5m std dev squared
      eskf.update_gps(data.gps_pos, R_gps);
    }

    if (data.has_mag) {
      Eigen::Matrix3d R_mag = Eigen::Matrix3d::Identity() * 0.01;
      eskf.update_mag(data.mag, R_mag);
    }

    // 3. Analysis / Logging (1Hz)
    if (std::fmod(t, 1.0) < dt) {
      auto est = eskf.nominal_state();
      double pos_err = (est.p - data.true_pos).norm();

      // Quaternion angular distance
      Eigen::Quaterniond q_err_q = est.q.conjugate() * data.true_quat;
      double att_err_deg =
          2.0 * std::atan2(q_err_q.vec().norm(), std::abs(q_err_q.w())) *
          180.0 / std::numbers::pi;

      std::cout << std::format("{:<10.2f} {:<25.8f} {:<25.8f} ", t, pos_err,
                               att_err_deg);

      if (pos_err < 5.0 && att_err_deg < 5.0) {
        std::cout << "[CONVERGED]\n";
      } else {
        std::cout << "[ALIGNING]\n";
      }
    }

    t += dt;
  }

  // Final Report
  auto final_est = eskf.nominal_state();
  std::cout << "\nSimulation Complete.\n";
  std::cout << "Final Bias Est (Acc): " << final_est.b_a.transpose() << "\n";
  std::cout << "True Bias (Acc)     : 0.1 -0.1 0.05\n";

  return 0;
}
// NOLINTEND
