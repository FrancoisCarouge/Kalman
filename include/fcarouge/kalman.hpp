/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
https://github.com/FrancoisCarouge/Kalman

SPDX-License-Identifier: Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org> */

#ifndef FCAROUGE_KALMAN_HPP
#define FCAROUGE_KALMAN_HPP

//! @file
//! @brief The Kalman filter class and library top-level header.
//!
//! @details Provides the library public definitions of filters, algorithms,
//! utilities, and documentation. Only this header file is intended for
//! inclusion in third party software.

#include "kalman_forward.hpp"
#include "kalman_internal/factory.hpp"
#include "kalman_internal/format.hpp"
#include "kalman_internal/print.hpp"
#include "kalman_internal/utility.hpp"

namespace fcarouge {
//! @name Types
//! @{

//! @brief A generic Kalman filter.
//!
//! @details The Kalman filter is a Bayesian filter that uses multivariate
//! Gaussians, a recursive state estimator, a linear quadratic estimator (LQE),
//! and an Infinite Impulse Response (IIR) filter. It is a control theory tool
//! applicable to signal estimation, sensor fusion, or data assimilation
//! problems. The filter is applicable for unimodal and uncorrelated
//! uncertainties. The filter assumes white noise, propagation and measurement
//! functions are differentiable, and that the uncertainty stays centered on the
//! state estimate. The filter is the optimal linear filter under assumptions.
//! The filter updates estimates by multiplying Gaussians and predicts estimates
//! by adding Gaussians. Designing a filter is as much art as science. Design
//! the state $X$, $P$, the process $F$, $Q$, the measurement $Z$, $R$, the
//! measurement function $H$, and if the system has control inputs $U$, $G$.
//!
//! This library supports various simple and extended filters. The
//! implementation is independent from linear algebra backends. Arbitrary
//! parameters can be added to the prediction and update stages to participate
//! in gain-scheduling or linear parameter varying (LPV) systems. The default
//! filter type is a generalized, customizable, and extended filter. The default
//! type parameters implement a one-state, one-output, and double-precision
//! floating-point type filter. The default update equation uses the Joseph
//! form. Examples illustrate various usages and implementation tradeoffs. A
//! standard formatter specialization is included for representation of the
//! filter states. Filters with `state x output x input` dimensions as 1x1x1 and
//! 1x1x0 (no input) are supported through vanilla C++. Higher dimension filters
//! require a linear algebra backend. Customization points and type injections
//! allow for implementation tradeoffs.
//!
//! @tparam Filter Exposition only. The deduced internal filter template
//! parameter. Class template argument deduction (CTAD) figures out the filter
//! type based on the declared configuration. See deduction guide. The internal
//! implementation, filtering strategies, and presence of members vary based on
//! the constructed, configured, declared, or deduced filter.
//!
//! @todo Make this class usable in constant expressions.
//! @todo Is this filter restricted to Newton's equations of motion? That is
//! only a discretized continuous-time kinematic filter? How about non-Newtonian
//! systems?
//! @todo Symmetrization support might be superfluous. How to confirm it is safe
//! to remove? Optional?
//! @todo Would we want to support smoothers?
//! @todo Prepare support for larger dataset recording for graphing, metrics of
//! large test data to facilitate tuning.
//! @todo Support filter generator from equation? Third party integration?
//! @todo Compare performance of general filter with its equivalent generated?
//! @todo Support ranges operator filter?
//! @todo Support mux pipes https://github.com/joboccara/pipes operator filter?
//! @todo Reproduce Ardupilot's inertial navigation EKF and comparison
//! benchmarks in SITL (software in the loop simulation).
//! @todo Should we provide the operator[] for the vector characteristics
//! regardless of implementation? And for the matrix ones too? It could simplify
//! client code.
//! @todo Should we provide the operator[] for state directly on the filter? Is
//! the state X always what the user would want?
//! @todo Support, test complex number filters?
//! @todo Use automatic (Eigen::AutoDiffScalar?), symbolic, numerical solvers to
//! define the filter characteristics and simplify solving the dynamic system
//! for non-mathematicians.
//! @todo Should we add back the call operator? How to resolve the
//! update/predict ordering? And parameter ordering?
//! @todo Should we support the noise cross covariance `N = E[wvᵀ]` for
//! correlated noise sources, with default to null?
//! @todo Can we implement Temporal Parallelization of Bayesian Smoothers, Simo
//! Sarkka, Senior Member, IEEE, Angel F. Garc ıa-Fernandez,
//! https://arxiv.org/pdf/1905.13002.pdf ? GPU implementation? Parallel
//! implementation?
template <typename Filter>
class kalman : public kalman_internal::conditional_member_types<Filter> {
private:
  //! @name Private Member Variables
  //! @{

  //! @brief Encapsulates the implementation details of the filter.
  //!
  //! @details Optionally exposes a variety of members and methods according to
  //! the selected implementation.
  Filter filter;

  //! @}

public:
  //! @name Public Member Functions
  //! @{

  //! @brief Constructs a Kalman filter from its declared configuration.
  //!
  //! @see Deduction guide for details.
  //!
  //! @complexity Constant.
  template <typename... Arguments>
  inline constexpr kalman(Arguments... arguments);

  //! @brief Copy constructs a filter.
  //!
  //! @details Copy constructor. Constructs the filter with the contents of
  //! the `other` filter using copy semantics (i.e. the data in `other`
  //! filter is copied from the other into this filter).
  //!
  //! @param other Another filter to be used as source to initialize the
  //! elements of the filter with.
  //!
  //! @complexity Constant.
  inline constexpr kalman(const kalman &other) = default;

  //! @brief Move constructs a filter.
  //!
  //! @warning Some filter types have a known move memory safety defect.
  //!
  //! @details Move constructor. Constructs the filter with the contents of
  //! the `other` filter using move semantics (i.e. the data in `other`
  //! filter is moved from the other into this filter).
  //!
  //! @param other Another filter to be used as source to initialize the
  //! elements of the filter with.
  //!
  //! @complexity Constant.
  inline constexpr kalman(kalman &&other) noexcept = default;

  //! @brief Copy assignment operator.
  //!
  //! @details Replaces the contents of the filter with those of the `other`
  //! filter using copy semantics (i.e. the data in `other` filter is copied
  //! from the other into this filter).
  //!
  //! @param other Another filter to be used as source to initialize the
  //! elements of the filter with.
  //!
  //! @return The reference value of this implicit object filter parameter,
  //! i.e. `*this`.
  //!
  //! @complexity Constant.
  inline constexpr auto operator=(const kalman &other) -> kalman & = default;

  //! @brief Move assignment operator.
  //!
  //! @warning Some filter types have a known move memory safety defect.
  //!
  //! @details Replaces the contents of the filter with those of the `other`
  //! filter using move semantics (i.e. the data in `other` filter is moved from
  //! the other into this filter). The other filter is in a valid but
  //! unspecified state afterwards.
  //!
  //! @param other Another filter to be used as source to initialize the
  //! elements of the filter with.
  //!
  //! @return The reference value of this implicit object filter parameter,
  //! i.e. `*this`.
  //!
  //! @complexity Constant.
  inline constexpr auto
  operator=(kalman &&other) noexcept -> kalman & = default;

  //! @brief Destructs the Kalman filter.
  //!
  //! @complexity Constant.
  inline constexpr ~kalman() = default;
  //! @}

  //! @name Public Characteristics Member Functions
  //! @{

  //! @brief Read, write the estimated state column vector X.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the estimated state
  //! column vector X characteristic.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) x(this auto &&self, const auto &...values)
    requires(kalman_internal::has_state<Filter>);

  //! @brief Read, write the observation column vector Z.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the observation
  //! column vector Z characteristic.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) z(this auto &&self, const auto &...values)
    requires(kalman_internal::has_output<Filter>);

  //! @brief Read, write the control column vector U.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the control column
  //! vector U characteristic.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) u(this auto &&self, const auto &...values)
    requires(kalman_internal::has_input<Filter>);

  //! @brief Read, write the estimated covariance matrix P.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the estimated
  //! covariance matrix P characteristic.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) p(this auto &&self, const auto &...values)
    requires(kalman_internal::has_estimate_uncertainty<Filter>);

  //! @brief Read, write the process noise covariance matrix or function Q.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the process noise
  //! covariance matrix Q characteristic. The characteristic may also be a
  //! callable of the form `process_uncertainty(const state &, const
  //! PredictionTypes &...)`.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) q(this auto &&self, const auto &...values)
    requires(kalman_internal::has_process_uncertainty<Filter>);

  //! @brief Read, write the observation noise covariance matrix R.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the observation
  //! noise covariance matrix R characteristic. The characteristic may also be a
  //! callable of the form `output_uncertainty(const state &, const output &,
  //! const UpdateTypes &...)`.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) r(this auto &&self, const auto &...values)
    requires(kalman_internal::has_output_uncertainty<Filter>);

  //! @brief Read, write the state transition matrix F.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the state transition
  //! matrix F characteristic. The characteristic may also be a callable of the
  //! form `state_transition(const state &, const input
  //! &, const PredictionTypes &...)`. For non-linear system, or extended
  //! filter, F is the Jacobian of the state transition function: `F = ∂f/∂X =
  //! ∂fj/∂xi` that is each row i contains the derivatives of the state
  //! transition function for every element j in the state column vector X.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) f(this auto &&self, const auto &...values)
    requires(kalman_internal::has_state_transition<Filter>);

  //! @brief Read, write the observation transition matrix H.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the observation
  //! transition matrix H characteristic. The characteristic may also be a
  //! callable of the form `output_model(const state &, const UpdateTypes
  //! &...)`. For non-linear system, or extended filter, H is the Jacobian of
  //! the state observation function: `H = ∂h/∂X = ∂hj/∂xi` that is each row i
  //! contains the derivatives of the state observation function for every
  //! element j in the state column vector X. This member function is not
  //! present when the filter has no output model.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) h(this auto &&self, const auto &...values)
    requires(kalman_internal::has_output_model<Filter>);

  //! @brief Read, write the control transition matrix G.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the control
  //! transition matrix G. characteristic. The characteristic may also be a
  //! callable of the form `input_control(const PredictionTypes &...)`.
  //! This member function is not present when the filter has no input control.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) g(this auto &&self, const auto &...values)
    requires(kalman_internal::has_input_control<Filter>);

  //! @brief Read, write the gain matrix K.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the gain matrix K
  //! characteristic.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) k(this auto &&self, const auto &...values)
    requires(kalman_internal::has_gain<Filter>);

  //! @brief Read, write the innovation column vector Y.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the innovation
  //! column vector Y characteristic.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) y(this auto &&self, const auto &...values)
    requires(kalman_internal::has_innovation<Filter>);

  //! @brief Read, write the innovation uncertainty matrix S.
  //!
  //! @param self Explicit object parameter. Internal implementation detail.
  //! @param values The optional copied initializers to set the innovation
  //! uncertainty matrix S characteristic.
  //!
  //! @complexity Constant.
  inline constexpr decltype(auto) s(this auto &&self, const auto &...values)
    requires(kalman_internal::has_innovation_uncertainty<Filter>);

  //! @}

  //! @name Public Filtering Member Functions
  //! @{

  //! @brief Produces estimates of the state variables and uncertainties.
  //!
  //! @details Also known as the propagation step. Implements the total
  //! probability theorem. Estimate the next state by suming the known
  //! probabilities.
  //!
  //! @param arguments The prediction and input parameters of
  //! the filter, in that order. The arguments need to be compatible with the
  //! filter types. The prediction parameters convertible to the
  //! `PredictionTypes` template pack types are passed through for computations
  //! of prediction matrices. The control parameter pack types convertible to
  //! the `Input` template type. The prediction types are explicitly defined
  //! with the class definition.
  //!
  //! @todo Consider if returning the state column vector X would be preferable?
  //! Or fluent interface? Would be compatible with an ES-EKF implementation?
  //! @todo Can the parameter pack of `PredictionTypes` be explicit in the
  //! method declaration for user clarity?
  inline constexpr void predict(const auto &...arguments);

  //! @brief Returns the Nth prediction argument.
  //!
  //! @details Convenience access to the last used prediction arguments.
  //!
  //! @tparam The non-type template parameter index position of the prediction
  //! argument types.
  //!
  //! @return The prediction argument corresponding to the Nth position of the
  //! parameter pack of the tuple `PredictionTypes` class template type.
  //!
  //! @complexity Constant.
  template <auto Position> inline constexpr auto predict() const;

  //! @brief Updates the estimates with the outcome of a measurement.
  //!
  //! @details Also known as the observation or correction step. Implements the
  //! Bayes' theorem. Combine one measurement and the prior estimate by applying
  //! the multiplicative law.
  //!
  //! @param arguments The update and output parameters of
  //! the filter, in that order. The arguments need to be compatible with the
  //! filter types. The update parameters convertible to the
  //! `UpdateTypes` template pack types are passed through for computations of
  //! update matrices. The observation parameter pack types convertible to
  //! the `Output` template type. The update types are explicitly
  //! defined with the class definition.
  //!
  //! @todo Consider if returning the state column vector X would be preferable?
  //! Or fluent interface? Would be compatible with an ES-EKF implementation?
  //! @todo Can the parameter pack of `UpdateTypes` be explicit in the method
  //! declaration for user clarity?
  inline constexpr void update(const auto &...arguments);

  //! @brief Returns the Nth update argument.
  //!
  //! @details Convenience access to the last used update arguments.
  //!
  //! @tparam The non-type template parameter index position of the update
  //! argument types.
  //!
  //! @return The update argument corresponding to the Nth position of the
  //! parameter pack of the tuple `UpdateTypes` class template type.
  //!
  //! @complexity Constant.
  template <auto Position> inline constexpr auto update() const;
  //! @}
};

//! @brief State type wrapper for filter declaration support.
//!
//! @todo Use alias from internal when Clang supports CTAD for alias?
using kalman_internal::state;

//! @brief Estimate uncertainty type wrapper for filter declaration support.
using kalman_internal::estimate_uncertainty;

//! @brief Output uncertainty type wrapper for filter declaration support.
using kalman_internal::output_uncertainty;

//! @brief Process uncertainty type wrapper for filter declaration support.
using kalman_internal::process_uncertainty;

//! @brief Input value wrapper for filter declaration support.
using kalman_internal::input;

//! @brief Input type wrapper for filter declaration support.
using kalman_internal::input_t;

//! @brief Output value wrapper for filter declaration support.
using kalman_internal::output;

//! @brief Output type wrapper for filter declaration support.
using kalman_internal::output_t;

//! @brief Output model type wrapper for filter declaration support.
using kalman_internal::output_model;

//! @brief Transition function type wrapper for filter declaration support.
using kalman_internal::transition;

//! @brief Observation function type wrapper for filter declaration support.
using kalman_internal::observation;

//! @brief Update types wrapper for filter declaration support.
using kalman_internal::update_types;

//! @brief Prediction types wrapper for filter declaration support.
using kalman_internal::prediction_types;

//! @brief State transition types wrapper for filter declaration support.
using kalman_internal::state_transition;

//! @brief Input control types wrapper for filter declaration support.
using kalman_internal::input_control;

//! @}

//! @name Deduction Guides
//! @{

//! @brief Deduces the filter type from its declared configuration.
//!
//! @details The configuration arguments passed are used to determine at compile
//! time the type of fiter to use. The objecive is to select the most performant
//! filter within the defined configuraton parameters.
//!
//! @tparam Arguments The declarations of the filter configuration.
//!
//! @todo Should the parameter be named configurations?
//! @todo Should the configuration examples, supports be documented here?
template <typename... Arguments>
kalman(Arguments... arguments)
    -> kalman<kalman_internal::deduce_filter<Arguments...>>;

//! @}

//! @name Decorators
//! @{

//! @brief Filter decorator to print activities to the standard output.
//!
//! @details Pipe decorator to filter declaration to print out its activities:
//! construction, destruction, updates, predictions, and characteristic changes.
//! Prints with default formatting. Takes no parameters.
inline constexpr printer print;

//! @}

} // namespace fcarouge

#include "kalman_internal/kalman.tpp"

#endif // FCAROUGE_KALMAN_HPP
