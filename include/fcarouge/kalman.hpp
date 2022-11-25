/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.2.0
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
//! @brief The main Kalman filter class.

#include "internal/format.hpp"
#include "internal/kalman.hpp"
#include "internal/utility.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace fcarouge {

//! @brief Convenience tuple-like empty pack type.
using empty_pack = internal::empty_pack;

//! @brief Convenience tuple-like pack type.
template <typename... Types> using pack = internal::pack<Types...>;

//! @brief A generic Kalman filter for C++23.
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
//! @tparam State The type template parameter of the state column vector x.
//! State variables can be observed (measured), or hidden variables (inferred).
//! This is the the mean of the multivariate Gaussian. Defaults to `double`.
//! @tparam Output The type template parameter of the measurement column vector
//! z. Defaults to `double`.
//! @tparam Input The type template parameter of the control u. A `void` input
//! type can be used for systems with no input control to disable all of the
//! input control features, the control transition matrix G support, and the
//! other related computations from the filter. Defaults to `void`.
//! @tparam Divide The customization point object template parameter of the
//! matrix division functor. Default to the standard division
//! `std::divides<void>` function object.
//! @tparam UpdateTypes The additional update function parameter types passed in
//! through a tuple-like parameter type, composing zero or more types.
//! Parameters such as delta times, variances, or linearized values. The
//! parameters are propagated to the function objects used to compute the state
//! observation H and the observation noise R matrices. The parameters are also
//! propagated to the state observation function object h. Defaults to no
//! parameter types, the empty pack.
//! @tparam PredictionTypes The additional prediction function parameter types
//! passed in through a tuple-like parameter type, composing zero or more types.
//! Parameters such as delta times, variances, or linearized values. The
//! parameters are propagated to the function objects used to compute the
//! process noise Q, the state transition F, and the control transition G
//! matrices. The parameters are also propagated to the state transition
//! function object f. Defaults to no parameter types, the empty pack.
//!
//! @note This class could be usable in constant expressions if `std::function`
//! could too. The polymorphic function wrapper was used in place of function
//! pointers to enable default initialization from this class, captured member
//! variables.
//!
//! @todo Is this filter restricted to Newton's equations of motion? That is
//! only a discretized continuous-time kinematic filter? How about non-Newtonian
//! systems?
//! @todo Would it be beneficial to support initialization list for
//! characteristics?
//! @todo Symmetrization support might be superfluous. How to confirm it is safe
//! to remove?
//! @todo Would we want to support smoothers?
//! @todo How to add or associate constraints on the types and operation to
//! support compilation and semantics?
//! @todo Which constructors to support? Consider constructors? CTAD? Guides?
//! @todo Prepare support for larger dataset recording for graphing, metrics of
//! large test data to facilitate tuning.
//! @todo Support filter generator? Integration? Reflection in C++...
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
//! @todo Consider if a fluent interface would be preferable for
//! characteristics?
//! @todo Consider additional characteristics method overloads?
//! @todo A clear or reset member equivalent may be useful for real-time
//! re-initializations but to what default?
//! @todo Expand std::format support with standard arguments and Eigen3 types.
//! @todo Support, test complex number filters?
//! @todo Use automatic (Eigen::AutoDiffScalar?), symbolic, numerical solvers to
//! define the filter characteristics and simplify solving the dynamic system
//! for non-mathematicians.
//! @todo Support, use "Taking Static Type-Safety to the Next Level - Physical
//! Units for Matrices" by Daniel Withopf and record the lesson learned: both
//! usage and development is harder without compile time units verification.
//! @todo Should we add back the call operator? How to resolve the
//! update/predict ordering? And parameter ordering?
template <typename State = double, typename Output = double,
          typename Input = void, typename Divide = std::divides<void>,
          typename UpdateTypes = empty_pack,
          typename PredictionTypes = empty_pack>
class kalman final {
private:
  //! @name Private Member Types
  //! @{

  //! @brief Implementation details of the filter.
  //!
  //! @brief The internal implementation unpacks the parameter packs from
  //! tuple-like types which allows for multiple parameter pack deductions.
  using implementation = internal::kalman<State, Output, Input, Divide,
                                          internal::repack_t<UpdateTypes>,
                                          internal::repack_t<PredictionTypes>>;

  //! @}

  //! @name Private Member Variables
  //! @{

  //! @brief Encapsulates the implementation details of the filter.
  implementation filter;

  //! @}

public:
  //! @name Public Member Types
  //! @{

  //! @brief Type of the state estimate column vector X.
  using state = typename implementation::state;

  //! @brief Type of the observation column vector Z.
  //!
  //! @details Also known as Y or O.
  using output = typename implementation::output;

  //! @brief Type of the control column vector U.
  //!
  //! @todo Conditionally remove this member type when no input is present.
  using input = typename implementation::input;

  //! @brief Type of the estimated correlated variance matrix P.
  //!
  //! @details Also known as Σ.
  using estimate_uncertainty = typename implementation::estimate_uncertainty;

  //! @brief Type of the process noise correlated variance matrix Q.
  using process_uncertainty = typename implementation::process_uncertainty;

  //! @brief Type of the observation noise correlated variance matrix R.
  using output_uncertainty = typename implementation::output_uncertainty;

  //! @brief Type of the state transition matrix F.
  //!
  //! @details Also known as the fundamental matrix, propagation, Φ, or A.
  using state_transition = typename implementation::state_transition;

  //! @brief Type of the observation transition matrix H.
  //!
  //! @details Also known as the measurement transition matrix or C.
  using output_model = typename implementation::output_model;

  //! @brief Type of the control transition matrix G.
  //!
  //! @details Also known as B.
  //!
  //! @todo Conditionally remove this member type when no input is present.
  using input_control = typename implementation::input_control;

  //! @brief Type of the gain matrix K.
  using gain = typename implementation::gain;

  //! @brief Type of the innovation column vector Y.
  using innovation = typename implementation::innovation;

  //! @brief Type of the innovation uncertainty matrix S.
  using innovation_uncertainty =
      typename implementation::innovation_uncertainty;

  //! @}

  //! @name Public Member Functions
  //! @{

  //! @brief Constructs a Kalman filter without configuration.
  //!
  //! @complexity Constant.
  inline constexpr kalman() = default;

  //! @brief Copy constructs a filter.
  //!
  //! @details Constructs the filter with the copy of the contents of the
  //! `other` filter.
  //!
  //! @param other Another filter to be used as source to initialize the
  //! elements of the filter with.
  //!
  //! @complexity Constant.
  inline constexpr kalman(const kalman &other) = default;

  //! @brief Move constructs a filter.
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
  //! @details Destroys or copy-assigns the contents with a copy of the contents
  //! of the other filter.
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
  inline constexpr auto operator=(kalman &&other) noexcept
      -> kalman & = default;

  //! @brief Destructs the kalman filter.
  //!
  //! @complexity Constant.
  inline constexpr ~kalman() = default;

  //! @}

  //! @name Public Characteristics Member Functions
  //! @{

  //! @brief Returns the state estimate column vector X.
  //!
  //! @return The state estimate column vector X.
  //!
  //! @complexity Constant.
  //!
  //! @note Overloading the operator dot would have been nice had it existed.
  //!
  //! @todo Collapse cv-ref qualifier-aware member functions per C++23 P0847 to
  //! avoid duplication: `inline constexpr auto & x(this auto&& self)`.
  inline constexpr auto x() const -> const state &;
  inline constexpr auto x() -> state &;

  //! @brief Sets the state estimate column vector X.
  //!
  //! @param value The first copied initializer used to set the state estimate
  //! column vector X.
  //! @param values The optional second and other copied initializers to set the
  //! state estimate column vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(const auto &value, const auto &...values);

  //! @brief Sets the state estimate column vector X.
  //!
  //! @param value The first moved initializer used to set the state estimate
  //! column vector X.
  //! @param values The optional second and other moved initializers to set the
  //! state estimate column vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(auto &&value, auto &&...values);

  //! @brief Returns the last observation column vector Z.
  //!
  //! @return The last observation column vector Z.
  //!
  //! @complexity Constant.
  inline constexpr auto z() const -> const output &;

  //! @brief Returns the last control column vector U.
  //!
  //! @details This member function is not present when the filter has no input.
  //!
  //! @return The last control column vector U.
  //!
  //! @complexity Constant.
  inline constexpr auto u() const
      -> const input &requires(not std::is_same_v<Input, void>);

  //! @brief Returns the estimated covariance matrix P.
  //!
  //! @return The estimated correlated variance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr auto p() const -> const estimate_uncertainty &;
  inline constexpr auto p() -> estimate_uncertainty &;

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @param value The first copied initializer used to set the estimated
  //! covariance matrix P.
  //! @param values The optional second and other copied initializers to set the
  //! estimated covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(const auto &value, const auto &...values);

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @param value The first moved initializer used to set the estimated
  //! covariance matrix P.
  //! @param values The optional second and other moved initializers to set the
  //! estimated covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(auto &&value, auto &&...values);

  //! @brief Returns the process noise covariance matrix Q.
  //!
  //! @return The process noise correlated variance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr auto q() const -> const process_uncertainty &;
  inline constexpr auto q() -> process_uncertainty &;

  //! @brief Sets the process noise covariance matrix Q.
  //!
  //! @param value The first copied initializer used to set the process noise
  //! covariance matrix Q is of type `process_uncertainty` and the function is
  //! of the form `process_uncertainty(const state &, const PredictionTypes
  //! &...)`. The copied process noise covariance matrix Q or the copied target
  //! Callable object (function object, pointer to function, reference to
  //! function, pointer to member function, or pointer to data member) that will
  //! be bound to the prediction arguments and called by the filter to compute
  //! the process noise covariance matrix Q on prediction steps.
  //! @param values The optional second and other copied initializers to set the
  //! process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(const auto &value, const auto &...values);

  //! @brief Sets the process noise covariance matrix Q.
  //!
  //! @param value The first moved initializer used to set the process noise
  //! covariance matrix Q is of type `process_uncertainty` and the function is
  //! of the form `process_uncertainty(const state &, const PredictionTypes
  //! &...)`. The moved process noise covariance matrix Q or the moved target
  //! Callable object (function object, pointer to function, reference to
  //! function, pointer to member function, or pointer to data member) that will
  //! be bound to the prediction arguments and called by the filter to compute
  //! the process noise covariance matrix Q on prediction steps.
  //! @param values The optional second and other moved initializers to set the
  //! process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(auto &&value, auto &&...values);

  //! @brief Returns the observation noise covariance
  //! matrix R.
  //!
  //! @details The variance there is in each measurement.
  //!
  //! @return The observation noise correlated variance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr auto r() const -> const output_uncertainty &;
  inline constexpr auto r() -> output_uncertainty &;

  //! @brief Sets the observation noise covariance matrix R.
  //!
  //! @param value The first copied initializer used to set the observation
  //! noise covariance matrix R is of type `output_uncertainty` and the function
  //! is of the form `output_uncertainty(const state &, const output &, const
  //! UpdateTypes &...)`. The copied observation noise covariance matrix R or
  //! the copied target Callable object (function object, pointer to function,
  //! reference to function, pointer to member function, or pointer to data
  //! member) that will be called by the filter to compute the observation noise
  //! covariance matrix R on prediction steps.
  //! @param values The optional second and other copied initializers to set the
  //! observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(const auto &value, const auto &...values);

  //! @brief Sets the observation noise covariance matrix R.
  //!
  //! @param value The first moved initializer used to set the observation noise
  //! covariance matrix R is of type `output_uncertainty` and the function is of
  //! the form `output_uncertainty(const state &, const output &, const
  //! UpdateTypes &...)`. The moved observation noise covariance matrix R or the
  //! moved target Callable object (function object, pointer to function,
  //! reference to function, pointer to member function, or pointer to data
  //! member) that will be called by the filter to compute the observation noise
  //! covariance matrix R on prediction steps.
  //! @param values The optional second and other moved initializers to set the
  //! observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(auto &&value, auto &&...values);

  //! @brief Returns the state transition matrix F.
  //!
  //! @return The state transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr auto f() const -> const state_transition &;
  inline constexpr auto f() -> state_transition &;

  //! @brief Sets the state transition matrix F.
  //!
  //! @details The state transition matrix F is of type `state_transition` and
  //! the function is of the form `state_transition(const state &, const input
  //! &, const PredictionTypes &...)`. For non-linear system, or extended
  //! filter, F is the Jacobian of the state transition function: `F = ∂f/∂X =
  //! ∂fj/∂xi` that is each row i contains the derivatives of the state
  //! transition function for every element j in the state column vector X.
  //!
  //! @param value The first copied initializer used to set the copied state
  //! transition matrix F or the copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the state transition matrix F function on
  //! prediction steps.
  //! @param values The optional second and other copied initializers to set the
  //! state transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(const auto &value, const auto &...values);

  //! @brief Sets the state transition matrix F.
  //!
  //! @details The first copied initializer used to set the state transition
  //! matrix F is of type `state_transition` and the function is of the form
  //! `state_transition(const state &, const input &, const PredictionTypes
  //! &...)`. For non-linear system, or extended filter, F is the Jacobian of
  //! the state transition function: `F = ∂f/∂X = ∂fj/∂xi` that is each row i
  //! contains the derivatives of the state transition function for every
  //! element j in the state column vector X.
  //!
  //! @param value The first moved initializer used to set the moved state
  //! transition matrix F or the moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the state transition matrix F function on
  //! prediction steps.
  //! @param values The optional second and other moved initializers to set the
  //! state transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(auto &&value, auto &&...values);

  //! @brief Returns the observation transition matrix H.
  //!
  //! @return The observation, measurement transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr auto h() const -> const output_model &;
  inline constexpr auto h() -> output_model &;

  //! @brief Sets the observation transition matrix H.
  //!
  //! @details The observation transition matrix H is of type `output_model` and
  //! the function is of the form `output_model(const state &, const UpdateTypes
  //! &...)`. For non-linear system, or extended filter, H is the Jacobian of
  //! the state observation function: `H = ∂h/∂X = ∂hj/∂xi` that is each row i
  //! contains the derivatives of the state observation function for every
  //! element j in the state column vector X.
  //!
  //! @param value The first copied initializer used to set the copied
  //! observation transition matrix H or the copied target Callable object
  //! (function object, pointer to function, reference to function, pointer to
  //! member function, or pointer to data member) that will be bound to the
  //! prediction arguments and called by the filter to compute the observation,
  //! measurement transition matrix H on update steps.
  //! @param values The optional second and other copied initializers to set the
  //! observation transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(const auto &value, const auto &...values);

  //! @brief Sets the observation transition matrix H.
  //!
  //! @details The observation transition matrix H is of type `output_model` and
  //! the function is of the form `output_model(const state &, const UpdateTypes
  //! &...)`. For non-linear system, or extended filter, H is the Jacobian of
  //! the state observation function: `H = ∂h/∂X = ∂hj/∂xi` that is each row i
  //! contains the derivatives of the state observation function for every
  //! element j in the state column vector X.
  //!
  //! @param value The first moved initializer used to set the moved observation
  //! transition matrix H or the moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the observation, measurement transition
  //! matrix H on update steps.
  //! @param values The optional second and other moved initializers to set the
  //! observation transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(auto &&value, auto &&...values);

  //! @brief Returns the control transition matrix G.
  //!
  //! @details This member function is not present when the filter has no input.
  //!
  //! @return The control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr auto g() const
      -> const input_control &requires(not std::is_same_v<Input, void>);
  inline constexpr auto g()
      -> input_control &requires(not std::is_same_v<Input, void>);

  //! @brief Sets the control transition matrix G.
  //!
  //! @details The control transition matrix G is of type `input_control` and
  //! the function is of the form `input_control(const PredictionTypes &...)`.
  //! This member function is not present when the filter has no input.
  //!
  //! @param value The first copied initializer used to set the copied control
  //! transition matrix G or the copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the control transition matrix G on
  //! prediction steps.
  //! @param values The optional second and other copied initializers to set the
  //! control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(const auto &value, const auto &...values)
    requires(not std::is_same_v<Input, void>);

  //! @brief Sets the control transition matrix G.
  //!
  //! @details The control transition matrix G is of type `input_control` and
  //! the function is of the form `input_control(const PredictionTypes &...)`.
  //! This member function is not present when the filter has no input.
  //!
  //! @param value The first moved initializer used to set the moved control
  //! transition matrix G or the moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the control transition matrix G on
  //! prediction steps.
  //! @param values The optional second and other moved initializers to set the
  //! control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(auto &&value, auto &&...values)
    requires(not std::is_same_v<Input, void>);

  //! @brief Returns the gain matrix K.
  //!
  //! @return The gain matrix K.
  //!
  //! @complexity Constant.
  inline constexpr auto k() const -> const gain &;

  //! @brief Returns the innovation column vector Y.
  //!
  //! @return The innovation column vector Y.
  //!
  //! @complexity Constant.
  inline constexpr auto y() const -> const innovation &;

  //! @brief Returns the innovation uncertainty matrix S.
  //!
  //! @return The innovation uncertainty matrix S.
  //!
  //! @complexity Constant.
  inline constexpr auto s() const -> const innovation_uncertainty &;

  //! @brief Sets the extended state transition function f(x).
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the next state X on
  //! prediction steps of expression `state(const state &, const input &, const
  //! PredictionTypes &...)`. The default function `f(x) = F * X` is suitable
  //! for linear systems. For non-linear system, or extended filter, implement a
  //! linearization of the transition function f and the state transition F
  //! matrix is the Jacobian of the state transition function.
  //!
  //! @complexity Constant.
  inline constexpr void transition(const auto &callable);

  //! @brief Sets the extended state transition function f(x).
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the next state X on
  //! prediction steps of expression `state(const state &, const input &, const
  //! PredictionTypes &...)`. The default function `f(x) = F * X` is suitable
  //! for linear systems. For non-linear system, or extended filter, implement a
  //! linearization of the transition function f and the state transition F
  //! matrix is the Jacobian of the state transition function.
  //!
  //! @complexity Constant.
  inline constexpr void transition(auto &&callable);

  //! @brief Sets the extended state observation function h(x).
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the observation Z
  //! on update steps of expression `output(const state &, const UpdateTypes
  //! &...arguments)`. The default function `h(x) = H * X` is suitable for
  //! linear systems. For non-linear system, or extended filter, the client
  //! implements a linearization of the observation function hand the state
  //! observation H matrix is the Jacobian of the state observation function.
  //!
  //! @complexity Constant.
  inline constexpr void observation(const auto &callable);

  //! @brief Sets the extended state observation function h(x).
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the observation Z
  //! on update steps of expression `output(const state &, const UpdateTypes
  //! &...arguments)`. The default function `h(x) = H * X` is suitable for
  //! linear systems. For non-linear system, or extended filter, the client
  //! implements a linearization of the observation function hand the state
  //! observation H matrix is the Jacobian of the state observation function.
  //!
  //! @complexity Constant.
  inline constexpr void observation(auto &&callable);

  //! @}

  //! @name Public Filtering Member Functions
  //! @{

  //! @brief Produces estimates of the state variables and uncertainties.
  //!
  //! @details Implements the total probability theorem.
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
  //! parameter pack of the tuple-like `PredictionTypes` class template type.
  //!
  //! @complexity Constant.
  template <std::size_t Position> inline constexpr auto predict() const;

  //! @brief Updates the estimates with the outcome of a measurement.
  //!
  //! @details Also known as the observation or correction step. Implements the
  //! Bayes' theorem. Combine one measurement and the prior estimate.
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
  //! parameter pack of the tuple-like `UpdateTypes` class template type.
  //!
  //! @complexity Constant.
  template <std::size_t Position> inline constexpr auto update() const;

  //! @}
};

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::x() const
    -> const state & {
  return filter.x;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::x()
    -> state & {
  return filter.x;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::x(
    const auto &value, const auto &...values) {
  filter.x = std::move(state{value, values...});
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::x(
    auto &&value, auto &&...values) {
  filter.x = std::move(state{std::forward<decltype(value)>(value),
                             std::forward<decltype(values)>(values)...});
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation column vector Z is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::z() const
    -> const output & {
  return filter.z;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned control column vector U is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::u() const
    -> const input &requires(not std::is_same_v<Input, void>) {
                      return filter.u;
                    }

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto kalman<State, Output, Input,
                                                         Divide, UpdateTypes,
                                                         PredictionTypes>::p()
    const -> const estimate_uncertainty & {
  return filter.p;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::p()
    -> estimate_uncertainty & {
  return filter.p;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::p(
    const auto &value, const auto &...values) {
  filter.p = std::move(estimate_uncertainty{value, values...});
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::p(
    auto &&value, auto &&...values) {
  filter.p = std::move(
      estimate_uncertainty{std::forward<decltype(value)>(value),
                           std::forward<decltype(values)>(values)...});
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::q() const
    -> const process_uncertainty & {
  return filter.q;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::q()
    -> process_uncertainty & {
  return filter.q;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::q(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), process_uncertainty>) {
    filter.q = std::move(process_uncertainty{value, values...});
  } else {
    using noise_process_function = decltype(filter.noise_process_q);
    filter.noise_process_q =
        std::move(noise_process_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::q(
    auto &&value, auto &&...values) {
  if constexpr (std::is_convertible_v<decltype(value), process_uncertainty>) {
    filter.q = std::move(
        process_uncertainty{std::forward<decltype(value)>(value),
                            std::forward<decltype(values)>(values)...});
  } else {
    using noise_process_function = decltype(filter.noise_process_q);
    filter.noise_process_q = std::move(
        noise_process_function{std::forward<decltype(value)>(value),
                               std::forward<decltype(values)>(values)...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::r() const
    -> const output_uncertainty & {
  return filter.r;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::r()
    -> output_uncertainty & {
  return filter.r;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::r(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), output_uncertainty>) {
    filter.r = std::move(output_uncertainty{value, values...});
  } else {
    using noise_observation_function = decltype(filter.noise_observation_r);
    filter.noise_observation_r =
        std::move(noise_observation_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::r(
    auto &&value, auto &&...values) {
  if constexpr (std::is_convertible_v<decltype(value), output_uncertainty>) {
    filter.r = std::move(
        output_uncertainty{std::forward<decltype(value)>(value),
                           std::forward<decltype(values)>(values)...});
  } else {
    using noise_observation_function = decltype(filter.noise_observation_r);
    filter.noise_observation_r = std::move(
        noise_observation_function{std::forward<decltype(value)>(value),
                                   std::forward<decltype(values)>(values)...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::f() const
    -> const state_transition & {
  return filter.f;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::f()
    -> state_transition & {
  return filter.f;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::f(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), state_transition>) {
    filter.f = std::move(state_transition{value, values...});
  } else {
    using transition_state_function = decltype(filter.transition_state_f);
    filter.transition_state_f =
        std::move(transition_state_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::f(
    auto &&value, auto &&...values) {
  if constexpr (std::is_convertible_v<decltype(value), state_transition>) {
    filter.f =
        std::move(state_transition{std::forward<decltype(value)>(value),
                                   std::forward<decltype(values)>(values)...});
  } else {
    using transition_state_function = decltype(filter.transition_state_f);
    filter.transition_state_f = std::move(
        transition_state_function{std::forward<decltype(value)>(value),
                                  std::forward<decltype(values)>(values)...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::h() const
    -> const output_model & {
  return filter.h;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::h()
    -> output_model & {
  return filter.h;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::h(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), output_model>) {
    filter.h = std::move(output_model{value, values...});
  } else {
    using observation_state_function = decltype(filter.observation_state_h);
    filter.observation_state_h =
        std::move(observation_state_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::h(
    auto &&value, auto &&...values) {
  if constexpr (std::is_convertible_v<decltype(value), output_model>) {
    filter.h =
        std::move(output_model{std::forward<decltype(value)>(value),
                               std::forward<decltype(values)>(values)...});
  } else {
    using observation_state_function = decltype(filter.observation_state_h);
    filter.observation_state_h = std::move(
        observation_state_function{std::forward<decltype(value)>(value),
                                   std::forward<decltype(values)>(values)...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::g() const
    -> const input_control &requires(not std::is_same_v<Input, void>) {
                              return filter.g;
                            }

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto kalman<State, Output, Input,
                                                         Divide, UpdateTypes,
                                                         PredictionTypes>::g()
    -> input_control &requires(not std::is_same_v<Input, void>) {
                        return filter.g;
                      }

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void kalman<State, Output, Input, Divide, UpdateTypes,
                             PredictionTypes>::g(const auto &value,
                                                 const auto &...values)
  requires(not std::is_same_v<Input, void>)
{
  if constexpr (std::is_convertible_v<decltype(value), input_control>) {
    filter.g = std::move(input_control{value, values...});
  } else {
    using transition_control_function = decltype(filter.transition_control_g);
    filter.transition_control_g =
        std::move(transition_control_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::g(
    auto &&value, auto &&...values)
  requires(not std::is_same_v<Input, void>)
{
  if constexpr (std::is_convertible_v<decltype(value), input_control>) {
    filter.g =
        std::move(input_control{std::forward<decltype(value)>(value),
                                std::forward<decltype(values)>(values)...});
  } else {
    using transition_control_function = decltype(filter.transition_control_g);
    filter.transition_control_g = std::move(
        transition_control_function{std::forward<decltype(value)>(value),
                                    std::forward<decltype(values)>(values)...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned gain matrix K is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::k() const
    -> const gain & {
  return filter.k;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned innovation column vector Y is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::y() const
    -> const innovation & {
  return filter.y;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned innovation uncertainty matrix S is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::s() const
    -> const innovation_uncertainty & {
  return filter.s;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::transition(
    const auto &callable) {
  filter.transition = callable;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::transition(
    auto &&callable) {
  filter.transition = std::forward<decltype(callable)>(callable);
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::observation(
    const auto &callable) {
  filter.observation = callable;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::observation(
    auto &&callable) {
  filter.observation = std::forward<decltype(callable)>(callable);
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::update(
    const auto &...arguments) {
  filter.update(arguments...);
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
template <std::size_t Position>
[[nodiscard("The returned update argument is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::update()
    const {
  return std::get<Position>(filter.update_arguments);
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::predict(
    const auto &...arguments) {
  filter.predict(arguments...);
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
template <std::size_t Position>
[[nodiscard("The returned prediction argument is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::predict()
    const {
  return std::get<Position>(filter.prediction_arguments);
}

} // namespace fcarouge

#endif // FCAROUGE_KALMAN_HPP
