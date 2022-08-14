/*_  __          _      __  __          _   _
 | |/ /    /\   | |    |  \/  |   /\   | \ | |
 | ' /    /  \  | |    | \  / |  /  \  |  \| |
 |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
 | . \  / ____ \| |____| |  | |/ ____ \| |\  |
 |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.1.0
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

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace fcarouge
{
//! @brief Function object for providing an identity matrix.
//!
//! @todo Could we remove this for a standard facility? Perhaps a form of
//! std::integral_constant?
//!
//! @note Could this function object template be a variable template as proposed
//! in paper P2008R0 entitled "Enabling variable template template parameters"?
struct identity_matrix {
  //! @brief Returns `1`, the 1-by-1 identity matrix equivalent.
  //!
  //! @tparam Type The type template parameter of the value.
  //!
  //! @return The value `1`.
  template <typename Type>
  [[nodiscard]] inline constexpr auto operator()() const noexcept
  {
    return Type{ 1 };
  }
};

//! @brief Convenience tuple-like empty pack type.
using empty_pack = internal::empty_pack;

//! @brief Convenience tuple-like pack type.
template <typename... Types> using pack = internal::pack<Types...>;

//! @brief Kalman filter.
//!
//! @details A Bayesian filter that uses multivariate Gaussians.
//! Applicable for unimodal and uncorrelated uncertainties. Kalman filters
//! assume white noise, propagation and measurement functions are
//! differentiable, and that the uncertainty stays centered on the state
//! estimate. The filter updates estimates by multiplying Gaussians and predicts
//! estimates by adding Gaussians. Design the state (X, P), the process (F, Q),
//! the measurement (Z, R), the measurement function H, and if the system has
//! control inputs (U, B). Designing a filter is as much art as science.
//! Filters with `state x output x input` dimensions as 1x1x1 and 1x1x0 (no
//! input) are supported through the Standard Templated Library (STL). Higher
//! dimension filters require Eigen 3 support.
//!
//! @tparam Type The type template parameter of the filter data value type used
//! for computation. Defaults to `double`.
//! @tparam State The non-type template size of the state column vector x. State
//! variables can be observed (measured), or hidden variables (inferred). This
//! is the the mean of the multivariate Gaussian. Defaults to one `1` state.
//! @tparam Output The non-type template size of the measurement column vector
//! z. Defaults to one `1` output.
//! @tparam Input The non-type template size of the control column vector u. A
//! zero `0` input size value can be used for systems with no input control to
//! disable all of the input control features, the control transition matrix G
//! support, and the other related computations from the filter. Defaults to
//! no `0` input.
//! @tparam Transpose The customization point object template parameter of the
//! matrix transpose functor. Defaults to the standard passthrough
//! `std::identity` function object since the transposed value of an arithmetic
//! type is itself.
//! @tparam Symmetrize The customization point object template parameter of the
//! matrix symmetrization functor. Defaults to the standard passthrough
//! `std::identity` function object since the symmetric value of an arithmetic
//! type is itself.
//! @tparam Divide The customization point object template parameter of the
//! matrix division functor. Default to the standard division
//! `std::divides<void>` function object.
//! @tparam Identity The customization point object template parameter of the
//! matrix identity functor. Defaults to an `identity_matrix` function object
//! returning the arithmetic `1` value.
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
//! @todo Which constructors to support?
//! @todo Is the Kalman filter a recursive state estimation, confirm
//! terminology?
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
//! @todo Consider additional constructors?
//! @todo Consider additional characteristics method overloads?
//! @todo A clear or reset member equivalent may be useful for real-time
//! re-initializations but to what default?
//! @todo Could the Input be void by default? Or empty?
//! @todo Expand std::format support with standard arguments and Eigen3 types.
//! @todo Support complex number filters?
template <
    typename Type = double, std::size_t State = 1, std::size_t Output = 1,
    std::size_t Input = 0, typename Transpose = std::identity,
    typename Symmetrize = std::identity, typename Divide = std::divides<void>,
    typename Identity = identity_matrix, typename UpdateTypes = empty_pack,
    typename PredictionTypes = empty_pack>
class kalman
{
  private:
  //! @name Private Member Types
  //! @{

  //! @brief Implementation details of the filter.
  //!
  //! @brief The internal implementation unpacks the parameter packs from
  //! tuple-like types which allows for multiple parameter pack deductions.
  using implementation =
      internal::kalman<Type, State, Output, Input, Transpose, Symmetrize,
                       Divide, Identity, internal::repack_t<UpdateTypes>,
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

  //! @brief Type of the filter scalar data.
  using value_type = typename implementation::value_type;

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

  //! @brief Type of the callable observation state function.
  //!
  //! @details The function is of the form `output_model(const state &, const
  //! UpdateTypes &...)`.
  using observation_state_function =
      typename implementation::observation_state_function;

  //! @brief Type of the callable noise observation function.
  //!
  //! @details The function is of the form `output_uncertainty(const state &,
  //! const output &, const UpdateTypes &...)`.
  using noise_observation_function =
      typename implementation::noise_observation_function;

  //! @brief Type of the callable transition state function.
  //!
  //! @details The function is of the form `state_transition(const state &,
  //! const PredictionTypes &..., const input &)`.
  using transition_state_function =
      typename implementation::transition_state_function;

  //! @brief Type of the callable noise process function.
  //!
  //! @details The function is of the form `process_uncertainty(const state &,
  //! const PredictionTypes &...)`.
  using noise_process_function =
      typename implementation::noise_process_function;

  //! @brief Type of the callable transition control function.
  //!
  //! @details The function is of the form `input_control(const PredictionTypes
  //! &...)`.
  //!
  //! @todo Conditionally remove this member type when no input is present.
  using transition_control_function =
      typename implementation::transition_control_function;

  //! @brief Type of the callable transition function.
  //!
  //! @details The function is of the form `state(const state &, const
  //! PredictionTypes &...)`.
  using transition_function = typename implementation::transition_function;

  //! @brief Type of the callable observation function.
  //!
  //! @details The function is of the form `output(const state &, const
  //! UpdateTypes &...arguments)`.
  using observation_function = typename implementation::observation_function;

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
  //! filter using move semantics (i.e. the data in `other` filter is
  //! moved from the other into this filter). The other filter is in a
  //! valid but unspecified state afterwards.
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
  inline constexpr auto x() const -> state;

  //! @brief Sets the state estimate column vector X.
  //!
  //! @param value The copied state estimate column vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(const state &value);

  //! @brief Sets the state estimate column vector X.
  //!
  //! @param value The moved state estimate column vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(state &&value);

  //! @brief Sets the state estimate column vector X.
  //!
  //! @param value The first copied initializer used to set the state estimate
  //! column vector X.
  //! @param values The second and other copied initializers to set the state
  //! estimate column vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(const value_type &value,
                          const std::same_as<value_type> auto &...values);

  //! @brief Sets the state estimate column vector X.
  //!
  //! @param value The first moved initializer used to set the state estimate
  //! column vector X.
  //! @param values The second and other moved initializers to set the state
  //! estimate column vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(value_type &&value,
                          std::same_as<value_type> auto &&...values);

  //! @brief Returns the last observation column vector Z.
  //!
  //! @return The last observation column vector Z.
  //!
  //! @complexity Constant.
  inline constexpr auto z() const -> output;

  //! @brief Returns the last control column vector U.
  //!
  //! @details Not present when the filter has no input.
  //!
  //! @return The last control column vector U.
  //!
  //! @complexity Constant.
  inline constexpr auto u() const -> input requires(Input > 0);

  //! @brief Returns the estimated covariance matrix P.
  //!
  //! @return The estimated correlated variance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr auto p() const -> estimate_uncertainty;

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @param value The copied estimated covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(const estimate_uncertainty &value);

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @param value The moved estimated covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(estimate_uncertainty &&value);

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @param value The first copied initializer used to set the estimated
  //! covariance matrix P.
  //! @param values The second and other copied initializers to set the
  //! estimated covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(const value_type &value,
                          const std::same_as<value_type> auto &...values);

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @param value The first moved initializer used to set the estimated
  //! covariance matrix P.
  //! @param values The second and other moved initializers to set the estimated
  //! covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(value_type &&value,
                          std::same_as<value_type> auto &&...values);

  //! @brief Returns the process noise covariance matrix Q.
  //!
  //! @return The process noise correlated variance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr auto q() const -> process_uncertainty;

  //! @brief Sets the process noise covariance matrix Q.
  //!
  //! @param value The copied process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(const process_uncertainty &value);

  //! @brief Sets the process noise covariance matrix Q.
  //!
  //! @param value The moved process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(process_uncertainty &&value);

  //! @brief Sets the process noise covariance matrix Q.
  //!
  //! @param value The first copied initializer used to set the process noise
  //! covariance matrix Q.
  //! @param values The second and other copied initializers to set the process
  //! noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(const value_type &value,
                          const std::same_as<value_type> auto &...values);

  //! @brief Sets the process noise covariance matrix Q.
  //!
  //! @param value The first moved initializer used to set the process noise
  //! covariance matrix Q.
  //! @param values The second and other moved initializers used to set the
  //! process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(value_type &&value,
                          std::same_as<value_type> auto &&...values);

  //! @brief Sets the process noise covariance matrix Q function.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the process noise covariance matrix Q on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void q(const noise_process_function &callable);

  //! @brief Sets the process noise covariance matrix Q function.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the process noise covariance matrix Q on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void q(noise_process_function &&callable);

  //! @brief Returns the observation noise covariance
  //! matrix R.
  //!
  //! @details The variance there is in each measurement.
  //!
  //! @return The observation noise correlated variance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr auto r() const -> output_uncertainty;

  //! @brief Sets the observation noise covariance matrix R.
  //!
  //! @param value The copied observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(const output_uncertainty &value);

  //! @brief Sets the observation noise covariance matrix R.
  //!
  //! @param value The moved observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(output_uncertainty &&value);

  //! @brief Sets the observation noise covariance matrix R.
  //!
  //! @param value The first copied initializer used to set the observation
  //! noise covariance matrix R.
  //! @param values The second and other copied initializers to set the
  //! observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(const value_type &value,
                          const std::same_as<value_type> auto &...values);

  //! @brief Sets the observation noise covariance matrix R.
  //!
  //! @param value The first moved initializer used to set the observation noise
  //! covariance matrix R.
  //! @param values The second and other moved initializers to set the
  //! observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(value_type &&value,
                          std::same_as<value_type> auto &&...values);

  //! @brief Sets the observation noise covariance matrix R function.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be
  //! called by the filter to compute the observation noise covariance matrix R
  //! on prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void r(const noise_observation_function &callable);

  //! @brief Sets the observation noise covariance matrix R function.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be
  //! called by the filter to compute the observation noise covariance matrix R
  //! on prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void r(noise_observation_function &&callable);

  //! @brief Returns the state transition matrix F.
  //!
  //! @return The state transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr auto f() const -> state_transition;

  //! @brief Sets the state transition matrix F.
  //!
  //! @param value The copied state transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(const state_transition &value);

  //! @brief Sets the state transition matrix F.
  //!
  //! @param value The moved state transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(state_transition &&value);

  //! @brief Sets the state transition matrix F.
  //!
  //! @param value The first copied initializer used to set the state transition
  //! matrix F.
  //! @param values The second and other copied initializers to set the state
  //! transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(const value_type &value,
                          const std::same_as<value_type> auto &...values);

  //! @brief Sets the state transition matrix F.
  //!
  //! @param value The first moved initializer used to set the state transition
  //! matrix F.
  //! @param values The second and other moved initializers to set the state
  //! transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(value_type &&value,
                          std::same_as<value_type> auto &&...values);

  //! @brief Sets the state transition matrix F function.
  //!
  //! @details For non-linear system, or extended filter, F is the Jacobian of
  //! the state transition function: `F = ∂f/∂X = ∂fj/∂xi` that is each row i
  //! contains the derivatives of the state transition function for every
  //! element j in the state column vector X.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the state transition matrix F function on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void f(const transition_state_function &callable);

  //! @brief Sets the state transition matrix F function.
  //!
  //! @details For non-linear system, or extended filter, F is the Jacobian of
  //! the state transition function: `F = ∂f/∂X = ∂fj/∂xi` that is each row i
  //! contains the derivatives of the state transition function for every
  //! element j in the state column vector X.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the state transition matrix F function on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void f(transition_state_function &&callable);

  //! @brief Returns the observation transition matrix H.
  //!
  //! @return The observation, measurement transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr auto h() const -> output_model;

  //! @brief Sets the observation transition matrix H.
  //!
  //! @param value The copied observation transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(const output_model &value);

  //! @brief Sets the observation transition matrix H.
  //!
  //! @param value The moved observation transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(output_model &&value);

  //! @brief Sets the observation, measurement transition matrix H.
  //!
  //! @param value The first copied initializer used to set the observation,
  //! measurement transition matrix H.
  //! @param values The second and other copied initializers to set the
  //! observation, measurement transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(const value_type &value,
                          const std::same_as<value_type> auto &...values);

  //! @brief Sets the observation, measurement transition matrix H.
  //!
  //! @param value The first moved initializer used to set the observation,
  //! measurement transition matrix H.
  //! @param values The second and other moved initializers to set the
  //! observation, measurement transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(value_type &&value,
                          std::same_as<value_type> auto &&...values);

  //! @brief Sets the observation, measurement transition matrix H function.
  //!
  //! @details For non-linear system, or extended filter, H is the Jacobian of
  //! the state observation function: `H = ∂h/∂X = ∂hj/∂xi` that is each row i
  //! contains the derivatives of the state observation function for every
  //! element j in the state column vector X.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the observation, measurement transition
  //! matrix H on update steps.
  //!
  //! @complexity Constant.
  inline constexpr void h(const observation_state_function &callable);

  //! @brief Sets the observation, measurement transition matrix H function.
  //!
  //! @details For non-linear system, or extended filter, H is the Jacobian of
  //! the state observation function: `H = ∂h/∂X = ∂hj/∂xi` that is each row i
  //! contains the derivatives of the state observation function for every
  //! element j in the state column vector X.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the observation, measurement transition
  //! matrix H on update steps.
  //!
  //! @complexity Constant.
  inline constexpr void h(observation_state_function &&callable);

  //! @brief Returns the control transition matrix G.
  //!
  //! @return The control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr auto g() const -> input_control requires(Input > 0);

  //! @brief Sets the control transition matrix G.
  //!
  //! @param value The copied control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(const input_control &value) requires(Input > 0);

  //! @brief Sets the control transition matrix G.
  //!
  //! @param value The moved control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(input_control &&value) requires(Input > 0);

  //! @brief Sets the control transition matrix G.
  //!
  //! @param value The first copied initializer used to set the control
  //! transition matrix G.
  //! @param values The second and other copied initializers to set the control
  //! transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(const value_type &value,
                          const std::same_as<value_type> auto &...values);

  //! @brief Sets the control transition matrix G.
  //!
  //! @param value The first moved initializer used to set the control
  //! transition matrix G.
  //! @param values The second and other moved initializers to set the control
  //! transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(value_type &&value,
                          std::same_as<value_type> auto &&...values);

  //! @brief Sets the control transition matrix G function.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the control transition matrix G on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void g(const transition_control_function &callable);

  //! @brief Sets the control transition matrix G function.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the control transition matrix G on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void g(transition_control_function &&callable);

  //! @brief Returns the gain matrix K.
  //!
  //! @return The gain matrix K.
  //!
  //! @complexity Constant.
  inline constexpr auto k() const -> gain;

  //! @brief Returns the innovation column vector Y.
  //!
  //! @return The innovation column vector Y.
  //!
  //! @complexity Constant.
  inline constexpr auto y() const -> innovation;

  //! @brief Returns the innovation uncertainty matrix S.
  //!
  //! @return The innovation uncertainty matrix S.
  //!
  //! @complexity Constant.
  inline constexpr auto s() const -> innovation_uncertainty;

  //! @brief Sets the extended state transition function f(x).
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the next state X on
  //! prediction steps. The default function `f(x) = F * X` is suitable for
  //! linear systems. For non-linear system, or extended filter, implement a
  //! linearization of the transition function f and the state transition F
  //! matrix is the Jacobian of the state transition function.
  //!
  //! @complexity Constant.
  inline constexpr void transition(const transition_function &callable);

  //! @brief Sets the extended state transition function f(x).
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the next state X on
  //! prediction steps. The default function `f(x) = F * X` is suitable for
  //! linear systems. For non-linear system, or extended filter, implement a
  //! linearization of the transition function f and the state transition F
  //! matrix is the Jacobian of the state transition function.
  //!
  //! @complexity Constant.
  inline constexpr void transition(transition_function &&callable);

  //! @brief Sets the extended state observation function h(x).
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the observation Z
  //! on update steps. The default function `h(x) = H * X` is suitable for
  //! linear systems. For non-linear system, or extended filter, the client
  //! implements a linearization of the observation function hand the state
  //! observation H matrix is the Jacobian of the state observation function.
  //!
  //! @complexity Constant.
  inline constexpr void observation(const observation_function &callable);

  //! @brief Sets the extended state observation function h(x).
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the observation Z
  //! on update steps. The default function `h(x) = H * X` is suitable for
  //! linear systems. For non-linear system, or extended filter, the client
  //! implements a linearization of the observation function hand the state
  //! observation H matrix is the Jacobian of the state observation function.
  //!
  //! @complexity Constant.
  inline constexpr void observation(observation_function &&callable);

  //! @}

  //! @name Public Filtering Member Functions
  //! @{

  //! @brief Runs a step of the filter.
  //!
  //! @details Predicts and updates the estimates per prediction arguments,
  //! control input, and measurement output.
  //!
  //! @param arguments The prediction, update, input, and output parameters of
  //! the filter, in that order. The arguments need to be compatible with the
  //! filter types. The prediction parameters convertible to the
  //! `PredictionTypes` template pack types are passed through for computations
  //! of prediction matrices. The update parameters convertible to the
  //! `UpdateTypes` template pack types are passed through for computations of
  //! update matrices. The control parameter pack types convertible to the
  //! `Input` template type. The observation parameter pack types convertible to
  //! the `Output` template type. The update and prediction types are explicitly
  //! defined with the class definition and the observation parameter pack types
  //! are always deduced per the greedy matching rule. However the control
  //! parameter pack types must always be explicitly defined per the fair
  //! matching rule.
  //!
  //! @note Called as `k(...);` with prediction values and output values when
  //! the filter has no input parameters. The input type list is explicitly
  //! empty. Otherwise can be called as `k.template operator()<input1_t,
  //! input2_t, ...>(...);` with prediction values, input values, and output
  //! values. The input type list being explicitly specified per the fair
  //! matching rule. A lambda can come in handy to reduce the verbose call
  //! `const auto kf{ [&k](const auto
  //! &...args) { k.template operator()<input1_t, input2_t,
  //! ...>(args...); } };` then called as `kf(...);`.
  //!
  //! @todo Consider if returning the state column vector X would be preferable?
  //! Or fluent interface? Would be compatible with an ES-EKF implementation?
  //! @todo Understand why the implementation cannot be moved out of the class.
  //! @todo What should be the order of the parameters? Update first?
  template <typename... InputTypes>
  inline constexpr void operator()(const auto &...arguments);

  //! @brief Updates the estimates with the outcome of a measurement.
  //!
  //! @details Implements the Bayes' theorem. Combine one measurement and the
  //! prior estimate.
  //!
  //! @param arguments The update and output parameters of
  //! the filter, in that order. The arguments need to be compatible with the
  //! filter types. The update parameters convertible to the
  //! `UpdateTypes` template pack types are passed through for computations of
  //! update matrices. The observation parameter pack types convertible to
  //! the `Output` template type. The update types are explicitly
  //! defined with the class definition.
  //!
  //! @todo Consider whether this method needs to exist or if the operator() is
  //! sufficient for all clients?
  //! @todo Consider if returning the state column vector X would be preferable?
  //! Or fluent interface? Would be compatible with an ES-EKF implementation?
  inline constexpr void update(const auto &...arguments);

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
  //! @todo Consider whether this method needs to exist or if the operator() is
  //! sufficient for all clients?
  //! @todo Consider if returning the state column vector X would be preferable?
  //! Or fluent interface? Would be compatible with an ES-EKF implementation?
  inline constexpr void predict(const auto &...arguments);

  //! @}
};

} // namespace fcarouge

#include "internal/kalman.tpp"

#endif // FCAROUGE_KALMAN_HPP
