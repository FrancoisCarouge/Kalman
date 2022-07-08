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

#include "internal/kalman.hpp"

#include <concepts>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fcarouge
{
//! @brief Function object for providing an identity matrix.
using identity_matrix = internal::identity_matrix;

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
//!
//! @tparam Type The type template parameter of the value type of the filter.
//! @tparam State The type template parameter of the state vector X. State
//! variables can be observed (measured), or hidden variables (inferred). This
//! is the the mean of the multivariate Gaussian.
//! @tparam Output The type template parameter of the measurement vector Z.
//! @tparam Input The type template parameter of the control U.
//! @tparam Transpose The customization point object template parameter of the
//! matrix transpose functor.
//! @tparam Symmetrize The customization point object template parameter of the
//! matrix symmetrization functor.
//! @tparam Divide The customization point object template parameter of the
//! matrix division functor.
//! @tparam Identity The customization point object template parameter of the
//! matrix identity functor.
//! @tparam UpdateArguments The variadic type template parameter for additional
//! update function parameters. Parameters such as delta times, variances, or
//! linearized values. The parameters are propagated to the function objects
//! used to compute the state observation H and the observation noise R
//! matrices. The parameters are also propagated to the state observation
//! function object h.
//! @tparam PredictionArguments The variadic type template parameter for
//! additional prediction function parameters. Parameters such as delta times,
//! variances, or linearized values. The parameters are propagated to the
//! function objects used to compute the process noise Q, the state transition
//! F, and the control transition G matrices. The parameters are also propagated
//! to the state transition function object f.
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
//! @todo Prepare support for std::format?
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
//! @todo Could we do away with std::tuple, replaced by a template template?
//! @todo A clear or reset member equivalent may be useful for real-time
//! re-initializations but to what default?
//! @todo Could the Input be void by default? Or empty?
template <
    typename Type = double, typename State = Type, typename Output = State,
    typename Input = State, typename Transpose = std::identity,
    typename Symmetrize = std::identity, typename Divide = std::divides<void>,
    typename Identity = identity_matrix,
    typename UpdateArguments = std::tuple<>,
    typename PredictionArguments = std::tuple<>>
class kalman
{
};

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
class kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
             Identity, std::tuple<UpdateArguments...>,
             std::tuple<PredictionArguments...>>
{
  private:
  //! @name Private Member Types
  //! @{

  //! @brief Implementation details of the filter.
  using implementation =
      internal::kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                       Identity, std::tuple<UpdateArguments...>,
                       std::tuple<PredictionArguments...>>;

  //! @}

  public:
  //! @name Public Member Types
  //! @{

  //! @brief The type of the filtered data elements.
  using value_type = Type;

  //! @brief Type of the state estimate vector X.
  using state = typename implementation::state;

  //! @brief Type of the observation vector Z.
  //!
  //! @details Also known as Y or O.
  using output = typename implementation::output;

  //! @brief Type of the control vector U.
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
  using input_control = typename implementation::input_control;

  //! @brief Type of the gain matrix K.
  using gain = typename implementation::gain;

  //! @brief Type of the innovation vector Y.
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

  //! @brief Returns the state estimate vector X.
  //!
  //! @return The state estimate vector X.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned state estimate vector X is unexpectedly "
              "discarded.")]] inline constexpr auto
  x() const -> state;

  //! @brief Sets the state estimate vector X.
  //!
  //! @param value The copied state estimate vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(const state &value);

  //! @brief Sets the state estimate vector X.
  //!
  //! @param value The moved state estimate vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(state &&value);

  //! @brief Sets the state estimate vector X.
  //!
  //! @param value The first copied initializer used to set the state estimate
  //! vector X.
  //! @param values The second and other copied initializers to set the state
  //! estimate vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(const auto &value, const auto &...values);

  //! @brief Sets the state estimate vector X.
  //!
  //! @param value The first moved initializer used to set the state estimate
  //! vector X.
  //! @param values The second and other moved initializers to set the state
  //! estimate vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(auto &&value, auto &&...values);

  //! @brief Returns the observation vector Z.
  //!
  //! @return The observation vector Z.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned observation vector Z is unexpectedly "
              "discarded.")]] inline constexpr auto
  z() const -> output;

  //! @brief Returns the control vector U.
  //!
  //! @return The control vector U.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned control vector U is unexpectedly "
              "discarded.")]] inline constexpr auto
  u() const -> input;

  //! @brief Returns the estimated covariance matrix P.
  //!
  //! @return The estimated correlated variance matrix P.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned estimated covariance matrix P is unexpectedly "
              "discarded.")]] inline constexpr auto
  p() const -> estimate_uncertainty;

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
  inline constexpr void p(const auto &value, const auto &...values);

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @param value The first moved initializer used to set the estimated
  //! covariance matrix P.
  //! @param values The second and other moved initializers to set the estimated
  //! covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(auto &&value, auto &&...values);

  //! @brief Returns the process noise covariance matrix Q.
  //!
  //! @return The process noise correlated variance matrix Q.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
              "discarded.")]] inline constexpr auto
  q() const -> process_uncertainty;

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
  inline constexpr void q(const auto &value, const auto &...values);

  //! @brief Sets the process noise covariance matrix Q.
  //!
  //! @param value The first moved initializer used to set the process noise
  //! covariance matrix Q.
  //! @param values The second and other moved initializers used to set the
  //! process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(auto &&value, auto &&...values);

  //! @brief Sets the process noise covariance matrix Q function.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the process noise covariance matrix Q on
  //! prediction steps.
  //!
  //! @complexity Constant.
  //!
  //! @todo Understand why Clang Tidy doesn't find the out-of-line definition.
  inline constexpr void q(const auto &callable) requires std::is_invocable_r_v <
      process_uncertainty,
      std::decay_t<decltype(callable)>,
  const state &, const PredictionArguments &... >
  {
    filter.noise_process_q = callable;
  }

  //! @brief Sets the process noise covariance matrix Q function.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the process noise covariance matrix Q on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void
      q(auto &&callable) requires std::is_invocable_r_v < process_uncertainty,
      std::decay_t<decltype(callable)>,
  const state &, const PredictionArguments &... >
  {
    filter.noise_process_q = std::forward<decltype(callable)>(callable);
  }

  //! @brief Returns the observation noise covariance
  //! matrix R.
  //!
  //! @details The variance there is in each measurement.
  //!
  //! @return The observation noise correlated variance matrix R.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned observation noise covariance matrix R is "
              "unexpectedly discarded.")]] inline constexpr auto
  r() const -> output_uncertainty;

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
  inline constexpr void r(const auto &value, const auto &...values);

  //! @brief Sets the observation noise covariance matrix R.
  //!
  //! @param value The first moved initializer used to set the observation noise
  //! covariance matrix R.
  //! @param values The second and other moved initializers to set the
  //! observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(auto &&value, auto &&...values);

  //! @brief Sets the observation noise covariance matrix R function.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be
  //! called by the filter to compute the observation noise covariance matrix R
  //! on prediction steps.
  //!
  //! @complexity Constant.
  //!
  //! @todo Understand why Clang Tidy doesn't find the out-of-line definition.
  inline constexpr void r(const auto &callable) requires std::is_invocable_r_v <
      output_uncertainty,
      std::decay_t<decltype(callable)>,
  const state &, const output &, const UpdateArguments &... >
  {
    filter.noise_observation_r = callable;
  }

  //! @brief Sets the observation noise covariance matrix R function.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be
  //! called by the filter to compute the observation noise covariance matrix R
  //! on prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void
      r(auto &&callable) requires std::is_invocable_r_v < output_uncertainty,
      std::decay_t<decltype(callable)>,
  const state &, const output &, const UpdateArguments &... >
  {
    filter.noise_observation_r = std::forward<decltype(callable)>(callable);
  }

  //! @brief Returns the state transition matrix F.
  //!
  //! @return The state transition matrix F.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned state transition matrix F is unexpectedly "
              "discarded.")]] inline constexpr auto
  f() const -> state_transition;

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
  inline constexpr void f(const auto &value, const auto &...values);

  //! @brief Sets the state transition matrix F.
  //!
  //! @param value The first moved initializer used to set the state transition
  //! matrix F.
  //! @param values The second and other moved initializers to set the state
  //! transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(auto &&value, auto &&...values);

  //! @brief Sets the state transition matrix F function.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the state transition matrix F function on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void
      f(const auto &callable) requires std::is_invocable_r_v < state_transition,
      std::decay_t<decltype(callable)>,
  const state &, const PredictionArguments &..., const input & >
  {
    filter.transition_state_f = callable;
  }

  //! @brief Sets the state transition matrix F function.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the state transition matrix F function on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void
      f(auto &&callable) requires std::is_invocable_r_v < state_transition,
      std::decay_t<decltype(callable)>,
  const state &, const PredictionArguments &..., const input & >
  {
    filter.transition_state_f = std::forward<decltype(callable)>(callable);
  }

  //! @brief Returns the observation transition matrix H.
  //!
  //! @return The observation, measurement transition matrix H.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned observation transition matrix H is unexpectedly "
              "discarded.")]] inline constexpr auto
  h() const -> output_model;

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
  inline constexpr void h(const auto &value, const auto &...values);

  //! @brief Sets the observation, measurement transition matrix H.
  //!
  //! @param value The first moved initializer used to set the observation,
  //! measurement transition matrix H.
  //! @param values The second and other moved initializers to set the
  //! observation, measurement transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(auto &&value, auto &&...values);

  //! @brief Sets the observation, measurement transition matrix H function.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the observation, measurement transition
  //! matrix H on update steps.
  //!
  //! @complexity Constant.
  //!
  //! @todo Understand why Clang Tidy doesn't find the out-of-line definition.
  inline constexpr void
      h(const auto &callable) requires std::is_invocable_r_v < output_model,
      std::decay_t<decltype(callable)>,
  const state &, const UpdateArguments &... >
  {
    filter.observation_state_h = callable;
  }

  //! @brief Sets the observation, measurement transition matrix H function.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the observation, measurement transition
  //! matrix H on update steps.
  //!
  //! @complexity Constant.
  inline constexpr void
      h(auto &&callable) requires std::is_invocable_r_v < output_model,
      std::decay_t<decltype(callable)>,
  const state &, const UpdateArguments &... >
  {
    filter.observation_state_h = std::forward<decltype(callable)>(callable);
  }

  //! @brief Returns the control transition matrix G.
  //!
  //! @return The control transition matrix G.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned control transition matrix G is unexpectedly "
              "discarded.")]] inline constexpr auto
  g() const -> input_control;

  //! @brief Sets the control transition matrix G.
  //!
  //! @param value The copied control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(const input_control &value);

  //! @brief Sets the control transition matrix G.
  //!
  //! @param value The moved control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(input_control &&value);

  //! @brief Sets the control transition matrix G.
  //!
  //! @param value The first copied initializer used to set the control
  //! transition matrix G.
  //! @param values The second and other copied initializers to set the control
  //! transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(const auto &value, const auto &...values);

  //! @brief Sets the control transition matrix G.
  //!
  //! @param value The first moved initializer used to set the control
  //! transition matrix G.
  //! @param values The second and other moved initializers to set the control
  //! transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(auto &&value, auto &&...values);

  //! @brief Sets the control transition matrix G function.
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the control transition matrix G on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void
      g(const auto &callable) requires std::is_invocable_r_v < input_control,
      std::decay_t<decltype(callable)>,
  const PredictionArguments &... >
  {
    filter.transition_control_g = callable;
  }

  //! @brief Sets the control transition matrix G function.
  //!
  //! @param callable The moved target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the filter to compute the control transition matrix G on
  //! prediction steps.
  //!
  //! @complexity Constant.
  inline constexpr void
      g(auto &&callable) requires std::is_invocable_r_v < input_control,
      std::decay_t<decltype(callable)>,
  const PredictionArguments &... >
  {
    filter.transition_control_g = std::forward<decltype(callable)>(callable);
  }

  //! @brief Returns the gain matrix K.
  //!
  //! @return The gain matrix K.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned gain matrix K is unexpectedly "
              "discarded.")]] inline constexpr auto
  k() const -> gain;

  //! @brief Returns the innovation vector Y.
  //!
  //! @return The innovation vector Y.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned innovation vector Y is unexpectedly "
              "discarded.")]] inline constexpr auto
  y() const -> innovation;

  //! @brief Returns the innovation uncertainty matrix S.
  //!
  //! @return The innovation uncertainty matrix S.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned innovation uncertainty matrix S is unexpectedly "
              "discarded.")]] inline constexpr auto
  s() const -> innovation_uncertainty;

  inline constexpr void
      transition(const auto &callable) requires std::is_invocable_r_v < state,
      std::decay_t<decltype(callable)>,
  const state &, const PredictionArguments &... >
  {
    filter.transition = callable;
  }

  inline constexpr void
      transition(auto &&callable) requires std::is_invocable_r_v < state,
      std::decay_t<decltype(callable)>,
  const state &, const PredictionArguments &... >
  {
    filter.transition = std::forward<decltype(callable)>(callable);
  }

  inline constexpr void
      observation(const auto &callable) requires std::is_invocable_r_v < output,
      std::decay_t<decltype(callable)>,
  const state &, const UpdateArguments &... >
  {
    filter.observation = callable;
  }

  inline constexpr void
      observation(auto &&callable) requires std::is_invocable_r_v < output,
      std::decay_t<decltype(callable)>,
  const state &, const UpdateArguments &... >
  {
    filter.observation = std::forward<decltype(callable)>(callable);
  }

  //! @}

  //! @name Public Filtering Member Functions
  //! @{

  //! @brief Runs a step of the filter.
  //!
  //! @details Predicts and updates the estimates per prediction arguments,
  //! control input, and measurement output.
  //!
  //! @param prediction_arguments Optional prediction parameters passed through
  //! for computations of prediction matrices.
  //! @param update_arguments Optional update parameters passed through for
  //! computations of update matrices.
  //! @param input_u Optional control parameters. Types must be compatible with
  //! the `Input` types. The parameter pack types must always be explicitly
  //! defined per the
  // fair matching rule.
  //! @param output_z Observation parameters. Types must be compatible with the
  //! `output` type. The parameter pack types are always deduced per the greedy
  //! matching rule.
  //!
  //! @note Called as `k(...);` with prediction values and output values when
  //! the filter has no input parameters. The input type list is explicitly
  //! empty. Otherwise can be called as `k.template operator()<input1_t,
  //! input2_t, ...>(...);` with prediction values, input values, and output
  //! values. The input type list being explicitly specified. A lambda can come
  //! in handy to reduce the verbose call `const auto kf{ [&k](const auto
  //! &...args) { k.template operator()<input1_t, input2_t,
  //! ...>(args...); } };` then called as `kf(...);`.
  //!
  //! @todo Consider if returning the state vector X would be preferable? Or
  //! fluent interface? Would be compatible with an ES-EKF implementation?
  //! @todo Understand why the implementation cannot be moved out of the class.
  //! @todo What should be the order of the parameters? Update first?
  inline constexpr void
  operator()(const PredictionArguments &...prediction_arguments,
             const UpdateArguments &...update_arguments, const auto &...input_u,
             const auto &...output_z)
  {
    filter.update(update_arguments..., output_z...);
    filter.predict(prediction_arguments..., input_u...);
  }

  //! @brief Updates the estimates with the outcome of a measurement.
  //!
  //! @details Implements the Bayes' theorem. Combine one measurement and the
  //! prior estimate.
  //!
  //! @tparam output_z Observation parameters. Types must be compatible with the
  //! `output` type.
  //!
  //! @todo Consider whether this method needs to exist or if the operator() is
  //! sufficient for all clients?
  //! @todo Consider if returning the state vector X would be preferable? Or
  //! fluent interface? Would be compatible with an ES-EKF implementation?
  inline constexpr void update(const UpdateArguments &...arguments,
                               const auto &...output_z);

  //! @brief Produces estimates of the state variables and uncertainties.
  //!
  //! @details Implements the total probability theorem.
  //!
  //! @param arguments Optional prediction parameters passed through for
  //! computations of prediction matrices.
  //! @param input_u Optional control parameters. Types must be compatible with
  //! the `Input` types. The parameter pack types are always deduced per the
  //! greedy matching rule.
  //!
  //! @todo Consider whether this method needs to exist or if the operator() is
  //! sufficient for all clients?
  //! @todo Consider if returning the state vector X would be preferable? Or
  //! fluent interface? Would be compatible with an ES-EKF implementation?
  inline constexpr void predict(const PredictionArguments &...arguments,
                                const auto &...input_u);

  //! @}

  private:
  //! @name Private Member Variables
  //! @{

  //! @brief Encapsulates the implementation details of the filter.
  implementation filter;

  //! @}
};

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::x()
    const -> state
{
  return filter.x;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::x(const state &value)
{
  filter.x = value;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::x(state &&value)
{
  filter.x = std::move(value);
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::x(const auto &value,
                                              const auto &...values)
{
  filter.x = std::move(state{ value, values... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::x(auto &&value, auto &&...values)
{
  filter.x = std::move(state{ std::forward<decltype(value)>(value),
                              std::forward<decltype(values)>(values)... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::z()
    const -> output
{
  return filter.z;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::u()
    const -> input
{
  return filter.u;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::p()
    const -> estimate_uncertainty
{
  return filter.p;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::p(const estimate_uncertainty &value)
{
  filter.p = value;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::p(estimate_uncertainty &&value)
{
  filter.p = std::move(value);
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::p(const auto &value,
                                              const auto &...values)
{
  filter.p = std::move(estimate_uncertainty{ value, values... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::p(auto &&value, auto &&...values)
{
  filter.p = std::move(
      estimate_uncertainty{ std::forward<decltype(value)>(value),
                            std::forward<decltype(values)>(values)... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::q()
    const -> process_uncertainty
{
  return filter.q;
}

//! @todo Don't we need to reset functions or values when the other is set?
template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::q(const process_uncertainty &value)
{
  filter.q = value;
}

//! @todo Don't we need to reset functions or values when the other is set?
template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::q(process_uncertainty &&value)
{
  filter.q = std::move(value);
}

//! @todo Don't we need to reset functions or values when the other is set?
template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::q(const auto &value,
                                              const auto &...values)
{
  filter.q = std::move(process_uncertainty{ value, values... });
}

//! @todo Don't we need to reset functions or values when the other is set?
template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::q(auto &&value, auto &&...values)
{
  filter.q = std::move(
      process_uncertainty{ std::forward<decltype(value)>(value),
                           std::forward<decltype(values)>(values)... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::r()
    const -> output_uncertainty
{
  return filter.r;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::r(const output_uncertainty &value)
{
  filter.r = value;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::r(output_uncertainty &&value)
{
  filter.r = std::move(value);
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::r(const auto &value,
                                              const auto &...values)
{
  filter.r = output_uncertainty{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::r(auto &&value, auto &&...values)
{
  filter.r = std::move(
      output_uncertainty{ std::forward<decltype(value)>(value),
                          std::forward<decltype(values)>(values)... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::f()
    const -> state_transition
{
  return filter.f;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::f(const state_transition &value)
{
  filter.f = value;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::f(state_transition &&value)
{
  filter.f = std::move(value);
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::f(const auto &value,
                                              const auto &...values)
{
  filter.f = state_transition{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::f(auto &&value, auto &&...values)
{
  filter.f =
      std::move(state_transition{ std::forward<decltype(value)>(value),
                                  std::forward<decltype(values)>(values)... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::h()
    const -> output_model
{
  return filter.h;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::h(const output_model &value)
{
  filter.h = value;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::h(output_model &&value)
{
  filter.h = std::move(value);
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::h(const auto &value,
                                              const auto &...values)
{
  filter.h = output_model{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::h(auto &&value, auto &&...values)
{
  filter.h =
      std::move(output_model{ std::forward<decltype(value)>(value),
                              std::forward<decltype(values)>(values)... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::g()
    const -> input_control
{
  return filter.g;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::g(const input_control &value)
{
  filter.g = value;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::g(input_control &&value)
{
  filter.g = std::move(value);
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::g(const auto &value,
                                              const auto &...values)
{
  filter.g = input_control{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>,
       std::tuple<PredictionArguments...>>::g(auto &&value, auto &&...values)
{
  filter.g =
      std::move(input_control{ std::forward<decltype(value)>(value),
                               std::forward<decltype(values)>(values)... });
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::k()
    const -> gain
{
  return filter.k;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::y()
    const -> innovation
{
  return filter.y;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::s()
    const -> innovation_uncertainty
{
  return filter.s;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::
    update(const UpdateArguments &...arguments, const auto &...output_z)
{
  filter.update(arguments..., output_z...);
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateArguments,
          typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       std::tuple<UpdateArguments...>, std::tuple<PredictionArguments...>>::
    predict(const PredictionArguments &...arguments, const auto &...input_u)
{
  filter.predict(arguments..., input_u...);
}

} // namespace fcarouge

#endif // FCAROUGE_KALMAN_HPP
