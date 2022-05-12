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

namespace fcarouge
{
//! @brief Kalman filter.
//!
//! @details A Bayesian filter that uses multivariate Gaussians. Kalman filters
//! update estimates by multiplying Gaussians and predict estimates by adding
//! Gaussians. Design the state (x, P), the process (F, Q), the measurement (z,
//! R), the measurement function H, and if the system has control inputs (u, B).
//! Designing a filter is as much art as science. Kalman filters assume white
//! noise.
//!
//! @tparam Type The type template parameter of the value type of the filter.
//! @tparam State The type template parameter of the state vector x. State
//! variables can be observed (measured), or hidden variables (infeered). This
//! is the the mean of the multivariate Gaussian.
//! @tparam Output The type template parameter of the measurement vector z.
//! @tparam Input The type template parameter of the control u.
//! @tparam Transpose The customization point object template parameter of the
//! matrix transpose functor.
//! @tparam Symmetrize The customization point object template parameter of the
//! matrix symmetrization functor.
//! @tparam Divide The customization point object template parameter of the
//! matrix division functor.
//! @tparam Identity The customization point object template parameter of the
//! matrix identity functor.
//! @tparam PredictionArguments The variadic type template parameter for
//! additional prediction function parameters. Time, or a delta thereof, is
//! often a prediction parameter. The parameters are propagated to the function
//! objects used to compute the process noise Q, the state transition F, and the
//! control transition G matrices.
//!
//! @note This class could be usable in constant expressions if `std::function`
//! could too. The polymorphic function wrapper was used in place of function
//! pointers to enable default initialization from this class, captured member
//! variables.
//!
//! @todo Is this filter restricted to Newton's equations of motion? That is
//! only a discretized continuous-time kinematic filter? How about non-Newtonian
//! systems?
//! @todo Would it be beneficial to support `Type` and `value_type` prior to the
//! `State` type template parameter?
//! @todo Would it be beneficial to support initialization list for
//! characteristis?
//! @todo Symmetrization support might be superflous. How to confirm it is safe
//! to remove?
//! @todo Would we want to support smoothers?
//! @todo How to add or associate constraints on the types and operation to
//! support compilation and semantics?
template <
    typename Type = double, typename State = Type, typename Output = State,
    typename Input = State, typename Transpose = std::identity,
    typename Symmetrize = std::identity, typename Divide = std::divides<void>,
    typename Identity = internal::identity, typename... PredictionArguments>
class kalman
{
  private:
  //! @name Private Member Types
  //! @{

  //! @brief Implementation details of the filter.
  using implementation =
      internal::kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                       Identity, PredictionArguments...>;

  //! @}

  public:
  //! @name Public Member Types
  //! @{

  //! @brief The type of the filtered data elements.
  using value_type = State;

  //! @brief Type of the state estimate vector X.
  using state = typename implementation::state;

  //! @brief Type of the observation vector Z.
  //!
  //! @details Also known as Y.
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
  //! @details Also known as the fundamental matrix, Φ, or A.
  using state_transition = typename implementation::state_transition;

  //! @brief Type of the observation transition matrix H.
  //!
  //! @details Also known as the measurement transition matrix or C.
  using output_model = typename implementation::output_model;

  //! @brief Type of the control transition matrix G.
  //!
  //! @details Also known as B.
  using input_control = typename implementation::input_control;

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
  inline constexpr kalman &operator=(const kalman &other) = default;

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
  inline constexpr kalman &operator=(kalman &&other) noexcept = default;

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
              "discarded.")]] inline constexpr state
  x() const;

  //! @brief Sets the state estimate vector X.
  //!
  //! @complexity Constant.
  //!
  //! @todo Consider if a fluent interface would be preferrable? In addition to
  //! constructors? Same question for all characteristics set methods.
  inline constexpr void x(const auto &value, const auto &...values);

  //! @brief Returns the observation vector Z.
  //!
  //! @return The observation vector Z.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned observation vector Z is unexpectedly "
              "discarded.")]] inline constexpr output
  z() const;

  //! @brief Returns the control vector U.
  //!
  //! @return The control vector U.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned control vector U is unexpectedly "
              "discarded.")]] inline constexpr input
  u() const;

  //! @brief Returns the estimated covariance matrix P.
  //!
  //! @return The estimated correlated variance matrix P.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned estimated covariance matrix P is unexpectedly "
              "discarded.")]] inline constexpr estimate_uncertainty
  p() const;

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(const auto &value, const auto &...values);

  //! @brief Returns the process noise covariance matrix Q.
  //!
  //! @return The process noise correlated variance matrix Q.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
              "discarded.")]] inline constexpr process_uncertainty
  q() const;

  //! @brief Sets the process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(const auto &value, const auto &...values);

  //! @brief Sets the process noise covariance matrix Q function.
  //!
  //! @complexity Constant.
  inline constexpr void q(const auto &value) requires std::constructible_from <
      std::function<process_uncertainty(const PredictionArguments &...)>,
  decltype(value) >
  {
    filter.noise_process_q = value;
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
              "unexpectedly discarded.")]] inline constexpr output_uncertainty
  r() const;

  //! @brief Sets the observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(const auto &value, const auto &...values);

  //! @brief Sets the observation noise covariance matrix R function.
  //!
  //! @complexity Constant.
  inline constexpr void r(const auto &value) requires std::constructible_from <
      std::function<output_uncertainty()>,
  decltype(value) >
  {
    filter.noise_observation_r = value;
  }

  //! @brief Returns the state transition matrix F.
  //!
  //! @return The state transition matrix F.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned state transition matrix F is unexpectedly "
              "discarded.")]] inline constexpr state_transition
  f() const;

  //! @brief Sets the state transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(const auto &value, const auto &...values);

  //! @brief Sets the state transition matrix F function.
  //!
  //! @complexity Constant.
  inline constexpr void f(const auto &value) requires std::constructible_from <
      std::function<state_transition(const PredictionArguments &...)>,
  decltype(value) >
  {
    filter.transition_state_f = value;
  }

  //! @brief Returns the observation transition matrix H.
  //!
  //! @return The observation, measurement transition matrix H.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned observation transition matrix H is unexpectedly "
              "discarded.")]] inline constexpr output_model
  h() const;

  //! @brief Sets the observation, measurement transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(const auto &value, const auto &...values);

  //! @brief Sets the observation, measurement transition matrix H function.
  //!
  //! @complexity Constant.
  inline constexpr void h(const auto &value) requires std::constructible_from <
      std::function<output_model(const PredictionArguments &...)>,
  decltype(value) >
  {
    filter.transition_observation_h = value;
  }

  //! @brief Returns the control transition matrix G.
  //!
  //! @return The control transition matrix G.
  //!
  //! @complexity Constant.
  [[nodiscard("The returned control transition matrix G is unexpectedly "
              "discarded.")]] inline constexpr input_control
  g() const;

  //! @brief Sets the control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(const auto &value, const auto &...values);

  //! @brief Sets the control transition matrix G function.
  //!
  //! @complexity Constant.
  inline constexpr void g(const auto &value) requires std::constructible_from <
      std::function<input_control(const PredictionArguments &...)>,
  decltype(value) >
  {
    filter.transition_control_g = value;
  }

  //! @}

  //! @name Public Filtering Member Functions
  //! @{

  //! @brief Runs a step of the filter.
  //!
  //! @details Predicts and updates the estimates per prediction arguments,
  //! control input, and measurement output.
  //!
  //! @param arguments Optional prediction parameters passed through for
  //! computations of prediction matrices.
  //! @param input_u Optional control parameters. Types must be compatible with
  //! the `Input` types. The parameter pack types must always be explicitely
  //! defined per the
  // fair matching rule.
  //! @param output_z Observation parameters. Types must be compatible with the
  //! `output` type. The parameter pack types are always deduced per the greedy
  //! matching rule.
  //!
  //! @note Called as `k(...);` with prediction values and output values when
  //! the filter has no input parameters. The input type list is explicitely
  //! empty. Otherwise can be called as `k.template operator()<input1_t,
  //! input2_t, ...>(...);` with prediction values, input values, and output
  //! values. The input type list being explicitely specified. A lambda can come
  //! in handy to reduce the verbose call `const auto kf{ [&k](const auto
  //! &...args) { k.template operator()<input1_t, input2_t,
  //! ...>(args...); } };` then called as `kf(...);`.
  //!
  //! @todo Consider if returning the state vector X would be preferrable? And
  //! if it would be compatible with an ES-EKF implementation? Or if a fluent
  //! interface would be preferrable?
  inline constexpr void operator()(const PredictionArguments &...arguments,
                                   const auto &...input_u,
                                   const auto &...output_z)
  {
    filter.predict(arguments..., input_u...);
    filter.update(output_z...);
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
  //! sufficient for all clients.
  //! @todo Consider if returning the state vector X would be preferrable? And
  //! if it would be compatible with an ES-EKF implementation? Or if a fluent
  //! interface would be preferrable?
  inline constexpr void update(const auto &...output_z);

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
  //! sufficient for all clients.
  //! @todo Consider if returning the state vector X would be preferrable? And
  //! if it would be compatible with an ES-EKF implementation? Or if a fluent
  //! interface would be preferrable?
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
          typename Identity, typename... PredictionArguments>
inline constexpr
    typename kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::state
    kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::x() const
{
  return filter.x;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::x(const auto &value, const auto &...values)
{
  filter.x = state{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr
    typename kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::estimate_uncertainty
    kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::p() const
{
  return filter.p;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::p(const auto &value, const auto &...values)
{
  filter.p = estimate_uncertainty{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr
    typename kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::process_uncertainty
    kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::q() const
{
  return filter.q;
}

//! @todo Don't we need to reset functions or values when the other is set?
template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::q(const auto &value, const auto &...values)
{
  filter.q = process_uncertainty{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr
    typename kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::output_uncertainty
    kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::r() const
{
  return filter.r;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::r(const auto &value, const auto &...values)
{
  filter.r = output_uncertainty{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr
    typename kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::state_transition
    kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::f() const
{
  return filter.f;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::f(const auto &value, const auto &...values)
{
  filter.f = state_transition{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr
    typename kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::output_model
    kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::h() const
{
  return filter.h;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::h(const auto &value, const auto &...values)
{
  filter.h = output_model{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr
    typename kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::input_control
    kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::g() const
{
  return filter.g;
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::g(const auto &value, const auto &...values)
{
  filter.g = input_control{ value, values... };
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::update(const auto &...output_z)
{
  filter.update(output_z...);
}

template <typename Type, typename State, typename Output, typename Input,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... PredictionArguments>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::predict(const PredictionArguments &...arguments,
                                        const auto &...input_u)
{
  filter.predict(arguments..., input_u...);
}

} // namespace fcarouge

#endif // FCAROUGE_KALMAN_HPP
