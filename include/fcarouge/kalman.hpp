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

namespace fcarouge
{
//! @brief Kalman filter.
//!
//! @tparam State The type template parameter of the state vector x.
//! @tparam Output The type template parameter of the measurement vector z.
//! @tparam Input The type template parameter of the control u.
//! @tparam Transpose The template template parameter of the transpose functor.
//! @tparam Divide The template template parameter of the division functor.
//! @tparam Identity The template template parameter of the identity functor.
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
template <typename State, typename Output = State, typename Input = State,
          template <typename> typename Transpose = internal::transpose,
          template <typename> typename Symmetrize = internal::symmetrize,
          template <typename, typename> typename Divide = internal::divide,
          template <typename> typename Identity = internal::identity,
          typename... PredictionArguments>
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

  //! @brief Type of the state estimate vector X.
  using state = typename implementation::state;

  //! @brief Type of the observation vector Z.
  //!
  //! @details Also known as Y.
  using output = typename implementation::output;

  //! @brief Type of the control vector U.
  using input = typename implementation::input;

  //! @brief Type of the estimated covariance matrix P.
  //!
  //! @details Also known as Σ.
  using estimate_uncertainty = typename implementation::estimate_uncertainty;

  //! @brief Type of the process noise covariance matrix Q.
  using process_uncertainty = typename implementation::process_uncertainty;

  //! @brief Type of the observation, measurement noise covariance matrix R.
  using output_uncertainty = typename implementation::output_uncertainty;

  //! @brief Type of the state transition matrix F.
  //!
  //! @details Also known as Φ or A.
  using state_transition = typename implementation::state_transition;

  //! @brief Type of the observation transition matrix H.
  //!
  //! @details Also known as C.
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
  constexpr kalman() = default;

  //! @brief Copy constructs a filter.
  //!
  //! @details Constructs the filter with the copy of the contents of the
  //! `other` filter.
  //!
  //! @param other Another filter to be used as source to initialize the
  //! elements of the filter with.
  //!
  //! @complexity Constant.
  constexpr kalman(const kalman &other) = default;

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
  constexpr kalman(kalman &&other) noexcept = default;

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
  constexpr kalman &operator=(const kalman &other) = default;

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
  constexpr kalman &operator=(kalman &&other) noexcept = default;

  //! @brief Destructs the kalman filter.
  //!
  //! @complexity Constant.
  constexpr ~kalman() = default;

  //! @}

  //! @name Public Characteristics Member Functions
  //! @{

  //! @brief Returns the state estimate vector X.
  //!
  //! @return The state estimate vector X.
  //!
  //! @complexity Constant.
  [[nodiscard]] inline constexpr state x() const;

  //! @brief Sets the state estimate vector X.
  //!
  //! @complexity Constant.
  inline constexpr void x(const auto &value, const auto &...values);

  //! @brief Returns the observation vector Z.
  //!
  //! @return The observation vector Z.
  //!
  //! @complexity Constant.
  [[nodiscard]] inline constexpr output z() const;

  //! @brief Returns the control vector U.
  //!
  //! @return The control vector U.
  //!
  //! @complexity Constant.
  [[nodiscard]] inline constexpr input u() const;

  //! @brief Returns the estimated covariance matrix P.
  //!
  //! @return The estimated covariance matrix P.
  //!
  //! @complexity Constant.
  [[nodiscard]] inline constexpr estimate_uncertainty p() const;

  //! @brief Sets the estimated covariance matrix P.
  //!
  //! @complexity Constant.
  inline constexpr void p(const auto &value, const auto &...values);

  //! @brief Returns the process noise covariance matrix Q.
  //!
  //! @return The process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  [[nodiscard]] inline constexpr process_uncertainty q() const;

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

  //! @brief Returns the observation, measurement noise covariance matrix R.
  //!
  //! @return The observation, measurement noise covariance matrix R.
  //!
  //! @complexity Constant.
  [[nodiscard]] inline constexpr output_uncertainty r() const;

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
  [[nodiscard]] inline constexpr state_transition f() const;

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
  //! @return The observation transition matrix H.
  //!
  //! @complexity Constant.
  [[nodiscard]] inline constexpr output_model h() const;

  //! @brief Sets the observation transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(const auto &value, const auto &...values);

  //! @brief Sets the observation transition matrix H function.
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
  [[nodiscard]] inline constexpr input_control g() const;

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

  //! @name Public Modifiers Member Functions
  //! @{

  //! @brief Updates the estimates with the outcome of a measurement.
  //!
  //! @tparam output_z Observation parameters. Types must be compatible with the
  //! `output` type.
  inline constexpr void observe(const auto &...output_z);

  //! @brief Produces estimates of the state variables and uncertainties.
  //!
  //! @param arguments Optional prediction parameters passed through for
  //! computations of prediction matrices.
  //! @param input_u Optional control parameters. Types must be compatible with
  //! the `Input` types.
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

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
[[nodiscard]] inline constexpr
    typename kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::state
    kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::x() const
{
  return filter.x;
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::x(const auto &value, const auto &...values)
{
  filter.x = state{ value, values... };
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
[[nodiscard]] inline constexpr
    typename kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::estimate_uncertainty
    kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::p() const
{
  return filter.p;
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::p(const auto &value, const auto &...values)
{
  filter.p = estimate_uncertainty{ value, values... };
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
[[nodiscard]] inline constexpr
    typename kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::process_uncertainty
    kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::q() const
{
  return filter.q;
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::q(const auto &value, const auto &...values)
{
  filter.q = process_uncertainty{ value, values... };
  // +reset function
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
[[nodiscard]] inline constexpr
    typename kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::output_uncertainty
    kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::r() const
{
  return filter.r;
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::r(const auto &value, const auto &...values)
{
  filter.r = output_uncertainty{ value, values... };
  // +reset function
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
[[nodiscard]] inline constexpr
    typename kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::state_transition
    kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::f() const
{
  return filter.f;
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::f(const auto &value, const auto &...values)
{
  filter.f = state_transition{ value, values... };
  // +reset function
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
[[nodiscard]] inline constexpr
    typename kalman<State, Output, Input, Transpose, Symmetrize, Divide,
                    Identity, PredictionArguments...>::output_model
    kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
           PredictionArguments...>::h() const
{
  return filter.h;
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::h(const auto &value, const auto &...values)
{
  filter.h = output_model{ value, values... };
  // +reset function
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::g(const auto &value, const auto &...values)
{
  filter.g = input_control{ value, values... };
  // +reset function
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::observe(const auto &...output_z)
{
  filter.observe(output_z...);
}

template <typename State, typename Output, typename Input,
          template <typename> typename Transpose,
          template <typename> typename Symmetrize,
          template <typename, typename> typename Divide,
          template <typename> typename Identity,
          typename... PredictionArguments>
inline constexpr void
kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       PredictionArguments...>::predict(const PredictionArguments &...arguments,
                                        const auto &...input_u)
{
  filter.predict(arguments..., input_u...);
}

} // namespace fcarouge

#endif // FCAROUGE_KALMAN_HPP
