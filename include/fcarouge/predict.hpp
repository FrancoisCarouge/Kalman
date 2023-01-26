/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
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

#ifndef FCAROUGE_PREDICT_HPP
#define FCAROUGE_PREDICT_HPP

//! @file
//! @brief The Kalman predict model class.
//!
//! @details Provides the library public definitions of the predict model.

namespace fcarouge {

//! @brief A generic Kalman predict model.
template <typename Implementation> class predict final {
private:
  //! @name Private Member Types
  //! @{

  //! @brief Implementation details of the model.
  using implementation = Implementation;

  //! @}

  //! @name Private Member Variables
  //! @{

  //! @brief Encapsulates the implementation details of the model.
  implementation model;

  //! @}

public:
  //! @name Public Member Types
  //! @{

  //! @brief Type of the state estimate column vector X.
  using state = typename Implementation::state;

  //! @brief Type of the control column vector U.
  //!
  //! @todo Conditionally remove this member type when no input is present.
  using input = typename Implementation::input;

  //! @brief Type of the estimated correlated variance matrix P.
  //!
  //! @details Also known as Σ.
  using estimate_uncertainty = typename Implementation::estimate_uncertainty;

  //! @brief Type of the process noise correlated variance matrix Q.
  using process_uncertainty = typename Implementation::process_uncertainty;

  //! @brief Type of the state transition matrix F.
  //!
  //! @details Also known as the fundamental matrix, propagation, Φ, or A.
  using state_transition = typename Implementation::state_transition;

  //! @brief Type of the control transition matrix G.
  //!
  //! @details Also known as B.
  //!
  //! @todo Conditionally remove this member type when no input is present.
  using input_control = typename Implementation::input_control;

  //! @}

  //! @name Public Member Functions
  //! @{

  //! @brief Constructs a Kalman prediction model without configuration.
  //!
  //! @complexity Constant.
  inline constexpr predict() = default;

  //! @brief Copy constructs a prediction model.
  //!
  //! @details Constructs the prediction model with the copy of the contents of
  //! the `other` prediction model.
  //!
  //! @param other Another prediction model to be used as source to initialize
  //! the elements of the prediction model with.
  //!
  //! @complexity Constant.
  //!
  //! @todo Implement and test.
  inline constexpr predict(const predict &other) = delete;

  //! @brief Move constructs a prediction model.
  //!
  //! @details Move constructor. Constructs the prediction model with the
  //! contents of the `other` prediction model using move semantics (i.e. the
  //! data in `other` prediction model is moved from the other into this
  //! prediction model).
  //!
  //! @param other Another prediction model to be used as source to initialize
  //! the elements of the prediction model with.
  //!
  //! @complexity Constant.
  //!
  //! @todo Implement and test.
  inline constexpr predict(predict &&other) noexcept = delete;

  //! @brief Copy assignment operator.
  //!
  //! @details Destroys or copy-assigns the contents with a copy of the contents
  //! of the other prediction model.
  //!
  //! @param other Another prediction model to be used as source to initialize
  //! the elements of the prediction model with.
  //!
  //! @return The reference value of this implicit object prediction model
  //! parameter, i.e. `*this`.
  //!
  //! @complexity Constant.
  //!
  //! @todo Implement and test.
  inline constexpr auto operator=(const predict &other) -> predict & = delete;

  //! @brief Move assignment operator.
  //!
  //! @details Replaces the contents of the prediction model with those of the
  //! `other` prediction model using move semantics (i.e. the data in `other`
  //! prediction model is moved from the other into this prediction model). The
  //! other prediction model is in a valid but unspecified state afterwards.
  //!
  //! @param other Another prediction model to be used as source to initialize
  //! the elements of the prediction model with.
  //!
  //! @return The reference value of this implicit object prediction model
  //! parameter, i.e. `*this`.
  //!
  //! @complexity Constant.
  //!
  //! @todo Implement and test.
  inline constexpr auto operator=(predict &&other) noexcept
      -> predict & = delete;

  //! @brief Destructs the Kalman prediction model.
  //!
  //! @complexity Constant.
  inline constexpr ~predict() = default;

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

  //! @brief Returns the last control column vector U.
  //!
  //! @details This member function is not present when the prediction model has
  //! no input.
  //!
  //! @return The last control column vector U.
  //!
  //! @complexity Constant.
  inline constexpr auto u() const -> const input &requires(
      not std::is_same_v<typename Implementation::input, void>);

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
  //! be bound to the prediction arguments and called by the prediction model to
  //! compute the process noise covariance matrix Q on prediction steps.
  //! @param values The optional second and other copied initializers to set the
  //! process noise covariance matrix Q.
  //!
  //! @complexity Constant.
  inline constexpr void q(const auto &value, const auto &...values);

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
  //! prediction model, F is the Jacobian of the state transition function: `F =
  //! ∂f/∂X = ∂fj/∂xi` that is each row i contains the derivatives of the state
  //! transition function for every element j in the state column vector X.
  //!
  //! @param value The first copied initializer used to set the copied state
  //! transition matrix F or the copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the prediction model to compute the state transition matrix F
  //! function on prediction steps.
  //! @param values The optional second and other copied initializers to set the
  //! state transition matrix F.
  //!
  //! @complexity Constant.
  inline constexpr void f(const auto &value, const auto &...values);

  //! @brief Returns the control transition matrix G.
  //!
  //! @details This member function is not present when the prediction model has
  //! no input.
  //!
  //! @return The control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr auto g() const -> const input_control &requires(
      not std::is_same_v<typename Implementation::input, void>);
  inline constexpr auto g() -> input_control &requires(
      not std::is_same_v<typename Implementation::input, void>);

  //! @brief Sets the control transition matrix G.
  //!
  //! @details The control transition matrix G is of type `input_control` and
  //! the function is of the form `input_control(const PredictionTypes &...)`.
  //! This member function is not present when the prediction model has no
  //! input.
  //!
  //! @param value The first copied initializer used to set the copied control
  //! transition matrix G or the copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be bound to the prediction arguments and
  //! called by the prediction model to compute the control transition matrix G
  //! on prediction steps.
  //! @param values The optional second and other copied initializers to set the
  //! control transition matrix G.
  //!
  //! @complexity Constant.
  inline constexpr void g(const auto &value, const auto &...values)
    requires(not std::is_same_v<typename Implementation::input, void>);

  //! @brief Sets the extended state transition function f(x).
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the next state X on
  //! prediction steps of expression `state(const state &, const input &, const
  //! PredictionTypes &...)`. The default function `f(x) = F * X` is suitable
  //! for linear systems. For non-linear system, or extended prediction model,
  //! implement a linearization of the transition function f and the state
  //! transition F matrix is the Jacobian of the state transition function.
  //!
  //! @complexity Constant.
  inline constexpr void transition(const auto &callable);

  //! @}

  //! @name Public Filtering Member Functions
  //! @{

  //! @brief Produces estimates of the state variables and uncertainties.
  //!
  //! @details Implements the total probability theorem.
  //!
  //! @param arguments The prediction and input parameters of
  //! the prediction model, in that order. The arguments need to be compatible
  //! with the prediction model types. The prediction parameters convertible to
  //! the `PredictionTypes` template pack types are passed through for
  //! computations of prediction matrices. The control parameter pack types
  //! convertible to the `typename Implementation::input` template type. The
  //! prediction types are explicitly defined with the class definition.
  //!
  //! @todo Consider if returning the state column vector X would be preferable?
  //! Or fluent interface? Would be compatible with an ES-EKF implementation?
  //! @todo Can the parameter pack of `PredictionTypes` be explicit in the
  //! method declaration for user clarity?
  inline constexpr void operator()(const auto &...arguments);

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
  template <std::size_t Position> inline constexpr auto operator()() const;

  //! @}
};

} // namespace fcarouge

#include "internal/predict.tpp"

#endif // FCAROUGE_PREDICT_HPP
