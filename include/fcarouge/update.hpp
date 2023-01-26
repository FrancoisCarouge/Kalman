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

#ifndef FCAROUGE_UPDATE_HPP
#define FCAROUGE_UPDATE_HPP

#include "internal/update.hpp"

//! @file
//! @brief The Kalman update model class.
//!
//! @details Provides the library public definitions of the update model.

namespace fcarouge {

//! @brief A generic Kalman update model.
template <typename Implementation> class update final {
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
  using state = typename implementation::state;

  //! @brief Type of the observation column vector Z.
  //!
  //! @details Also known as Y or O.
  using output = typename implementation::output;

  //! @brief Type of the estimated correlated variance matrix P.
  //!
  //! @details Also known as Σ.
  using estimate_uncertainty = typename implementation::estimate_uncertainty;

  //! @brief Type of the observation noise correlated variance matrix R.
  using output_uncertainty = typename implementation::output_uncertainty;

  //! @brief Type of the observation transition matrix H.
  //!
  //! @details Also known as the measurement transition matrix or C.
  using output_model = typename implementation::output_model;

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

  //! @brief Constructs a Kalman update model without configuration.
  //!
  //! @complexity Constant.
  inline constexpr update() = default;

  //! @brief Copy constructs an update model.
  //!
  //! @details Constructs the update model with the copy of the contents of the
  //! `other` update model.
  //!
  //! @param other Another update model to be used as source to initialize the
  //! elements of the update model with.
  //!
  //! @complexity Constant.
  //!
  //! @todo Implement and test.
  // inline constexpr udpate(const update &other) = delete;

  //! @brief Move constructs an update model.
  //!
  //! @details Move constructor. Constructs the update model with the contents
  //! of the `other` update model using move semantics (i.e. the data in `other`
  //! update model is moved from the other into this update model).
  //!
  //! @param other Another update model to be used as source to initialize the
  //! elements of the update model with.
  //!
  //! @complexity Constant.
  //!
  //! @todo Implement and test.
  inline constexpr update(update &&other) noexcept = delete;

  //! @brief Copy assignment operator.
  //!
  //! @details Destroys or copy-assigns the contents with a copy of the contents
  //! of the other update model.
  //!
  //! @param other Another update model to be used as source to initialize the
  //! elements of the update model with.
  //!
  //! @return The reference value of this implicit object update model
  //! parameter, i.e. `*this`.
  //!
  //! @complexity Constant.
  //!
  //! @todo Implement and test.
  inline constexpr auto operator=(const update &other) -> update & = delete;

  //! @brief Move assignment operator.
  //!
  //! @details Replaces the contents of the update model with those of the
  //! `other` update model using move semantics (i.e. the data in `other` update
  //! model is moved from the other into this update model). The other update
  //! model is in a valid but unspecified state afterwards.
  //!
  //! @param other Another update model to be used as source to initialize the
  //! elements of the update model with.
  //!
  //! @return The reference value of this implicit object update model
  //! parameter, i.e. `*this`.
  //!
  //! @complexity Constant.
  //!
  //! @todo Implement and test.
  inline constexpr auto operator=(update &&other) noexcept -> update & = delete;

  //! @brief Destructs the Kalman update model.
  //!
  //! @complexity Constant.
  inline constexpr ~update() = default;

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

  //! @brief Returns the last observation column vector Z.
  //!
  //! @return The last observation column vector Z.
  //!
  //! @complexity Constant.
  inline constexpr auto z() const -> const output &;

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
  //! member) that will be called by the update model to compute the observation
  //! noise covariance matrix R on prediction steps.
  //! @param values The optional second and other copied initializers to set the
  //! observation noise covariance matrix R.
  //!
  //! @complexity Constant.
  inline constexpr void r(const auto &value, const auto &...values);

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
  //! &...)`. For non-linear system, or extended update model, H is the Jacobian
  //! of the state observation function: `H = ∂h/∂X = ∂hj/∂xi` that is each row
  //! i contains the derivatives of the state observation function for every
  //! element j in the state column vector X.
  //!
  //! @param value The first copied initializer used to set the copied
  //! observation transition matrix H or the copied target Callable object
  //! (function object, pointer to function, reference to function, pointer to
  //! member function, or pointer to data member) that will be bound to the
  //! prediction arguments and called by the update model to compute the
  //! observation, measurement transition matrix H on update steps.
  //! @param values The optional second and other copied initializers to set the
  //! observation transition matrix H.
  //!
  //! @complexity Constant.
  inline constexpr void h(const auto &value, const auto &...values);

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
  //!
  //! @todo Add measurement post-fit residual by default and example?
  inline constexpr auto y() const -> const innovation &;

  //! @brief Returns the innovation uncertainty matrix S.
  //!
  //! @return The innovation uncertainty matrix S.
  //!
  //! @complexity Constant.
  inline constexpr auto s() const -> const innovation_uncertainty &;

  //! @brief Sets the extended state observation function h(x).
  //!
  //! @param callable The copied target Callable object (function object,
  //! pointer to function, reference to function, pointer to member function, or
  //! pointer to data member) that will be called to compute the observation Z
  //! on update steps of expression `output(const state &, const UpdateTypes
  //! &...arguments)`. The default function `h(x) = H * X` is suitable for
  //! linear systems. For non-linear system, or extended update model, the
  //! client implements a linearization of the observation function hand the
  //! state observation H matrix is the Jacobian of the state observation
  //! function.
  //!
  //! @complexity Constant.
  inline constexpr void observation(const auto &callable);

  //! @}

  //! @name Public Filtering Member Functions
  //! @{

  //! @brief Updates the estimates with the outcome of a measurement.
  //!
  //! @details Also known as the observation or correction step. Implements the
  //! Bayes' theorem. Combine one measurement and the prior estimate.
  //!
  //! @param arguments The update and output parameters of
  //! the update model, in that order. The arguments need to be compatible with
  //! the update model types. The update parameters convertible to the
  //! `UpdateTypes` template pack types are passed through for computations of
  //! update matrices. The observation parameter pack types convertible to
  //! the `Output` template type. The update types are explicitly
  //! defined with the class definition.
  //!
  //! @todo Consider if returning the state column vector X would be preferable?
  //! Or fluent interface? Would be compatible with an ES-EKF implementation?
  //! @todo Can the parameter pack of `UpdateTypes` be explicit in the method
  //! declaration for user clarity?
  inline constexpr void operator()(const auto &...arguments);

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
  template <std::size_t Position> inline constexpr auto operator()() const;

  //! @}
};

//! @name Deduction Guides
//! @{

update()->update<internal::update<>>;

//! @}

} // namespace fcarouge

#include "internal/update.tpp"

#endif // FCAROUGE_UPDATE_HPP
