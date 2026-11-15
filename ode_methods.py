"""Схемы с постоянным шагом для задачи Коши y' = f(t, y)."""

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Callable

import numpy as np

from nonlinear import ConvergenceError, newton


RK_ORDERS = {f"RK{order}": order for order in range(1, 5)}
ADAMS = {
    "AB1": np.array([1.0]),
    "AB2": np.array([3.0, -1.0]) / 2,
    "AB3": np.array([23.0, -16.0, 5.0]) / 12,
    "AB4": np.array([55.0, -59.0, 37.0, -9.0]) / 24,
}
RADAU_A = np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]])
RADAU_C = np.array([1 / 3, 1.0])
IMPLICIT_METHODS = ("Radau3", "BDF2")
NEWTON_TOLERANCE = 5e-15


@dataclass(frozen=True)
class InitialValueProblem:
    rhs: Callable[[float, np.ndarray], np.ndarray]
    jacobian: Callable[[float, np.ndarray], np.ndarray]
    initial: np.ndarray
    start: float
    end: float


@dataclass
class Trajectory:
    times: np.ndarray
    values: np.ndarray
    step: float
    failed_at: float | None = None
    failure: str | None = None


def runge_kutta_step(rhs, t, y, h, order, first_stage=None):
    k1 = rhs(t, y) if first_stage is None else first_stage
    if order == 1:
        return y + h * k1
    k2 = rhs(t + h / 2, y + h / 2 * k1)
    if order == 2:
        return y + h * k2
    if order == 3:
        k3 = rhs(t + h, y + h * (2 * k2 - k1))
        return y + h / 6 * (k1 + 4 * k2 + k3)
    k3 = rhs(t + h / 2, y + h / 2 * k2)
    k4 = rhs(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def radau_step(problem, t, y, h, identity):
    """Двухстадийный Radau IIA. Последняя стадия даёт y на новом слое."""
    dimension = y.size
    stage_times = t + h * RADAU_C

    def residual(flat_stages):
        stages = flat_stages.reshape(2, dimension)
        derivatives = np.array([
            problem.rhs(stage_times[j], stages[j]) for j in range(2)
        ])
        return (stages - y - h * (RADAU_A @ derivatives)).ravel()

    def jacobian(flat_stages):
        stages = flat_stages.reshape(2, dimension)
        derivatives = [problem.jacobian(stage_times[j], stages[j]) for j in range(2)]
        matrix = np.empty((2 * dimension, 2 * dimension))
        for i in range(2):
            rows = slice(i * dimension, (i + 1) * dimension)
            for j in range(2):
                columns = slice(j * dimension, (j + 1) * dimension)
                matrix[rows, columns] = -h * RADAU_A[i, j] * derivatives[j]
                if i == j:
                    matrix[rows, columns] += identity
        return matrix

    result = newton(residual, jacobian, np.tile(y, 2), tolerance=NEWTON_TOLERANCE, damped=True)
    return result.value[dimension:]


def bdf2_step(problem, time_next, previous, current, h, identity):
    center = (4 * current - previous) / 3
    factor = 2 * h / 3

    def residual(value):
        return value - center - factor * problem.rhs(time_next, value)

    def jacobian(value):
        return identity - factor * problem.jacobian(time_next, value)

    return newton(residual, jacobian, current, tolerance=NEWTON_TOLERANCE, damped=True).value


def integrate(problem, method, steps, *, state_limit=np.inf, adams_start_substeps=1):
    if steps < 1 or problem.end <= problem.start:
        raise ValueError("Нужны положительный интервал и хотя бы один шаг")
    if adams_start_substeps < 1:
        raise ValueError("Число подшагов запуска должно быть положительным")
    rk_order = RK_ORDERS.get(method)
    adams_weights = ADAMS.get(method)
    if rk_order is None and adams_weights is None and method not in IMPLICIT_METHODS:
        raise ValueError(f"Неизвестный метод {method}")

    times = np.linspace(problem.start, problem.end, steps + 1)
    initial = np.asarray(problem.initial, dtype=float)
    values = np.empty((steps + 1, initial.size))
    values[0] = initial
    h = (problem.end - problem.start) / steps
    identity = np.eye(initial.size)
    rhs = problem.rhs
    if adams_weights is not None:
        adams_order = len(adams_weights)
        history = deque(maxlen=adams_order)
        weights = h * adams_weights
        small_h = h / adams_start_substeps

    with np.errstate(over="ignore", invalid="ignore"):
        for k, t in enumerate(times[:-1]):
            current = values[k]
            try:
                if rk_order is not None:
                    candidate = runge_kutta_step(rhs, t, current, h, rk_order)
                elif adams_weights is not None:
                    derivative = rhs(t, current)
                    history.appendleft(derivative)
                    if k < adams_order - 1:
                        # Мелкие шаги РК заполняют начальные узлы грубой сетки.
                        candidate = current
                        for j in range(adams_start_substeps):
                            first_stage = derivative if j == 0 else None
                            candidate = runge_kutta_step(
                                rhs, t + j * small_h, candidate, small_h, adams_order, first_stage,
                            )
                    else:
                        candidate = current + weights @ np.asarray(history)
                elif method == "Radau3" or k == 0:
                    candidate = radau_step(problem, t, current, h, identity)
                else:
                    candidate = bdf2_step(problem, times[k + 1], values[k - 1], current, h, identity)
            except ConvergenceError as error:
                return Trajectory(times[:k + 1], values[:k + 1], h, times[k + 1], str(error))

            magnitude = float(np.max(np.abs(candidate)))
            if not isfinite(magnitude) or magnitude > state_limit:
                return Trajectory(times[:k + 1], values[:k + 1], h, times[k + 1], "выход за пределы решения")
            values[k + 1] = candidate

    return Trajectory(times, values, h)
