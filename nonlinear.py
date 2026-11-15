"""Метод Ньютона для системы нелинейных уравнений."""

from dataclasses import dataclass

import numpy as np


class ConvergenceError(RuntimeError):
    pass


@dataclass
class Solution:
    value: np.ndarray
    residuals: list[float]

    @property
    def iterations(self) -> int:
        return len(self.residuals) - 1


def newton(
    function, jacobian, start, *, tolerance=1e-10, max_iterations=50,
    frozen=False, damped=False,
):
    """Направление находится из J Δ = −F. frozen фиксирует J в начальной точке."""
    x = np.asarray(start, dtype=float).copy()
    value = function(x)
    residual = float(np.linalg.norm(value, ord=np.inf))
    residuals = [residual]
    matrix = None

    for _ in range(max_iterations):
        if not np.isfinite(residual):
            raise ConvergenceError("Метод Ньютона: невязка не конечна")
        if residual <= tolerance:
            return Solution(x, residuals)
        if matrix is None or not frozen:
            matrix = jacobian(x)
        try:
            direction = np.linalg.solve(matrix, -np.asarray(value))
        except np.linalg.LinAlgError as error:
            raise ConvergenceError("Метод Ньютона: вырожденный якобиан") from error

        factor = 1.0
        candidate = x + direction
        value = function(candidate)
        new_residual = float(np.linalg.norm(value, ord=np.inf))
        if damped:
            while not new_residual < residual and factor > 2.0**-16:
                factor *= 0.5
                candidate = x + factor * direction
                value = function(candidate)
                new_residual = float(np.linalg.norm(value, ord=np.inf))
            if not new_residual < residual:
                raise ConvergenceError("Метод Ньютона: шаг не уменьшает невязку")
        x, residual = candidate, new_residual
        residuals.append(residual)

    if residual <= tolerance:
        return Solution(x, residuals)
    raise ConvergenceError("Метод Ньютона: исчерпан лимит итераций")
