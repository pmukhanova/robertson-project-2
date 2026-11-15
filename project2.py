"""Проект 2. Система Робертсона, задача X.9.7 практикума."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from ode_methods import IMPLICIT_METHODS, InitialValueProblem, integrate


LARGE_STEP = 0.01
ADAMS_START_STEP = 0.0001
FINE_STEPS = {
    "RK1": 0.0005,
    "RK2": 0.0005,
    "RK3": 0.0005,
    "RK4": 0.0005,
    "AB1": 0.0005,
    "AB2": 0.00025,
    "AB3": 0.000125,
    "AB4": 0.00008,
}
IMPLICIT_STEPS = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625)
HORIZONS = (40.0, *(float(10**power) for power in range(2, 12)))
LONG_GRIDS = {300: 0.05, 600: 0.025}
METHOD_LABELS = {"Radau3": "Radau IIA 3", "BDF2": "Гир 2 (BDF2)"}
COLORS = {"reference": "#222222", "RK1": "#985408", "RK4": "#985408", "Radau3": "#28598a", "BDF2": "#15704c"}


def robertson_rhs(t, y):
    y1, y2, y3 = y
    transfer_12 = 0.04 * y1
    transfer_21 = 1.0e4 * y2 * y3
    transfer_23 = 3.0e7 * y2**2
    return np.array([-transfer_12 + transfer_21, transfer_12 - transfer_21 - transfer_23, transfer_23])


def robertson_jacobian(t, y):
    _, y2, y3 = y
    return np.array([
        [-0.04, 1.0e4 * y3, 1.0e4 * y2],
        [0.04, -1.0e4 * y3 - 6.0e7 * y2, -1.0e4 * y2],
        [0.0, 6.0e7 * y2, 0.0],
    ])


def run(problem, method, h, *, adams_start_substeps=1):
    steps = round((problem.end - problem.start) / h)
    return integrate(
        problem, method, steps, state_limit=10.0,
        adams_start_substeps=adams_start_substeps,
    )


def reference_solution(problem):
    reference = solve_ivp(
        problem.rhs, (problem.start, problem.end), problem.initial,
        method="Radau", jac=problem.jacobian, rtol=2e-13, atol=1e-15, dense_output=True,
    )
    if not reference.success:
        raise RuntimeError(reference.message)
    return reference.sol


def point_error(numerical, reference):
    return np.max(np.abs(numerical - reference), axis=-1)


def write_summary(directory, cases, reference):
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "series", "method", "h", "steps_completed", "stopped_at", "end_error",
            "mass_defect", "min_concentration", "y1_end", "y2_end", "y3_end",
        ])
        for series, method, trajectory in cases:
            times, values = trajectory.times, trajectory.values
            mass_defect = np.max(np.abs(np.sum(values, axis=1) - 1))
            end_error = point_error(values[-1], reference(times[-1])) if trajectory.failed_at is None else None
            end_state = values[-1] if end_error is not None else ("", "", "")
            writer.writerow([
                series, method, trajectory.step, len(times) - 1, trajectory.failed_at,
                end_error, mass_defect, np.min(values), *end_state,
            ])
            detail = f"E(T)={end_error:.3e}" if end_error is not None else f"остановка при t={trajectory.failed_at:g}"
            print(f"{series:12s} {method:7s} h={trajectory.step:<10g} {detail}", flush=True)

    with (directory / "reference.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "y1", "y2", "y3", "lambda_fast", "lambda_slow"])
        for time in (0.0, 0.001, 0.01, 1.0, 40.0):
            state = reference(time)
            eigenvalues = np.linalg.eigvals(robertson_jacobian(time, state))
            by_magnitude = sorted(eigenvalues, key=abs, reverse=True)
            writer.writerow([time, *state, by_magnitude[0], by_magnitude[1]])


def plot_profiles(directory, reference, fine, implicit, early):
    figure, axes = plt.subplots(2, 2, figsize=(8.25, 5.45), layout="constrained")
    full_times = np.linspace(0, 40, 701)
    full_states = reference(full_times).T
    trajectories = (
        ("RK4", fine["RK4"], "--", "", 0),
        ("Radau3", implicit["Radau3", 0.2], "-", "o", 8),
        ("BDF2", implicit["BDF2", 0.05], "-.", "s", 0),
    )
    components = ((0, 1, "$y_1(t)$"), (2, 1, "$y_3(t)$"), (1, 1e5, "$10^5 y_2(t)$"))
    for axis, (component, scale, ylabel) in zip(axes.flat, components):
        axis.plot(full_times, scale * full_states[:, component], color=COLORS["reference"], lw=2, alpha=0.65, label="Опорное решение")
        for method, trajectory, linestyle, marker, marker_offset in trajectories:
            stride = max(1, len(trajectory.times) // 315)
            stride_slice = slice(None, None, stride)
            method_label = METHOD_LABELS.get(method, method)
            step_label = f"{trajectory.step:g}".replace(".", ",")
            axis.plot(
                trajectory.times[stride_slice], scale * trajectory.values[stride_slice, component],
                color=COLORS[method], linestyle=linestyle, lw=1.1, marker=marker, markersize=3,
                markevery=(marker_offset, max(1, len(trajectory.times[stride_slice]) // 10)),
                label=f"{method_label}, h = {step_label}",
            )
        axis.set(xlabel="$t$", ylabel=ylabel, xlim=(0, 40))
        axis.grid(alpha=0.23)

    inset = axes[1, 1]
    times_early = np.linspace(0, 0.008, 301)
    inset.plot(times_early, 1e5 * reference(times_early)[1], color=COLORS["reference"], lw=1.7, label="Опорное решение")
    for method, trajectory in early.items():
        inset.plot(
            trajectory.times, 1e5 * trajectory.values[:, 1],
            color=COLORS[method], lw=1.1, linestyle="--" if method == "RK4" else "-.",
            marker="o" if method == "Radau3" else "s", markersize=2.5,
            label=f"{METHOD_LABELS.get(method, method)}, h = {trajectory.step:g}".replace(".", ","),
        )
    inset.set(xlabel="$t$", ylabel="$10^5 y_2(t)$", xlim=(0, 0.008), title="Первые 0,008")
    inset.legend(loc="lower right", frameon=False, fontsize=8)
    inset.grid(alpha=0.23)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.savefig(directory / "profiles.png", dpi=220)
    plt.close(figure)


def plot_stability(directory, reference, coarse, fine):
    figure, axis = plt.subplots(figsize=(8.2, 3.5), layout="constrained")
    stop = coarse["RK1"].failed_at
    limit = 0.08 if stop is None else max(0.055, min(stop + 0.02, 0.2))
    for method, trajectory, linestyle, label in (
        ("RK1", coarse["RK1"], "--", "Эйлер, h = 0,01"),
        ("Radau3", coarse["Radau3"], "-", "Radau IIA 3, h = 0,01"),
        ("BDF2", coarse["BDF2"], "-.", "Гир 2, h = 0,01"),
        ("RK1", fine["RK1"], "-", "Эйлер, h = 0,0005"),
    ):
        draw = (trajectory.times > 0) & (trajectory.times <= limit)
        times = trajectory.times[draw]
        if times.size == 0:
            continue
        errors = point_error(trajectory.values[draw], reference(times).T)
        axis.semilogy(
            times, np.where(errors > 0, errors, np.nan),
            color=COLORS[method], linestyle=linestyle,
            marker="o" if trajectory.step == LARGE_STEP else "", markersize=4, label=label,
        )
    if stop is not None:
        axis.axvline(stop, color="0.45", lw=0.9, linestyle=":", label="Остановка Эйлера при h = 0,01")
    axis.set(xlim=(0, limit), xlabel="$t$", ylabel=r"$\max_i |y_i(t)-y_i^{\mathrm{ref}}(t)|$")
    axis.grid(alpha=0.23)
    axis.legend(loc="upper right", ncol=2, framealpha=1, facecolor="white", edgecolor="none", fontsize=9)
    figure.savefig(directory / "stability.png", dpi=220)
    plt.close(figure)


def plot_refinement(directory, reference, implicit):
    figure, axis = plt.subplots(figsize=(8.0, 3.55), layout="constrained")
    for method, marker in (("Radau3", "o"), ("BDF2", "s")):
        steps, errors = [], []
        for h in sorted(IMPLICIT_STEPS):
            result = implicit[method, h]
            if result.failed_at is None:
                steps.append(h)
                errors.append(point_error(result.values[-1], reference(result.times[-1])))
        axis.loglog(steps, errors, marker=marker, color=COLORS[method], label=METHOD_LABELS[method])
    axis.set(xlabel="Шаг $h$", ylabel="Погрешность в конечной точке $E(40)$")
    axis.grid(which="major", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(directory / "refinement.png", dpi=220)
    plt.close(figure)


def long_study(directory, problem, implicit):
    reference = solve_ivp(
        problem.rhs, (problem.start, HORIZONS[-1]), problem.initial,
        method="Radau", jac=problem.jacobian, rtol=2e-13,
        atol=[1e-17, 1e-23, 1e-15], t_eval=HORIZONS,
    )
    if not reference.success:
        raise RuntimeError(reference.message)

    errors = {}
    with (directory / "long_horizons.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "method", "steps_per_segment", "t", "h", "y1", "y2", "y3",
            "ref_y1", "ref_y2", "ref_y3", "end_error", "relative_error_y1",
            "mass_defect", "min_concentration",
        ])
        for method in IMPLICIT_METHODS:
            for section_steps, initial_h in LONG_GRIDS.items():
                solution = implicit[method, initial_h]
                local_errors = []
                for index, end_time in enumerate(HORIZONS):
                    if index:
                        segment = replace(
                            problem, start=HORIZONS[index - 1], end=end_time,
                            initial=solution.values[-1],
                        )
                        solution = integrate(segment, method, section_steps, state_limit=10.0)
                    if solution.failed_at is not None:
                        raise RuntimeError(f"{method}, t={solution.failed_at:g}: {solution.failure}")
                    value = solution.values[-1]
                    ref_value = reference.y[:, index]
                    error = point_error(value, ref_value)
                    local_errors.append(error)
                    writer.writerow([
                        method, section_steps, end_time, solution.step, *value, *ref_value,
                        error, abs((value[0] - ref_value[0]) / ref_value[0]),
                        np.max(np.abs(solution.values.sum(axis=1) - 1)), np.min(solution.values),
                    ])
                errors[method, section_steps] = local_errors
                print(f"{method:7s} t={HORIZONS[-1]:g} N={section_steps}, E(T)={local_errors[-1]:.3e}", flush=True)

    figure, (states, refinement) = plt.subplots(1, 2, figsize=(8.4, 3.35), layout="constrained")
    states.loglog(HORIZONS, reference.y[0], "-o", color="#28598a", label="$y_1$")
    states.loglog(HORIZONS, reference.y[1], "--s", color="#985408", label="$y_2$")
    states.set(xlabel="$t$", ylabel="Опорная концентрация")
    states.legend(frameon=False)
    for (method, section_steps), error in errors.items():
        refinement.loglog(
            HORIZONS, error, "--" if section_steps == 300 else "-",
            color=COLORS[method], marker="o" if method == "Radau3" else "s",
            markersize=3.5, label=f"{METHOD_LABELS[method]}, {section_steps}",
        )
    refinement.set(xlabel="$t$", ylabel="Разность с опорным решением")
    refinement.legend(frameon=False, fontsize=8.5)
    for axis in (states, refinement):
        axis.grid(alpha=0.25)
    figure.savefig(directory / "long_horizons.png", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results_project2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10.5})

    initial = np.array([1.0, 0.0, 0.0])
    problem = InitialValueProblem(robertson_rhs, robertson_jacobian, initial, 0.0, 40.0)
    reference = reference_solution(problem)
    coarse = {
        method: run(problem, method, LARGE_STEP, adams_start_substeps=round(LARGE_STEP / ADAMS_START_STEP))
        for method in (*FINE_STEPS, *IMPLICIT_METHODS)
    }
    fine = {method: run(problem, method, h) for method, h in FINE_STEPS.items()}
    implicit = {(method, h): run(problem, method, h) for method in IMPLICIT_METHODS for h in IMPLICIT_STEPS}
    early_problem = InitialValueProblem(robertson_rhs, robertson_jacobian, initial, 0.0, 0.008)
    early = {method: run(early_problem, method, h) for method, h in [("RK4", 0.0005), ("Radau3", 0.002)]}

    cases = [
        *(("общий шаг", method, result) for method, result in coarse.items()),
        *(("мелкий шаг", method, result) for method, result in fine.items()),
        *(("сгущение", method, result) for (method, _), result in implicit.items()),
    ]
    write_summary(args.output, cases, reference)
    plot_profiles(args.output, reference, fine, implicit, early)
    plot_stability(args.output, reference, coarse, fine)
    plot_refinement(args.output, reference, implicit)
    long_study(args.output, problem, implicit)


if __name__ == "__main__":
    main()
