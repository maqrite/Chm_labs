"""Задание 2.2: решение системы нелинейных уравнений F(x) = 0.

Метод простой итерации (x = Phi(x)) и метод Ньютона. Размерность системы
берётся из файла данных и нигде в коде не фиксируется.
"""

import itertools
import math

from expr import DomainError
from linalg import (SingularMatrixError, mat_vec, matrix_norm_inf, norm_inf,
                    solve_gauss, vec_scale, vec_sub)

__all__ = [
    "SystemResult", "numeric_jacobian", "estimate_q_system",
    "system_simple_iteration", "system_newton",
]

_DIVERGENCE_LIMIT = 1e12


class SystemResult(object):
    def __init__(self, method, converged, root, history, message="", q=None,
                 stop_rule=""):
        self.method = method
        self.converged = converged
        self.root = root
        self.history = history
        self.message = message
        self.q = q
        self.stop_rule = stop_rule

    @property
    def iterations(self):
        return max(len(self.history) - 1, 0)


def numeric_jacobian(F, x, h=1e-7):
    """Якобиан по центральным разностям: J[i][j] = dF_i/dx_j."""
    n = len(x)
    m = len(F(x))
    J = [[0.0] * n for _ in range(m)]
    for j in range(n):
        step = h * max(1.0, abs(x[j]))
        xp = list(x); xp[j] += step
        xm = list(x); xm[j] -= step
        fp = F(xp)
        fm = F(xm)
        for i in range(m):
            J[i][j] = (fp[i] - fm[i]) / (2.0 * step)
    return J


def estimate_q_system(PHI, region, samples=21, max_points=20000):
    """Оценка q = max по области ||Phi'(x)||_inf.

    region -- список отрезков [[a1,b1], [a2,b2], ...] по каждой переменной.
    Достаточное условие сходимости метода простой итерации: q < 1.
    """
    n = len(region)
    while samples > 2 and samples ** n > max_points:
        samples -= 1
    axes = []
    for (a, b) in region:
        if samples == 1 or a == b:
            axes.append([a])
        else:
            axes.append([a + (b - a) * i / (samples - 1) for i in range(samples)])

    best = 0.0
    for point in itertools.product(*axes):
        try:
            J = numeric_jacobian(PHI, list(point))
        except DomainError:
            continue
        value = matrix_norm_inf(J)
        if math.isfinite(value):
            best = max(best, value)
    return best


def check_region(PHI, region, samples=21, max_points=20000):
    """Проверка Phi(G) подмножество G по сетке внутри области G."""
    n = len(region)
    while samples > 2 and samples ** n > max_points:
        samples -= 1
    axes = []
    for (a, b) in region:
        axes.append([a + (b - a) * i / (samples - 1) for i in range(samples)]
                    if samples > 1 else [a])

    lows = [math.inf] * n
    highs = [-math.inf] * n
    for point in itertools.product(*axes):
        try:
            y = PHI(list(point))
        except DomainError:
            return False, None, None
        for i in range(n):
            lows[i] = min(lows[i], y[i])
            highs[i] = max(highs[i], y[i])
    inside = all(region[i][0] <= lows[i] and highs[i] <= region[i][1] for i in range(n))
    return inside, lows, highs


def system_simple_iteration(PHI, x0, eps, q=None, max_iter=500, F=None):
    """Метод простой итерации для системы: x^{k+1} = Phi(x^k).

    Критерий остановки -- та же оценка, что и в скалярном случае, но в норме:
        ||x^k - x*|| <= q/(1-q) * ||x^k - x^{k-1}|| <= eps.
    """
    if q is not None and 0.0 < q < 1.0:
        factor = q / (1.0 - q)
        rule = "q/(1-q)*||x^k - x^(k-1)||_inf <= eps,  q = %.6f" % q
    else:
        factor = 1.0
        rule = "||x^k - x^(k-1)||_inf <= eps"

    x = [float(v) for v in x0]
    history = [{"k": 0, "x": list(x), "dx": None, "est": None,
                "residual": (norm_inf(F(x)) if F is not None else None)}]

    for k in range(1, max_iter + 1):
        try:
            x_new = PHI(x)
        except DomainError as exc:
            return SystemResult("Метод простой итерации", False, x, history,
                                "Итерации вышли из области определения Phi: %s" % exc,
                                q, rule)
        if any((not math.isfinite(v)) or abs(v) > _DIVERGENCE_LIMIT for v in x_new):
            return SystemResult("Метод простой итерации", False, x_new, history,
                                "Процесс расходится", q, rule)

        dx = norm_inf(vec_sub(x_new, x))
        est = factor * dx
        history.append({"k": k, "x": list(x_new), "dx": dx, "est": est,
                        "residual": (norm_inf(F(x_new)) if F is not None else None)})
        x = x_new
        if est <= eps:
            return SystemResult("Метод простой итерации", True, x, history,
                                "Достигнута заданная точность", q, rule)

    return SystemResult("Метод простой итерации", False, x, history,
                        "Исчерпан лимит в %d итераций" % max_iter, q, rule)


def system_newton(F, x0, eps, J=None, max_iter=500):
    """Метод Ньютона для системы.

    На каждом шаге решается линейная система J(x^k) * dx = -F(x^k)
    (метод Гаусса с выбором главного элемента, собственная реализация),
    затем x^{k+1} = x^k + dx.
    """
    jacobian = J if J is not None else (lambda p: numeric_jacobian(F, p))
    rule = "||x^k - x^(k-1)||_inf <= eps  и  ||F(x^k)||_inf <= eps"

    x = [float(v) for v in x0]
    history = [{"k": 0, "x": list(x), "dx": None, "est": None,
                "residual": norm_inf(F(x))}]

    for k in range(1, max_iter + 1):
        try:
            Fx = F(x)
            Jx = jacobian(x)
        except DomainError as exc:
            return SystemResult("Метод Ньютона", False, x, history,
                                "Вышли из области определения: %s" % exc, None, rule)
        try:
            delta = solve_gauss(Jx, vec_scale(Fx, -1.0))
        except SingularMatrixError as exc:
            return SystemResult("Метод Ньютона", False, x, history,
                                "Якобиан вырожден: %s" % exc, None, rule)

        x_new = [x[i] + delta[i] for i in range(len(x))]
        if any((not math.isfinite(v)) or abs(v) > _DIVERGENCE_LIMIT for v in x_new):
            return SystemResult("Метод Ньютона", False, x_new, history,
                                "Процесс расходится", None, rule)

        dx = norm_inf(delta)
        try:
            residual = norm_inf(F(x_new))
        except DomainError as exc:
            return SystemResult("Метод Ньютона", False, x_new, history,
                                "Вышли из области определения: %s" % exc, None, rule)
        history.append({"k": k, "x": list(x_new), "dx": dx, "est": dx,
                        "residual": residual})
        x = x_new
        if dx <= eps and residual <= max(eps, 1e-14):
            return SystemResult("Метод Ньютона", True, x, history,
                                "Достигнута заданная точность", None, rule)

    return SystemResult("Метод Ньютона", False, x, history,
                        "Исчерпан лимит в %d итераций" % max_iter, None, rule)


def residual_norm(F, x):
    return norm_inf(F(x))


# используется только для печати проверки J*dx = -F
def apply_matrix(A, v):
    return mat_vec(A, v)
