"""Минимальная линейная алгебра (собственная реализация, без NumPy).

Нужна методу Ньютона для систем: на каждой итерации решается СЛАУ
J(x^k) * dx = -F(x^k). Размерность системы нигде не фиксируется.
"""

__all__ = [
    "SingularMatrixError", "solve_gauss", "mat_vec", "vec_sub", "vec_add",
    "vec_scale", "norm_inf", "norm_1", "norm_2", "matrix_norm_inf", "zeros",
]


class SingularMatrixError(Exception):
    """Матрица вырождена (или численно вырождена) -- метод неприменим."""


def zeros(n):
    return [0.0] * n


def mat_vec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def vec_sub(u, v):
    return [ui - vi for ui, vi in zip(u, v)]


def vec_add(u, v):
    return [ui + vi for ui, vi in zip(u, v)]


def vec_scale(u, c):
    return [c * ui for ui in u]


def norm_inf(u):
    """Кубическая норма ||u||_inf = max |u_i|."""
    return max((abs(ui) for ui in u), default=0.0)


def norm_1(u):
    return sum(abs(ui) for ui in u)


def norm_2(u):
    return sum(ui * ui for ui in u) ** 0.5


def matrix_norm_inf(A):
    """Норма матрицы, согласованная с ||.||_inf: максимум суммы модулей по строке."""
    return max((sum(abs(v) for v in row) for row in A), default=0.0)


def solve_gauss(A, b, tol=1e-14):
    """Метод Гаусса с выбором главного элемента по столбцу.

    A -- квадратная матрица n x n (список списков), b -- вектор длины n.
    Исходные данные не портятся: работаем с копиями.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise SingularMatrixError("Матрица не квадратная")
    if len(b) != n:
        raise SingularMatrixError("Размер правой части не совпадает с матрицей")

    M = [list(row) + [b[i]] for i, row in enumerate(A)]

    # --- прямой ход ---
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < tol:
            raise SingularMatrixError(
                "Главный элемент на шаге %d практически нулевой -- матрица вырождена" % (k + 1)
            )
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        pivot = M[k][k]
        for i in range(k + 1, n):
            factor = M[i][k] / pivot
            if factor == 0.0:
                continue
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    # --- обратный ход ---
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x
