"""Задание 2.1: решение нелинейного уравнения f(x) = 0.

Реализованы метод простой итерации и метод Ньютона (касательных).
Вспомогательно: отделение корней перебором, метод половинного деления
(используется только для получения эталонного корня, с которым сравнивается
погрешность на каждой итерации).
"""

import math

from expr import DomainError

__all__ = [
    "IterationResult", "numeric_derivative", "separate_roots", "bisection",
    "estimate_q", "simple_iteration", "newton", "convergence_order",
]

_DIVERGENCE_LIMIT = 1e12


class IterationResult(object):
    """Результат работы итерационного метода."""

    def __init__(self, method, converged, root, history, message="", q=None,
                 stop_rule=""):
        self.method = method
        self.converged = converged
        self.root = root
        self.history = history          # список словарей по итерациям
        self.message = message
        self.q = q
        self.stop_rule = stop_rule

    @property
    def iterations(self):
        # нулевая строка истории -- начальное приближение, итерацией не считается
        return max(len(self.history) - 1, 0)


def numeric_derivative(f, x, h=1e-6):
    """Центральная разностная производная (на случай, если в файле данных
    аналитическая производная не задана)."""
    step = h * max(1.0, abs(x))
    return (f(x + step) - f(x - step)) / (2.0 * step)


def separate_roots(f, a, b, steps=1000):
    """Отделение корней: ищем отрезки, на концах которых f меняет знак.

    Это численный аналог графического отделения корня: тот же перебор,
    который глаз делает по картинке.
    """
    intervals = []
    prev_x = None
    prev_y = None
    for i in range(steps + 1):
        x = a + (b - a) * i / steps
        try:
            y = f(x)
        except DomainError:
            prev_x, prev_y = None, None
            continue
        if not math.isfinite(y):
            prev_x, prev_y = None, None
            continue
        if prev_y is not None:
            if y == 0.0:
                intervals.append((x, x))
            elif prev_y * y < 0.0:
                intervals.append((prev_x, x))
        prev_x, prev_y = x, y
    return intervals


def bisection(f, a, b, eps=1e-15, max_iter=200):
    """Метод половинного деления -- источник эталонного значения корня."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("На концах [%g, %g] функция не меняет знак" % (a, b))
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if fc == 0.0 or (b - a) / 2.0 <= eps:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def estimate_q(phi, a, b, samples=2001):
    """Оценка q = max |phi'(x)| на отрезке [a, b] (достаточное условие сходимости
    метода простой итерации: q < 1)."""
    best = 0.0
    for i in range(samples):
        x = a + (b - a) * i / (samples - 1)
        try:
            d = abs(numeric_derivative(phi, x))
        except DomainError:
            continue
        if math.isfinite(d):
            best = max(best, d)
    return best


def maps_into_itself(phi, a, b, samples=501):
    """Проверка условия phi([a,b]) подмножество [a,b]."""
    lo = math.inf
    hi = -math.inf
    for i in range(samples):
        x = a + (b - a) * i / (samples - 1)
        try:
            y = phi(x)
        except DomainError:
            return False, None, None
        lo = min(lo, y)
        hi = max(hi, y)
    return (a <= lo and hi <= b), lo, hi


def simple_iteration(phi, x0, eps, q=None, max_iter=500, f=None):
    """Метод простой итерации: x_{k+1} = phi(x_k).

    Критерий остановки: априорно-апостериорная оценка
        |x_k - x*| <= q/(1-q) * |x_k - x_{k-1}| <= eps,
    если известна константа сжатия q < 1; иначе -- просто |x_k - x_{k-1}| <= eps.
    """
    if q is not None and 0.0 < q < 1.0:
        factor = q / (1.0 - q)
        rule = "q/(1-q)*|x_k - x_(k-1)| <= eps,  q = %.6f" % q
    else:
        factor = 1.0
        rule = "|x_k - x_(k-1)| <= eps"

    x = float(x0)
    history = [{"k": 0, "x": x, "dx": None, "est": None,
                "fx": (f(x) if f is not None else None)}]

    for k in range(1, max_iter + 1):
        try:
            x_new = phi(x)
        except DomainError as exc:
            return IterationResult("Метод простой итерации", False, x, history,
                                   "Итерации вышли из области определения phi: %s" % exc,
                                   q, rule)
        if not math.isfinite(x_new) or abs(x_new) > _DIVERGENCE_LIMIT:
            history.append({"k": k, "x": x_new, "dx": None, "est": None, "fx": None})
            return IterationResult("Метод простой итерации", False, x_new, history,
                                   "Процесс расходится", q, rule)

        dx = abs(x_new - x)
        est = factor * dx
        history.append({"k": k, "x": x_new, "dx": dx, "est": est,
                        "fx": (f(x_new) if f is not None else None)})
        x = x_new
        if est <= eps:
            return IterationResult("Метод простой итерации", True, x, history,
                                   "Достигнута заданная точность", q, rule)

    return IterationResult("Метод простой итерации", False, x, history,
                           "Исчерпан лимит в %d итераций" % max_iter, q, rule)


def newton(f, x0, eps, df=None, max_iter=500, min_derivative=1e-14):
    """Метод Ньютона: x_{k+1} = x_k - f(x_k)/f'(x_k).

    Если производная df не задана в файле данных, используется центральная
    разностная формула.
    """
    derivative = df if df is not None else (lambda t: numeric_derivative(f, t))
    rule = "|x_k - x_(k-1)| <= eps  и  |f(x_k)| <= eps"

    x = float(x0)
    history = [{"k": 0, "x": x, "dx": None, "est": None, "fx": f(x)}]

    for k in range(1, max_iter + 1):
        try:
            fx = f(x)
            dfx = derivative(x)
        except DomainError as exc:
            return IterationResult("Метод Ньютона", False, x, history,
                                   "Вышли из области определения: %s" % exc, None, rule)
        if abs(dfx) < min_derivative:
            return IterationResult("Метод Ньютона", False, x, history,
                                   "f'(x) обратилась в ноль -- касательная параллельна оси Ox",
                                   None, rule)
        x_new = x - fx / dfx
        if not math.isfinite(x_new) or abs(x_new) > _DIVERGENCE_LIMIT:
            return IterationResult("Метод Ньютона", False, x_new, history,
                                   "Процесс расходится", None, rule)
        dx = abs(x_new - x)
        try:
            fx_new = f(x_new)
        except DomainError as exc:
            return IterationResult("Метод Ньютона", False, x_new, history,
                                   "Вышли из области определения: %s" % exc, None, rule)
        history.append({"k": k, "x": x_new, "dx": dx, "est": dx, "fx": fx_new})
        x = x_new
        if dx <= eps and abs(fx_new) <= max(eps, 1e-15):
            return IterationResult("Метод Ньютона", True, x, history,
                                   "Достигнута заданная точность", None, rule)

    return IterationResult("Метод Ньютона", False, x, history,
                           "Исчерпан лимит в %d итераций" % max_iter, None, rule)


def convergence_order(errors):
    """Эмпирический порядок сходимости p по трём подряд идущим погрешностям:
        p ~ ln(e_{k+1}/e_k) / ln(e_k/e_{k-1}).
    """
    orders = []
    for i in range(1, len(errors) - 1):
        e0, e1, e2 = errors[i - 1], errors[i], errors[i + 1]
        if min(e0, e1, e2) <= 0.0:
            continue
        try:
            top = math.log(e2 / e1)
            bottom = math.log(e1 / e0)
        except ValueError:
            continue
        if abs(bottom) < 1e-14:
            continue
        orders.append(top / bottom)
    return orders


def clean_errors(errors, floor=1e-14):
    """Отбросить хвост, в котором погрешность легла на уровень машинной точности
    (дальше о скорости сходимости говорить бессмысленно)."""
    cleaned = []
    for e in errors:
        if e is None:
            continue
        if e <= floor:
            cleaned.append(e)
            break
        cleaned.append(e)
    while len(cleaned) > 1 and cleaned[-1] <= floor:
        cleaned.pop()
    return cleaned


def mean_ratio(errors):
    """Средний множитель убывания погрешности e_(k+1)/e_k (геометрическое среднее).
    Для линейно сходящегося метода это и есть оценка q."""
    seq = [e for e in clean_errors(errors) if e > 0.0]
    if len(seq) < 2:
        return None
    return (seq[-1] / seq[0]) ** (1.0 / (len(seq) - 1))


def order_least_squares(errors):
    """Порядок сходимости p как наклон прямой ln e_(k+1) = p*ln e_k + C
    (метод наименьших квадратов -- устойчивее формулы по трём точкам)."""
    seq = [e for e in clean_errors(errors) if e > 0.0]
    if len(seq) < 3:
        return None
    xs = [math.log(e) for e in seq[:-1]]
    ys = [math.log(e) for e in seq[1:]]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    if abs(den) < 1e-14:
        return None
    return num / den
