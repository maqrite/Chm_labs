"""Безопасное вычисление математических выражений, заданных строками.

Все исходные данные (уравнения, производные, итерационные функции) задаются
в файлах конфигурации в виде строк, поэтому в коде нет ни одной "зашитой"
формулы. Здесь строка превращается в обычную питоновскую функцию.
"""

import math

__all__ = ["DomainError", "ExpressionError", "make_function", "ALLOWED_NAMES"]


class ExpressionError(Exception):
    """Ошибка разбора выражения (неизвестное имя, синтаксис и т.п.)."""


class DomainError(Exception):
    """Выражение не определено в точке (корень из отрицательного, log(0), ...)."""


def _ctg(t):
    s = math.sin(t)
    if s == 0.0:
        raise DomainError("ctg: sin(x) = 0")
    return math.cos(t) / s


def _base_namespace():
    """Функции и константы, разрешённые в выражениях."""
    return {
        # тригонометрия (в том числе привычные русские обозначения)
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "tg": math.tan,
        "ctg": _ctg, "cot": _ctg,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "arctg": math.atan, "arcsin": math.asin, "arccos": math.acos,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
        # экспонента и логарифмы
        "exp": math.exp, "log": math.log, "ln": math.log,
        "lg": math.log10, "log10": math.log10, "log2": math.log2,
        # прочее
        "sqrt": math.sqrt, "abs": abs, "fabs": math.fabs, "pow": math.pow,
        "sign": lambda t: (t > 0) - (t < 0),
        "pi": math.pi, "e": math.e,
    }


ALLOWED_NAMES = frozenset(_base_namespace())


def make_function(expression, variables, constants=None):
    """Скомпилировать строку в функцию f(v1, v2, ...).

    expression -- строка, например "2**x + x**2 - 2";
    variables  -- список имён аргументов, например ["x"] или ["x1", "x2"];
    constants  -- словарь параметров задачи, например {"a": 2}.

    Возвращает callable, бросающий DomainError, если в точке выражение
    не определено (это штатная ситуация при расходящихся итерациях).
    """
    constants = dict(constants or {})
    text = str(expression)

    try:
        code = compile(text, "<expr>", "eval")
    except SyntaxError as exc:
        raise ExpressionError("Синтаксическая ошибка в выражении %r: %s" % (text, exc))

    known = ALLOWED_NAMES | set(variables) | set(constants)
    unknown = sorted(set(code.co_names) - known)
    if unknown:
        raise ExpressionError(
            "В выражении %r встречаются неизвестные имена: %s" % (text, ", ".join(unknown))
        )

    globals_ = {"__builtins__": {}}
    base = _base_namespace()
    base.update(constants)

    def evaluate(*args):
        if len(args) != len(variables):
            raise ExpressionError(
                "Выражение %r ожидает %d аргумент(ов), получено %d"
                % (text, len(variables), len(args))
            )
        local = dict(base)
        local.update(zip(variables, args))
        try:
            value = eval(code, globals_, local)          # noqa: S307 - namespace закрыт
        except DomainError:
            raise
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise DomainError("%s в точке %s: %s" % (text, args, exc))
        return float(value)

    evaluate.expression = text
    evaluate.variables = list(variables)
    return evaluate


def make_vector_function(expressions, variables, constants=None):
    """Список строк -> функция, возвращающая список значений (вектор-функция F)."""
    funcs = [make_function(s, variables, constants) for s in expressions]

    def evaluate(args):
        return [f(*args) for f in funcs]

    evaluate.expressions = [f.expression for f in funcs]
    return evaluate


def make_matrix_function(expression_rows, variables, constants=None):
    """Матрица строк -> функция, возвращающая матрицу значений (якобиан J)."""
    rows = [[make_function(s, variables, constants) for s in row] for row in expression_rows]

    def evaluate(args):
        return [[f(*args) for f in row] for row in rows]

    evaluate.expressions = [[f.expression for f in row] for row in rows]
    return evaluate
