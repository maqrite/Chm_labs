"""Лабораторная работа № 2. Численные методы.

2.1. Методы простой итерации и Ньютона решения нелинейного уравнения.
2.2. Методы простой итерации и Ньютона решения системы нелинейных уравнений.

Вариант 7:
    2.1:  2^x + x^2 - 2 = 0
    2.2:  x1^2 + x2^2 - a^2 = 0,  x1 - e^(x2) + a = 0,   a = 2

Все исходные данные (уравнения, производные, итерационные функции, точность,
области) читаются из json-файлов каталога input/ -- в коде ничего не зашито.

Запуск:
    python3 main.py
    python3 main.py --task 1 --input input/task_2_1.json --eps 1e-12
"""

import argparse
import json
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import plotting                                             # noqa: E402
from expr import (DomainError, make_function, make_matrix_function,             # noqa: E402
                  make_vector_function)
from linalg import norm_inf, vec_sub                        # noqa: E402
from report import Reporter, esci, fnum, iters                     # noqa: E402
import scalar_methods as sm                                 # noqa: E402
import system_methods as sy                                 # noqa: E402

HEAD_ROWS, TAIL_ROWS, FULL_LIMIT = 12, 8, 26


# --------------------------------------------------------------------------
# служебное
# --------------------------------------------------------------------------
def rel(path):
    """Короткий путь для печати (если файл вне каталога проекта -- печатаем как есть)."""
    short = os.path.relpath(path, BASE_DIR)
    return path if short.startswith("..") else short


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def thin(rows):
    """Свернуть длинную таблицу: начало + '...' + конец."""
    if len(rows) <= FULL_LIMIT:
        return rows
    gap = [["..."] * len(rows[0])]
    return rows[:HEAD_ROWS] + gap + rows[-TAIL_ROWS:]


def second_derivative(f, x, h=1e-4):
    step = h * max(1.0, abs(x))
    return (f(x + step) - 2.0 * f(x) + f(x - step)) / (step * step)


def si_res_iters(si_results):
    """Наименьшее число итераций среди сошедшихся вариантов phi."""
    good = [r.iterations for _, r in si_results if r.converged]
    return min(good) if good else 0


def rate_summary(errors):
    """(средний множитель убывания, порядок сходимости) по истории погрешностей."""
    errors = [e for e in errors if e is not None]
    return sm.mean_ratio(errors), sm.order_least_squares(errors)


def fmt_rate(errors, theory):
    rho, p = rate_summary(errors)
    return "множитель e_(k+1)/e_k ~ %s,  порядок p ~ %s  (теоретически p = %d)" % (
        ("%.4f" % rho) if rho else "-", ("%.2f" % p) if p else "-", theory)


# --------------------------------------------------------------------------
# ЗАДАНИЕ 2.1
# --------------------------------------------------------------------------
def run_task_2_1(cfg, rep, out_dir):
    var = cfg.get("variable", "x")
    consts = cfg.get("constants", {})
    f = make_function(cfg["f"], [var], consts)
    df = make_function(cfg["df"], [var], consts) if cfg.get("df") else None
    eps = float(cfg["epsilon"])
    max_iter = int(cfg.get("max_iterations", 500))

    rep.title(cfg.get("title", "Задание 2.1"))
    rep.section("1. Исходные данные")
    rep.write("Уравнение:              f(%s) = %s = 0" % (var, cfg["f"]))
    rep.write("Производная:            f'(%s) = %s" % (var, cfg.get("df", "(численная)")))
    if consts:
        rep.write("Параметры:              %s" % consts)
    rep.write("Заданная точность:      eps = %s" % esci(eps))
    rep.write("Лимит итераций:         %d" % max_iter)

    # ---------- 2. графическое отделение корня ----------
    a_s, b_s = cfg.get("search_interval", [-5.0, 5.0])
    steps = int(cfg.get("search_steps", 600))

    rep.section("2. Графическое отделение корня")
    graphic = cfg.get("graphic")
    if graphic:
        left = make_function(graphic["left"], [var], consts)
        right = make_function(graphic["right"], [var], consts)
        rep.write("Уравнение приведено к виду  %s = %s,  корень -- точка пересечения"
                  % (graphic["left"], graphic["right"]))
        rep.write("(на графике: %s и %s)" % (graphic["labels"][0], graphic["labels"][1]))
        path = plotting.plot_two_curves(left, right, a_s, b_s,
                                        os.path.join(out_dir, "task_2_1_graphic.png"),
                                        title="Графическое отделение корня",
                                        labels=tuple(graphic["labels"]))
        if path:
            rep.write("График сохранён: %s" % rel(path))

    intervals = sm.separate_roots(f, a_s, b_s, steps)
    if not intervals:
        rep.write("Корней на [%g, %g] не найдено." % (a_s, b_s))
        return None
    rep.write("")
    rep.write("Отрезки перемены знака f(x) на [%g, %g]:" % (a_s, b_s))
    rep.table(["№", "a", "b", "f(a)", "f(b)"],
              [[i + 1, fnum(p, 6), fnum(q, 6), esci(f(p)), esci(f(q))]
               for i, (p, q) in enumerate(intervals)])

    positive = [iv for iv in intervals if iv[1] > 0.0]
    if cfg.get("positive_root_only", True) and positive:
        a, b = positive[0]
    else:
        a, b = intervals[0]
    rep.write("Для дальнейшей работы взят положительный корень с отрезка "
              "[%.6f, %.6f]." % (a, b))

    # ---------- 3. эталонный корень ----------
    rep.section("3. Эталонное значение корня")
    x_ref = sm.bisection(f, a, b, eps=1e-15)
    rep.write("Метод половинного деления (собственная реализация, точность 1e-15):")
    rep.write("    x* = %s,   f(x*) = %s" % (fnum(x_ref, 15), esci(f(x_ref))))
    path = plotting.plot_scalar(f, a_s, b_s, os.path.join(out_dir, "task_2_1_function.png"),
                                roots=[x_ref], title="f(x) = %s" % cfg["f"])
    if path:
        rep.write("График сохранён: %s" % rel(path))

    # ---------- 4. метод простой итерации ----------
    rep.section("4. Метод простой итерации")
    rep.write("Уравнение приводится к виду x = phi(x); итерации x_(k+1) = phi(x_k).")
    rep.write("Достаточное условие сходимости: phi отображает [a,b] в себя и")
    rep.write("q = max|phi'(x)| < 1 на [a,b]; тогда |x_k - x*| <= q/(1-q)*|x_k - x_(k-1)|.")

    si_results = []
    for variant in cfg.get("phi_variants", []):
        interval = variant.get("interval", [a, b])
        phi = make_function(variant["expr"], [var], consts)
        q = sm.estimate_q(phi, interval[0], interval[1])
        inside, lo, hi = sm.maps_into_itself(phi, interval[0], interval[1])
        x0 = variant.get("x0", 0.5 * (interval[0] + interval[1]))

        rep.write("")
        rep.write(">>> %s" % variant["name"])
        if variant.get("comment"):
            rep.write("    %s" % variant["comment"])
        rep.write("    отрезок [a, b]      = [%g, %g]" % (interval[0], interval[1]))
        rep.write("    q = max|phi'(x)|    = %.6f   -> %s"
                  % (q, "условие сжатия выполнено" if q < 1 else
                     "условие сжатия НАРУШЕНО (q >= 1)"))
        if lo is None:
            rep.write("    phi([a,b])          = выходит за область определения")
        else:
            rep.write("    phi([a,b])          = [%.6f, %.6f] -> %s"
                      % (lo, hi, "вложен в [a,b]" if inside else "НЕ вложен в [a,b]"))
        rep.write("    начальное x0        = %s" % fnum(x0, 6))

        res = sm.simple_iteration(phi, x0, eps, q=q, max_iter=max_iter, f=f)
        _print_scalar_history(rep, res, x_ref)
        si_results.append((variant["name"], res))

    good = [(n, r) for n, r in si_results if r.converged]
    si_name, si_res = good[0] if good else si_results[0]

    # ---------- 5. метод Ньютона ----------
    rep.section("5. Метод Ньютона (метод касательных)")
    rep.write("Итерации: x_(k+1) = x_k - f(x_k)/f'(x_k).")
    rep.write("Начальное приближение выбирается из условия f(x0)*f''(x0) > 0")
    rep.write("(тогда сходимость монотонная и гарантированная).")
    x0_n = None
    for candidate in (a, b, 0.5 * (a + b)):
        if f(candidate) * second_derivative(f, candidate) > 0.0:
            x0_n = candidate
            break
    if x0_n is None:
        x0_n = 0.5 * (a + b)
        rep.write("Условие ни на одном конце не выполнено, взята середина отрезка.")
    rep.write("    f(%.6f) = %s,  f''(%.6f) = %s  ->  x0 = %.6f"
              % (x0_n, esci(f(x0_n)), x0_n, esci(second_derivative(f, x0_n)), x0_n))
    nt_res = sm.newton(f, x0_n, eps, df=df, max_iter=max_iter)
    _print_scalar_history(rep, nt_res, x_ref)

    # ---------- 6. сравнение ----------
    rep.section("6. Сравнение методов: погрешность в зависимости от номера итерации")
    columns = [(n, [abs(row["x"] - x_ref) for row in r.history])
               for n, r in si_results if r.converged]
    nt_err = [abs(row["x"] - x_ref) for row in nt_res.history]
    columns.append(("метод Ньютона", nt_err))
    si_err = dict(columns)[si_name]

    depth = max(len(col) for _, col in columns)
    rows = []
    for k in range(depth):
        rows.append([k] + [esci(col[k]) if k < len(col) else "" for _, col in columns])
    rep.table(["k"] + ["|x_k - x*|: " + n for n, _ in columns], thin(rows))

    rep.write("")
    rep.write("Скорость сходимости (по убыванию погрешности):")
    for name, col in columns:
        rep.write("    %-42s %s" % (name + ":",
                  fmt_rate(col, 2 if name == "метод Ньютона" else 1)))

    rep.write("")
    summary = [[n, fnum(r.history[0]["x"], 6), r.iterations, fnum(r.root, 12),
                esci(abs(r.root - x_ref)), esci(abs(f(r.root)))]
               for n, r in si_results if r.converged]
    summary.append(["метод Ньютона", fnum(x0_n, 6), nt_res.iterations,
                    fnum(nt_res.root, 12), esci(abs(nt_res.root - x_ref)),
                    esci(abs(f(nt_res.root)))])
    rep.table(["Метод", "x0", "Итераций", "Найденный корень", "|x - x*|", "|f(x)|"],
              summary)
    faster = "Ньютона" if nt_res.iterations < si_res_iters(si_results) else "простой итерации"
    rep.write("Вывод: быстрее сходится метод %s -- %s против %s у метода простой "
              "итерации (квадратичная сходимость против линейной)."
              % (faster, iters(nt_res.iterations), iters(si_res_iters(si_results))))

    path = plotting.plot_convergence(
        columns, os.path.join(out_dir, "task_2_1_convergence.png"),
        title="2.1. Погрешность в зависимости от номера итерации")
    if path:
        rep.write("График сохранён: %s" % rel(path))

    # ---------- 7. число итераций против eps ----------
    series = cfg.get("epsilon_series")
    if series:
        rep.section("7. Число итераций в зависимости от заданной точности eps")
        rep.write("Метод простой итерации взят с %s; метод Ньютона -- с x0 = %s."
                  % (si_name, fnum(x0_n, 6)))
        phi_cfg = None
        for variant in cfg.get("phi_variants", []):
            if variant["name"] == si_name:
                phi_cfg = variant
        phi = make_function(phi_cfg["expr"], [var], consts)
        iv = phi_cfg.get("interval", [a, b])
        q = sm.estimate_q(phi, iv[0], iv[1])
        x0 = phi_cfg.get("x0", 0.5 * (iv[0] + iv[1]))
        rows = []
        for e in series:
            r1 = sm.simple_iteration(phi, x0, e, q=q, max_iter=max_iter, f=f)
            r2 = sm.newton(f, x0_n, e, df=df, max_iter=max_iter)
            rows.append([esci(e, 0), r1.iterations, esci(abs(r1.root - x_ref)),
                         r2.iterations, esci(abs(r2.root - x_ref))])
        rep.table(["eps", "итераций (прост.)", "факт. погрешность",
                   "итераций (Ньютон)", "факт. погрешность"], rows)
    return x_ref


def _print_scalar_history(rep, res, x_ref):
    rows = []
    for row in res.history:
        rows.append([row["k"], fnum(row["x"], 12), esci(row["dx"]), esci(row["est"]),
                     esci(abs(row["x"] - x_ref)), esci(row["fx"])])
    rep.write("    Критерий остановки: %s" % res.stop_rule)
    rep.table(["k", "x_k", "|x_k - x_(k-1)|", "оценка погрешности",
               "|x_k - x*|", "f(x_k)"], thin(rows))
    status = "СОШЁЛСЯ" if res.converged else "НЕ СОШЁЛСЯ"
    rep.write("    Результат: %s за %s. %s" % (status, iters(res.iterations), res.message))
    if res.converged:
        rep.write("    Корень: x = %s" % fnum(res.root, 12))


# --------------------------------------------------------------------------
# ЗАДАНИЕ 2.2
# --------------------------------------------------------------------------
def run_task_2_2(cfg, rep, out_dir):
    variables = cfg["variables"]
    consts = cfg.get("constants", {})
    n = len(variables)
    F = make_vector_function(cfg["F"], variables, consts)
    J = make_matrix_function(cfg["J"], variables, consts) if cfg.get("J") else None
    PHI = make_vector_function(cfg["PHI"], variables, consts) if cfg.get("PHI") else None
    eps = float(cfg["epsilon"])
    max_iter = int(cfg.get("max_iterations", 500))

    rep.title(cfg.get("title", "Задание 2.2"))
    rep.section("1. Исходные данные")
    rep.write("Система из %d уравнений с неизвестными %s:" % (n, ", ".join(variables)))
    for i, e in enumerate(cfg["F"]):
        rep.write("    f%d = %s = 0" % (i + 1, e))
    if consts:
        rep.write("Параметры: %s" % ", ".join("%s = %g" % kv for kv in consts.items()))
    rep.write("Матрица Якоби J(x) = df_i/dx_j:")
    for row in cfg.get("J", []):
        rep.write("    [ %s ]" % " ; ".join(row))
    if PHI:
        rep.write("Итерационные функции x = Phi(x):")
        for i, e in enumerate(cfg["PHI"]):
            rep.write("    %s = %s" % (variables[i], e))
        if cfg.get("phi_comment"):
            rep.write("    (%s)" % cfg["phi_comment"])
    rep.write("Заданная точность: eps = %s" % esci(eps))

    # ---------- 2. графическое отделение решения ----------
    plot_region = cfg.get("plot_region", cfg["region"])
    rep.section("2. Графическое отделение решения")
    if n == 2:
        path = plotting.plot_system_curves(
            F, plot_region, os.path.join(out_dir, "task_2_2_graphic.png"),
            title="Графическое отделение решения системы",
            labels=("f1(x1,x2) = 0", "f2(x1,x2) = 0"))
        if path:
            rep.write("График сохранён: %s" % rel(path))
    else:
        rep.write("n = %d: графическое отделение возможно только для n = 2." % n)

    # ---------- 3. все решения и выбор положительного ----------
    rep.section("3. Поиск всех решений (метод Ньютона из узлов сетки)")
    found = _scan_solutions(F, J, plot_region, max_iter,
                            nodes=(13 if n == 2 else 7))
    if found:
        rep.table(["№"] + variables + ["||F(x)||_inf"],
                  [[i + 1] + [fnum(v, 10) for v in s] + [esci(norm_inf(F(s)))]
                   for i, s in enumerate(found)])
    else:
        rep.write("Сетка не дала ни одного решения.")
    positive = [s for s in found if all(v > 0.0 for v in s)]
    if cfg.get("positive_solution_only", True) and positive:
        rep.write("Условию положительности всех неизвестных удовлетворяет решение:")
        rep.write("    " + ",  ".join("%s = %s" % (variables[i], fnum(positive[0][i], 12))
                                      for i in range(n)))

    x0 = cfg.get("initial_guess") or (positive[0] if positive else found[0])
    region = cfg["region"]

    # ---------- 4. эталонное решение ----------
    rep.section("4. Эталонное решение")
    ref = sy.system_newton(F, x0, 1e-14, J=J, max_iter=200)
    x_ref = ref.root
    rep.write("Метод Ньютона с точностью 1e-14 (%s):" % iters(ref.iterations))
    for i in range(n):
        rep.write("    %s* = %s" % (variables[i], fnum(x_ref[i], 15)))
    rep.write("    невязка ||F(x*)||_inf = %s" % esci(norm_inf(F(x_ref))))
    _numpy_cross_check(rep, F, x_ref)

    # ---------- 5. метод простой итерации ----------
    si_res = None
    if PHI:
        rep.section("5. Метод простой итерации для системы")
        rep.write("Итерации: x^(k+1) = Phi(x^k).")
        rep.write("Достаточное условие: Phi отображает область G в себя и")
        rep.write("q = max||Phi'(x)||_inf < 1 на G.")
        rep.write("Область G: %s" % ",  ".join(
            "%s in [%g, %g]" % (variables[i], region[i][0], region[i][1]) for i in range(n)))
        q = sy.estimate_q_system(PHI, region)
        inside, lows, highs = sy.check_region(PHI, region)
        rep.write("    q = max||Phi'(x)||_inf = %.6f  ->  %s"
                  % (q, "условие сжатия выполнено" if q < 1 else "условие НАРУШЕНО"))
        if lows is not None:
            rep.write("    Phi(G) = %s  ->  %s" % (
                ",  ".join("[%.6f, %.6f]" % (lows[i], highs[i]) for i in range(n)),
                "вложено в G" if inside else "НЕ вложено в G"))
        rep.write("    начальное приближение x0 = (%s)"
                  % ", ".join(fnum(v, 6) for v in x0))
        si_res = sy.system_simple_iteration(PHI, x0, eps, q=q, max_iter=max_iter, F=F)
        _print_system_history(rep, si_res, x_ref, variables)

    # ---------- 6. метод Ньютона ----------
    rep.section("6. Метод Ньютона для системы")
    rep.write("На каждом шаге решается СЛАУ  J(x^k)*dx = -F(x^k)  методом Гаусса")
    rep.write("с выбором главного элемента (собственная реализация), затем x^(k+1) = x^k + dx.")
    rep.write("    начальное приближение x0 = (%s)" % ", ".join(fnum(v, 6) for v in x0))
    nt_res = sy.system_newton(F, x0, eps, J=J, max_iter=max_iter)
    _print_system_history(rep, nt_res, x_ref, variables)

    # ---------- 7. сравнение ----------
    rep.section("7. Зависимость погрешности от номера итерации")
    nt_err = [norm_inf(vec_sub(row["x"], x_ref)) for row in nt_res.history]
    si_err = ([norm_inf(vec_sub(row["x"], x_ref)) for row in si_res.history]
              if si_res else [])
    rows = []
    for k in range(max(len(si_err), len(nt_err))):
        rows.append([k,
                     esci(si_err[k]) if k < len(si_err) else "",
                     esci(nt_err[k]) if k < len(nt_err) else ""])
    rep.table(["k", "||x^k - x*|| простая итерация", "||x^k - x*|| Ньютон"], thin(rows))

    rep.write("")
    rep.write("Скорость сходимости (по убыванию погрешности):")
    if si_err:
        rep.write("    %-22s %s" % ("простая итерация:", fmt_rate(si_err, 1)))
    rep.write("    %-22s %s" % ("метод Ньютона:", fmt_rate(nt_err, 2)))

    rep.write("")
    table_rows = []
    if si_res:
        table_rows.append(["Простая итерация", si_res.iterations,
                           ", ".join(fnum(v, 10) for v in si_res.root),
                           esci(norm_inf(vec_sub(si_res.root, x_ref))),
                           esci(norm_inf(F(si_res.root)))])
    table_rows.append(["Метод Ньютона", nt_res.iterations,
                       ", ".join(fnum(v, 10) for v in nt_res.root),
                       esci(norm_inf(vec_sub(nt_res.root, x_ref))),
                       esci(norm_inf(F(nt_res.root)))])
    rep.table(["Метод", "Итераций", "Решение (" + ", ".join(variables) + ")",
               "||x - x*||", "||F(x)||"], table_rows)
    if si_res:
        faster = "Ньютона" if nt_res.iterations < si_res.iterations else "простой итерации"
        rep.write("Вывод: быстрее сходится метод %s -- %s против %s "
                  "(квадратичная сходимость против линейной)."
                  % (faster, iters(min(nt_res.iterations, si_res.iterations)),
                     iters(max(nt_res.iterations, si_res.iterations))))

    series = [("простая итерация", si_err)] if si_err else []
    series.append(("метод Ньютона", nt_err))
    path = plotting.plot_convergence(
        series, os.path.join(out_dir, "task_2_2_convergence.png"),
        title="2.2. Погрешность в зависимости от номера итерации",
        ylabel="||x^k - x*||_inf")
    if path:
        rep.write("График сохранён: %s" % rel(path))

    if n == 2:
        path = plotting.plot_system_curves(
            F, region, os.path.join(out_dir, "task_2_2_solution.png"), root=x_ref,
            title="Окрестность найденного решения",
            labels=("f1(x1,x2) = 0", "f2(x1,x2) = 0"))
        if path:
            rep.write("График сохранён: %s" % rel(path))

    # ---------- 8. число итераций против eps ----------
    eseries = cfg.get("epsilon_series")
    if eseries:
        rep.section("8. Число итераций в зависимости от заданной точности eps")
        q = sy.estimate_q_system(PHI, region) if PHI else None
        rows = []
        for e in eseries:
            r2 = sy.system_newton(F, x0, e, J=J, max_iter=max_iter)
            if PHI:
                r1 = sy.system_simple_iteration(PHI, x0, e, q=q, max_iter=max_iter, F=F)
                rows.append([esci(e, 0), r1.iterations,
                             esci(norm_inf(vec_sub(r1.root, x_ref))),
                             r2.iterations, esci(norm_inf(vec_sub(r2.root, x_ref)))])
            else:
                rows.append([esci(e, 0), "-", "-", r2.iterations,
                             esci(norm_inf(vec_sub(r2.root, x_ref)))])
        rep.table(["eps", "итераций (прост.)", "факт. погрешность",
                   "итераций (Ньютон)", "факт. погрешность"], rows)
    return x_ref


def _scan_solutions(F, J, region, max_iter, nodes=13, tol=1e-6):
    """Запускаем метод Ньютона из узлов сетки и собираем различные решения."""
    import itertools
    axes = []
    for (a, b) in region:
        axes.append([a + (b - a) * i / (nodes - 1) for i in range(nodes)])
    found = []
    for point in itertools.product(*axes):
        try:
            res = sy.system_newton(F, list(point), 1e-12, J=J, max_iter=60)
        except (DomainError, ValueError, ZeroDivisionError):
            continue
        if not res.converged:
            continue
        x = res.root
        if any(not math.isfinite(v) for v in x):
            continue
        if any(v < region[i][0] - 1.0 or v > region[i][1] + 1.0
               for i, v in enumerate(x)):
            continue
        if all(norm_inf(vec_sub(x, other)) > tol for other in found):
            found.append([round(v, 12) for v in x])
    found.sort()
    return found


def _numpy_cross_check(rep, F, x):
    """Контрольная проверка невязки средствами NumPy (только проверка результата)."""
    try:
        import numpy as np
    except ImportError:
        return
    residual = np.array(F(list(x)), dtype=float)
    rep.write("    Контроль NumPy: ||F(x*)||_2 = %s, ||F(x*)||_inf = %s"
              % (esci(float(np.linalg.norm(residual))),
                 esci(float(np.max(np.abs(residual))))))


def _print_system_history(rep, res, x_ref, variables):
    rows = []
    for row in res.history:
        rows.append([row["k"]] + [fnum(v, 12) for v in row["x"]] +
                    [esci(row["dx"]), esci(row["est"]),
                     esci(norm_inf(vec_sub(row["x"], x_ref))), esci(row["residual"])])
    rep.write("    Критерий остановки: %s" % res.stop_rule)
    rep.table(["k"] + variables + ["||dx||", "оценка погрешности",
                                   "||x^k - x*||", "||F(x^k)||"], thin(rows))
    status = "СОШЁЛСЯ" if res.converged else "НЕ СОШЁЛСЯ"
    rep.write("    Результат: %s за %s. %s" % (status, iters(res.iterations), res.message))
    if res.converged:
        rep.write("    Решение: " + ",  ".join(
            "%s = %s" % (variables[i], fnum(res.root[i], 12)) for i in range(len(variables))))


# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Лабораторная работа № 2: нелинейные уравнения и системы")
    parser.add_argument("--task", choices=["1", "2", "all"], default="all",
                        help="какое задание выполнить (по умолчанию оба)")
    parser.add_argument("--input-1", default=os.path.join(BASE_DIR, "input", "task_2_1.json"))
    parser.add_argument("--input-2", default=os.path.join(BASE_DIR, "input", "task_2_2.json"))
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "out"),
                        help="каталог для отчёта и графиков")
    parser.add_argument("--eps", type=float, default=None,
                        help="переопределить точность из файла данных")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rep = Reporter()
    rep.write("ЛАБОРАТОРНАЯ РАБОТА № 2. Решение нелинейных уравнений и систем.")
    rep.write("Вариант 7. Все исходные данные читаются из json-файлов каталога input/.")
    if args.task in ("1", "all"):
        cfg = load_config(args.input_1)
        if args.eps:
            cfg["epsilon"] = args.eps
        run_task_2_1(cfg, rep, args.out)

    if args.task in ("2", "all"):
        cfg = load_config(args.input_2)
        if args.eps:
            cfg["epsilon"] = args.eps
        run_task_2_2(cfg, rep, args.out)

    path = rep.save(os.path.join(args.out, "report.txt"))
    print("\nОтчёт сохранён: %s" % rel(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
