"""Графика: отделение корня "графически" и график зависимости погрешности
от номера итерации.

Все графики строятся средствами matplotlib и сохраняются в png-файлы.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expr import DomainError


def _sample(f, a, b, n):
    xs, ys = [], []
    for i in range(n + 1):
        x = a + (b - a) * i / n
        try:
            y = f(x)
        except DomainError:
            xs.append(x); ys.append(float("nan")); continue
        xs.append(x)
        ys.append(y if math.isfinite(y) else float("nan"))
    return xs, ys


def plot_scalar(f, a, b, path, roots=(), title="", ylabel="f(x)"):
    xs, ys = _sample(f, a, b, 2000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, linewidth=1.8, label=ylabel)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    for r in roots:
        ax.plot([r], [0.0], "ro", markersize=7)
        ax.annotate("x* = %.6f" % r, (r, 0.0), textcoords="offset points",
                    xytext=(8, 12), color="red")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_two_curves(f_left, f_right, a, b, path, title="", labels=("y1", "y2")):
    """Графическое отделение корня в виде y1(x) = y2(x)."""
    xs1, ys1 = _sample(f_left, a, b, 2000)
    xs2, ys2 = _sample(f_right, a, b, 2000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs1, ys1, linewidth=1.8, label=labels[0])
    ax.plot(xs2, ys2, linewidth=1.8, label=labels[1])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_convergence(series, path, title="", ylabel="|x_k - x*|"):
    """series -- список (подпись, список погрешностей по итерациям)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for label, errors in series:
        ks = [k for k, e in enumerate(errors) if e is not None and e > 0.0]
        es = [errors[k] for k in ks]
        if not ks:
            continue
        ax.semilogy(ks, es, marker="o", markersize=4, linewidth=1.5, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_xlabel("номер итерации k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_system_curves(F, region, path, root=None, title="", labels=("f1 = 0", "f2 = 0"),
                       grid_size=400):
    """Кривые f1(x1,x2) = 0 и f2(x1,x2) = 0 на плоскости (только для n = 2)."""
    if len(region) != 2:
        return None
    (ax0, bx0), (ay0, by0) = region
    xs = [ax0 + (bx0 - ax0) * i / (grid_size - 1) for i in range(grid_size)]
    ys = [ay0 + (by0 - ay0) * i / (grid_size - 1) for i in range(grid_size)]
    Z1 = [[float("nan")] * grid_size for _ in range(grid_size)]
    Z2 = [[float("nan")] * grid_size for _ in range(grid_size)]
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            try:
                v = F([x, y])
            except DomainError:
                continue
            if math.isfinite(v[0]):
                Z1[i][j] = v[0]
            if math.isfinite(v[1]):
                Z2[i][j] = v[1]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    c1 = ax.contour(xs, ys, Z1, levels=[0.0], colors="tab:blue", linewidths=1.8)
    c2 = ax.contour(xs, ys, Z2, levels=[0.0], colors="tab:red", linewidths=1.8)
    for c, lab in ((c1, labels[0]), (c2, labels[1])):
        try:
            c.collections[0].set_label(lab)
        except (AttributeError, IndexError):
            pass
    if root is not None:
        ax.plot([root[0]], [root[1]], "ko", markersize=7)
        ax.annotate("(%.6f, %.6f)" % (root[0], root[1]), (root[0], root[1]),
                    textcoords="offset points", xytext=(8, 10))
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    handles = [plt.Line2D([], [], color="tab:blue", label=labels[0]),
               plt.Line2D([], [], color="tab:red", label=labels[1])]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
