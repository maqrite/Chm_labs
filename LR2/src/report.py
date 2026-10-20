"""Формирование текстового отчёта: печать в консоль + сохранение в файл."""

import io


class Reporter(object):
    def __init__(self):
        self._buffer = io.StringIO()

    def write(self, text=""):
        print(text)
        self._buffer.write(text + "\n")

    def title(self, text, char="="):
        line = char * max(len(text), 78)
        self.write("")
        self.write(line)
        self.write(text)
        self.write(line)

    def section(self, text):
        self.write("")
        self.write(text)
        self.write("-" * max(len(text), 60))

    def table(self, headers, rows, aligns=None):
        rows = [[("" if c is None else str(c)) for c in row] for row in rows]
        headers = [str(h) for h in headers]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        aligns = aligns or (["<"] + [">"] * (len(headers) - 1))

        def fmt(cells, force=None):
            parts = []
            for i, cell in enumerate(cells):
                a = force or aligns[i]
                parts.append("%*s" % (widths[i], cell) if a == ">" else
                             "%-*s" % (widths[i], cell))
            return "| " + " | ".join(parts) + " |"

        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        self.write(sep)
        self.write(fmt(headers, "<"))
        self.write(sep)
        for row in rows:
            self.write(fmt(row))
        self.write(sep)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self._buffer.getvalue())
        return path


def fnum(value, digits=12):
    """Аккуратный вывод вещественного числа."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return "%.*f" % (digits, value)


def esci(value, digits=3):
    if value is None:
        return "-"
    if value == 0.0:
        return "0"
    return "%.*e" % (digits, value)


def iters(n):
    """Согласование слова 'итерация' с числом."""
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        word = "итерация"
    elif 2 <= n10 <= 4 and not (12 <= n100 <= 14):
        word = "итерации"
    else:
        word = "итераций"
    return "%d %s" % (n, word)
