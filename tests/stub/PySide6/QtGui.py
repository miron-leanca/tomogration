from PySide6 import _Perm


class QFontMetrics:
    """A REAL QFontMetrics, because the canvas measures with it.

    Qt returns integers from horizontalAdvance()/lineSpacing(), and the note
    painter divides and compares them. Left as _Perm, `_Perm // int` and
    `_Perm > int` raise TypeError and the only thing a test can conclude is
    that the stub is not Qt. A fixed-width approximation is enough to exercise
    the wrapping arithmetic for real: what the tests check is that text fits
    the box, not that it matches any particular font.
    """
    CHAR_W = 6
    LINE_H = 14

    def __init__(self, *a, **k):
        pass

    def horizontalAdvance(self, text, *a, **k):
        return len(str(text)) * self.CHAR_W

    def lineSpacing(self):
        return self.LINE_H

    def height(self):
        return self.LINE_H

    def elidedText(self, text, mode=None, width=0, flags=0):
        text = str(text)
        try:
            width = int(width)
        except (TypeError, ValueError):
            return text
        if self.horizontalAdvance(text) <= width:
            return text
        keep = max(0, width // self.CHAR_W - 1)
        return text[:keep] + "…"

    def boundingRect(self, *a, **k):
        return _Perm()


def __getattr__(name):
    return _Perm
