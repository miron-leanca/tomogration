"""Permissive fake PySide6 so tomogration_app imports on a box with no Qt.
Every attribute resolves to _Perm, which is subclassable (class X(QObject)),
callable (Signal(str, str)), attribute-permissive (Qt.AlignCenter,
QFileDialog.Option.DontUseNativeDialog), and OR-able (flag | flag)."""


class _Meta(type):
    def __getattr__(cls, name):
        return _Perm()


class _Perm(metaclass=_Meta):
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, n):
        return _Perm()

    def __call__(self, *a, **k):
        return _Perm()

    def __or__(self, o):
        return self

    def __ror__(self, o):
        return self

    # Qt returns INTEGERS from count()/currentRow()/etc, and code does
    # range(w.topLevelItemCount()). Without these, any such loop dies with
    # "'_Perm' object cannot be interpreted as an integer" and the dialog
    # cannot be constructed in a test at all. Deliberately NOT __len__ or
    # __bool__: those would flip _Perm to falsy and change how every existing
    # `if widget:` in the app behaves under the stub.
    def __index__(self):
        return 0

    def __int__(self):
        return 0

    def __eq__(self, o):
        return False

    def __hash__(self):
        return id(self)


def __getattr__(name):
    return _Perm
