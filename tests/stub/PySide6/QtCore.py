from PySide6 import _Perm


def __getattr__(name):
    return _Perm
