"""A stub mrcfile that enforces the REAL format's dtype rules.

The point is the rules, not the I/O. J44 split all 72 volumes and then died on
`m.set_data(int32_array)` because MRC has no int32 mode — a permissive stub
would have written it happily and the test would have passed while the cluster
run failed at the very last step. So this refuses exactly what mrcfile refuses.
"""
import pickle
from pathlib import Path

import numpy as np

# mode: 0 int8, 1 int16, 2 float32, 4 complex64, 6 uint16, 12 float16
_MODES = {np.dtype("int8"): 0, np.dtype("int16"): 1, np.dtype("float32"): 2,
          np.dtype("complex64"): 4, np.dtype("uint16"): 6,
          np.dtype("float16"): 12}


def mode_from_dtype(dtype):
    if np.dtype(dtype) not in _MODES:
        raise ValueError(f"dtype '{dtype}' cannot be converted "
                         f"to an MRC file mode")
    return _MODES[np.dtype(dtype)]


class _VoxelSize:
    def __init__(self, x=1.0):
        self.x = self.y = self.z = float(x)

    def __float__(self):
        return float(self.x)


class _File:
    def __init__(self, path, write=False):
        self.path, self._write = Path(path), write
        if write:
            self.data, self.voxel_size = None, _VoxelSize()
        else:
            blob = pickle.loads(self.path.read_bytes())
            self.data, self.voxel_size = blob["d"], _VoxelSize(blob["v"])

    def set_data(self, data):
        data = np.asarray(data)
        mode_from_dtype(data.dtype)          # raises exactly as mrcfile does
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._write and self.data is not None:
            v = self.voxel_size
            v = float(getattr(v, "x", v))
            self.path.write_bytes(pickle.dumps({"d": self.data, "v": v}))


def open(name, permissive=False, **kw):      # noqa: A001 - mirrors mrcfile
    return _File(name)


def new(name, overwrite=False, **kw):
    return _File(name, write=True)
