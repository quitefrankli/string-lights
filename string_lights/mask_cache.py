"""Sequential access to compressed mask arrays without materializing the cache."""
from pathlib import Path
from typing import Iterator
import zipfile

import numpy as np
from numpy.lib import format as npy_format


def _header(stream):
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = npy_format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = npy_format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported mask array format: {version}")
    if len(shape) != 3 or dtype != np.dtype('uint8') or fortran or any(d < 0 for d in shape) or any(d == 0 for d in shape[1:]):
        raise ValueError("Mask cache must contain a C-order uint8 array of shape (frames, height, width)")
    return shape


class MaskCache:
    """A reopenable stream; only the small array header is read at construction."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive, archive.open('masks.npy') as stream:
            self.shape = _header(stream)
            expected_size = stream.tell() + int(np.prod(self.shape, dtype=object))
            if archive.getinfo('masks.npy').file_size != expected_size:
                raise ValueError("Mask cache size does not match its array header")

    def frames(self) -> Iterator[np.ndarray]:
        with zipfile.ZipFile(self.path) as archive, archive.open('masks.npy') as stream:
            if _header(stream) != self.shape:
                raise ValueError("Mask cache changed while rendering")
            frame_bytes = self.shape[1] * self.shape[2]
            for _ in range(self.shape[0]):
                raw = stream.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise ValueError("Truncated mask cache")
                yield np.frombuffer(raw, dtype=np.uint8).reshape(self.shape[1:])
