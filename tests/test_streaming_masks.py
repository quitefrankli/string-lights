import io
import zipfile

import numpy as np
import pytest

from string_lights.mask_cache import MaskCache
from string_lights.pipeline import masks_iter


def test_preview_reads_only_requested_masks_and_closes_stream(tmp_path, monkeypatch):
    path = tmp_path/'masks.npz'
    masks = np.arange(1000*64*64, dtype=np.uint8).reshape(1000,64,64)
    np.savez_compressed(path, masks=masks)
    reads = []
    closed = []
    original_read = zipfile.ZipExtFile.read
    original_close = zipfile.ZipExtFile.close

    def read(self, size=-1):
        assert 0 <= size <= 64*64, 'Must not read the entire member'
        result = original_read(self, size)
        reads.append(len(result))
        return result

    def close(self):
        closed.append(self)
        return original_close(self)

    monkeypatch.setattr(zipfile.ZipExtFile, 'read', read)
    monkeypatch.setattr(zipfile.ZipExtFile, 'close', close)
    cache = MaskCache(path)
    assert sum(reads) < 1024
    reader = masks_iter(cache, np.arange(1000), 1000, valid_frames=1000)
    np.testing.assert_array_equal(next(reader), masks[0])
    np.testing.assert_array_equal(next(reader), masks[1])
    reader.close()
    assert sum(reads) < 3*64*64
    assert all(stream.closed for stream in closed)
    # Reopening starts at frame zero and preserves mask values exactly.
    np.testing.assert_array_equal(next(cache.frames()), masks[0])


def test_streamed_sparse_masks_match_array_path_and_short_cache_fallback(tmp_path):
    path = tmp_path/'masks.npz'
    masks = np.array([[[64, 0]], [[128, 255]]], dtype=np.uint8)
    np.savez_compressed(path, masks=masks)
    indices = np.array([0, 2])
    streamed = list(masks_iter(MaskCache(path), indices, 7, valid_frames=4))
    expected = list(masks_iter(masks, indices, 7, valid_frames=4))
    np.testing.assert_array_equal(streamed, expected)
    assert [mask[0,0] for mask in streamed] == [64,64,128,128,0,0,0]


@pytest.mark.parametrize('array', [np.zeros((2,3),np.uint8),
                                   np.zeros((2,3,4),np.float32),
                                   np.asfortranarray(np.zeros((2,3,4),np.uint8))])
def test_rejects_nonstreamable_arrays(tmp_path, array):
    path=tmp_path/'invalid.npz'
    np.savez_compressed(path,masks=array)
    with pytest.raises(ValueError,match='C-order uint8'):
        MaskCache(path)


def test_rejects_truncated_array_before_rendering(tmp_path):
    buffer=io.BytesIO()
    np.save(buffer,np.zeros((3,4,5),np.uint8))
    path=tmp_path/'truncated.npz'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('masks.npy',buffer.getvalue()[:-1])
    with pytest.raises(ValueError,match='size does not match'):
        MaskCache(path)
