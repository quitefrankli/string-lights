from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
import pytest

from string_lights import frame_rendering
from string_lights.masking import blend_mask
from string_lights.strings import draw_projected_strings_frame


def test_parallel_compositing_matches_original_serial_fades_and_masks():
    rng=np.random.default_rng(42)
    frames=rng.integers(0,256,(24,80,100,3),dtype=np.uint8)
    masks=rng.integers(0,256,(24,80,100),dtype=np.uint8)
    masks[::3]=0
    lines=np.tile(np.array([[[5,15+s*8],[95,15+s*8]] for s in range(6)],float),(24,1,1,1))
    lines[5]=np.nan
    active=[[0,2] if i<3 else [4] if i==5 else [2,5] if 10<=i<14 else [] for i in range(24)]
    expected=[];last={}
    for i,frame in enumerate(frames.copy()):
        original=frame.copy()
        draw_projected_strings_frame(frame,i,lines[i],active[i],last,10.)
        expected.append(blend_mask(original,frame,masks[i]) if masks[i].any() else frame)
    old_threads=cv2.getNumThreads()
    for workers in [1,2,4]:
        actual=list(frame_rendering.composite_frames(iter(frames.copy()),iter(masks),lines,active,10.,workers))
        np.testing.assert_array_equal(actual,expected)
        assert cv2.getNumThreads()==old_threads


def test_parallel_queue_is_bounded_ordered_and_restores_threads_on_early_close(monkeypatch):
    release=threading.Event();started=threading.Event();read=[]
    def source():
        for i in range(1000):
            read.append(i)
            yield np.full((1,1,3),i,dtype=np.int64)
    def composite(job):
        started.set()
        assert release.wait(5)
        return job[0]
    monkeypatch.setattr(frame_rendering,'_composite',composite)
    old_threads=cv2.getNumThreads()
    generator=frame_rendering.composite_frames(source(),iter([np.zeros((1,1))]*1000),[None]*1000,[[]]*1000,30.,workers=2)
    with ThreadPoolExecutor(max_workers=1) as consumer:
        first=consumer.submit(next,generator)
        assert started.wait(5)
        # No additional frames can be queued while the first result is blocked.
        release.set()
        assert first.result(timeout=5)[0,0,0]==0
    assert read==list(range(4))
    generator.close()
    assert cv2.getNumThreads()==old_threads


def test_worker_failure_propagates_and_restores_threads(monkeypatch):
    def fail(job):raise RuntimeError('compositing failed')
    monkeypatch.setattr(frame_rendering,'_composite',fail)
    old_threads=cv2.getNumThreads()
    with pytest.raises(RuntimeError,match='compositing failed'):
        list(frame_rendering.composite_frames(iter([None]*8),iter([None]*8),[None]*8,[[]]*8,30.,workers=2))
    assert cv2.getNumThreads()==old_threads
