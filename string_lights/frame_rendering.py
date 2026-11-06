"""Bounded, ordered frame compositing with independent per-frame fade state."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import cv2
from .strings import draw_projected_strings_frame


def _composite(job):
    frame, index, lines, active, last_active, fps, mask = job
    draw_projected_strings_frame(frame, index, lines, active, last_active, fps, mask=mask)
    return frame


def composite_frames(frames, masks, projections, strings, fps, workers=1):
    """Yield frames in order, keeping at most twice the worker count in flight."""
    if workers < 1:
        raise ValueError('Render workers must be positive')

    def jobs():
        last_active = {}
        for index, (frame, mask) in enumerate(zip(frames, masks)):
            for string in strings[index]:
                last_active[string] = index
            yield (frame, index, projections[index], strings[index],
                   last_active.copy(), fps, mask)

    with closing(jobs()) as source:
        if workers == 1:
            for job in source:
                yield _composite(job)
            return
        # Avoid multiplying OpenCV's own thread pool by the number of workers.
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        pending = deque()
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            for _ in range(2 * workers):
                job = next(source, None)
                if job is None:
                    break
                pending.append(executor.submit(_composite, job))
            while pending:
                yield pending.popleft().result()
                job = next(source, None)
                if job is not None:
                    pending.append(executor.submit(_composite, job))
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            cv2.setNumThreads(previous_threads)
