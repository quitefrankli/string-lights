import numpy as np

from string_lights.audio import get_strings_to_highlight


def test_rendering_preserves_plucks_between_quarter_second_grid_lines(tmp_path):
    source = tmp_path / 'clip.mp4'
    frames = np.zeros((44, 6), dtype=np.int64)
    frames[6:8, 0] = 1
    frames[12:15, 5] = 1
    np.save(source.with_suffix('.npy'), frames)
    strings = get_strings_to_highlight(str(source), total_frames=44, video_fps=44)
    assert strings[6] == [0]
    assert strings[7] == [0]
    assert strings[5] == []
    assert strings[8] == []
    assert strings[12] == [5]
    assert strings[15] == []


def test_all_columns_reach_matching_projected_line(tmp_path):
    source = tmp_path / 'clip.mp4'
    np.save(source.with_suffix('.npy'), np.eye(6, dtype=np.int64))
    assert get_strings_to_highlight(str(source), 6, 44) == [[i] for i in range(6)]
