import pytest
from string_lights import timings

@pytest.mark.parametrize('delta,gap', [(float('nan'), .03), (.07, -1)])
def test_invalid_detector_settings(delta, gap):
    with pytest.raises(ValueError):
        timings.detect('unused.mp4', delta=delta, min_gap=gap)
