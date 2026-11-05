import json
import sys
from pathlib import Path
import numpy as np
import pytest
from string_lights import transcription as module
from string_lights.transcription_models import sha256


def test_cli_project_and_companion(tmp_path, monkeypatch):
    project=tmp_path/'song'; project.mkdir(); (project/'source.mp4').touch()
    def generate(video, **kwargs):
        assert video == project/'source.mp4'
        assert kwargs['capo']==2
        return np.eye(6,dtype=np.int64), dict(events=[{'start':.1,'string':0}],unassigned_count=0,capo=2)
    monkeypatch.setattr(module,'generate',generate)
    monkeypatch.setattr(sys,'argv',['generate-transcription','song','--projects-dir',str(tmp_path),'--capo','2'])
    module.main()
    output=project/'transcription.npy'
    np.testing.assert_array_equal(np.load(output),np.eye(6))
    metadata=json.loads(output.with_suffix('.json').read_text())
    assert metadata['capo']==2
    assert metadata['array_sha256']==sha256(output)


def test_sidecar_conflict_preserves_both(tmp_path):
    output=tmp_path/'transcription.npy'
    sidecar=output.with_suffix('.json');sidecar.write_text('original')
    with pytest.raises(ValueError,match='exists'):
        module.write_outputs(output,np.zeros((2,6)),{},False)
    assert not output.exists()
    assert sidecar.read_text()=='original'


def test_failed_serialization_does_not_replace_existing_pair(tmp_path):
    output=tmp_path/'transcription.npy';np.save(output,[[9]])
    sidecar=output.with_suffix('.json');sidecar.write_text('original')
    with pytest.raises(ValueError):
        module.write_outputs(output,np.zeros((2,6)),{'invalid':float('nan')},True)
    np.testing.assert_array_equal(np.load(output),[[9]])
    assert sidecar.read_text()=='original'


def test_video_offset_and_capo_preserve_pitch(tmp_path, monkeypatch):
    video=tmp_path/'source.mp4';video.write_bytes(b'video')
    monkeypatch.setattr(module,'extract',lambda *a:(np.ones(22050),22050,.2,1.))
    monkeypatch.setattr(module,'infer',lambda *a,**kw:([dict(start=.1,end=.5,pitch=66,velocity=90)],np.array([0.,1.]),np.array([[.001]*5+[.99]]*2),{'revision':'test'}))
    frames,meta=module.generate(video,capo=2,tuning=[40,45,50,55,59,64],model='gaps',device='cpu',cache_dir=tmp_path,duration=None,pulse=.09,evidence_window=.08)
    event=meta['events'][0]
    assert event['start']==pytest.approx(.3)
    assert (event['pitch'],event['string'],event['fret'])==(66,5,0)
    assert frames.shape==(44,6)
    assert frames[13,5]==1


def test_force_replaces_pair_and_checksum(tmp_path):
    output=tmp_path/'custom.npy'
    module.write_outputs(output,np.zeros((1,6)),{'capo':0})
    module.write_outputs(output,np.ones((2,6)),{'capo':2},force=True)
    assert np.load(output).shape==(2,6)
    meta=json.loads(output.with_suffix('.json').read_text())
    assert meta['capo']==2
    assert meta['array_sha256']==sha256(output)


def test_cli_external_input_output_overrides(tmp_path, monkeypatch):
    source=tmp_path/'input.mp4';source.touch()
    target=tmp_path/'artifacts'
    monkeypatch.setattr(module,'generate',lambda *a,**kw:(np.zeros((44,6)),{'events':[],'unassigned_count':0}))
    monkeypatch.setattr(sys,'argv',['generate-transcription',str(source),'--project-dir',str(target)])
    module.main()
    assert (target/'transcription.npy').exists()
    assert (target/'transcription.json').exists()
