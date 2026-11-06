import numpy as np
import pytest
from string_lights.tab_refinement import align,validate_score


def score(events,capo=2):
    return dict(capo=capo,issues=[],events=[dict(page=1,bar=i+2,uncertain=False,notes=[dict(string=s,fret=f) for s,f in notes]) for i,notes in enumerate(events)])


def test_alignment_changes_strings_preserves_audio_timing_and_skips_extra_note():
    notes=[dict(start=.1,end=.3,pitch=66,string=4,fret=5),
           dict(start=.5,end=.7,pitch=108,string=None,fret=None),
           dict(start=1.,end=1.3,pitch=61,string=3,fret=4)]
    refined,report=align(notes,score([[(5,0)],[(4,0)]]),[40,45,50,55,59,64],2)
    assert [n['string'] for n in refined]==[5,None,4]
    assert [(n['start'],n['end'],n['pitch']) for n in refined]==[(n['start'],n['end'],n['pitch']) for n in notes]
    assert report['changed_notes']==2
    assert report['unresolved_event_indices']==[1]


def test_chord_alignment_does_not_depend_on_strum_pitch_order():
    notes=[dict(start=.1,end=.8,pitch=66,string=4,fret=5),dict(start=.12,end=.8,pitch=52,string=1,fret=5)]
    refined,report=align(notes,score([[(2,0),(5,0)]]),[40,45,50,55,59,64],2)
    assert [n['string'] for n in refined]==[5,2]
    assert report['matched_groups']==1


def test_uncertain_events_do_not_override_audio():
    s=score([[(5,0)],[(4,0)]]);s['events'][0]['uncertain']=True
    events=[dict(start=.1,end=.3,pitch=66,string=3,fret=9)]
    refined,report=align(events,s,[40,45,50,55,59,64],2)
    assert refined==events
    assert report['matched_notes']==0


def test_capo_conflict_and_duplicate_string_rejected():
    with pytest.raises(ValueError,match='capo'):
        align([],score([[(5,0)]]),[40,45,50,55,59,64],0)
    with pytest.raises(ValueError,match='same string'):
        validate_score(score([[(5,0),(5,3)]]))


def test_missing_tab_event_does_not_shift_following_notes():
    events=[dict(start=.1,end=.3,pitch=66,string=3,fret=9),dict(start=1,end=1.5,pitch=52,string=1,fret=5)]
    refined,_=align(events,score([[(5,0)],[(4,0)],[(2,0)]]),[40,45,50,55,59,64],2)
    assert [n['string'] for n in refined]==[5,2]


def test_cli_uses_sidecar_and_preserves_source_pair(tmp_path,monkeypatch):
    import json,sys
    from string_lights import tab_refinement as module
    from string_lights.transcription import write_outputs
    from string_lights.transcription_models import sha256
    source=tmp_path/'original.npy';output=tmp_path/'refined.npy'
    events=[dict(start=.1,end=.5,pitch=66,string=4,fret=5)]
    write_outputs(source,np.zeros((44,6),dtype=np.int64),dict(events=events,capo=2,tuning_midi=[40,45,50,55,59,64],duration_seconds=1.,pulse_seconds=.09,column_order='low_to_high',caveats=[]))
    original=source.read_bytes();sidecar=source.with_suffix('.json').read_bytes()
    monkeypatch.setattr(module,'extract_tabs',lambda *a,**kw:(score([[(5,0)]]),{'test':True}))
    monkeypatch.setattr(sys,'argv',['refine-transcription','song','--tabs',str(tmp_path/'tab.pdf'),'--transcription',str(source),'--output',str(output)])
    module.main()
    assert source.read_bytes()==original and source.with_suffix('.json').read_bytes()==sidecar
    assert np.load(output)[4,5]==1
    meta=json.loads(output.with_suffix('.json').read_text())
    assert meta['array_sha256']==sha256(output)
    assert meta['events'][0]['start']==.1
    assert meta['refinement']['alignment']['changed_notes']==1


def test_extraction_cache_avoids_codex_inference(tmp_path,monkeypatch):
    import json
    from string_lights import tab_refinement as module
    pdf=tmp_path/'input.pdf';pdf.write_bytes(b'pdf')
    monkeypatch.setattr(module.shutil,'which',lambda cmd:'/bin/'+cmd)
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw:'codex-cli test')
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        if cmd[0]=='codex':
            assert cmd[cmd.index('--sandbox')+1]=='read-only'
            assert cmd[cmd.index('--model')+1]=='gpt-6.1-sol'
            assert cmd[cmd.index('-c')+1]=='model_reasoning_effort="low"'
            assert '--output-schema' in cmd and '--image' in cmd
            assert 'Do not run tools' in kwargs['input']
            from types import SimpleNamespace
            raw=cmd[cmd.index('-o')+1]
            Path(raw).write_text(json.dumps(score([[(5,0)]])))
            return SimpleNamespace(returncode=0)
    from pathlib import Path
    monkeypatch.setattr(module.subprocess,'run',run)
    for _ in range(2):
        module.extract_tabs(pdf,tmp_path/'work',pages=[1],first_bar=2,last_bar=2)
    assert sum(cmd[0]=='codex' for cmd in calls)==1
    manifest=json.loads((tmp_path/'work'/'extraction.json').read_text())
    assert manifest['request']['requested_model']=='gpt-6.1-sol'
    assert manifest['request']['reasoning_effort']=='low'
    manifest['request'].pop('reasoning_effort')
    (tmp_path/'work'/'extraction.json').write_text(json.dumps(manifest))
    module.extract_tabs(pdf,tmp_path/'work',pages=[1],first_bar=2,last_bar=2)
    assert sum(cmd[0]=='codex' for cmd in calls)==2
