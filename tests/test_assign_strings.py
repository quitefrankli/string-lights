import numpy as np
from string_lights.assign_strings import assign, export


def test_pitch_constraints_and_acoustic_preference():
    p=np.full((2,6),.01);p[:,4]=.99
    events=assign([dict(start=0.,end=.5,pitch=64)],[0.,1.],p)
    assert events[0]['string']==4
    assert events[0]['fret']==5
    frames=export(events,1.)
    assert frames.shape==(44,6)
    assert frames[:,4].sum()==4
    assert frames[:,[0,1,2,3,5]].sum()==0


def test_chord_cannot_use_same_string_twice_and_drops_impossible_pitch():
    notes=[dict(start=0.,end=.5,pitch=p) for p in [64,65,20]]
    events=assign(notes,[0.,1.],np.full((2,6),.9))
    strings=[e['string'] for e in events if e['string'] is not None]
    assert len(strings)==len(set(strings))==2
    assert events[2]['string'] is None


def test_capo_changes_open_pitch():
    p=np.full((2,6),.01);p[:,5]=.99
    e=assign([dict(start=.1,end=.5,pitch=66)],[0.,1.],p,capo=2)[0]
    assert (e['string'],e['fret'])==(5,0)


def test_fret_evidence_resolves_same_pitch_without_moving_attack():
    p=np.full((3,6),.8)
    tab=np.full((3,6,21),.001);tab[:,4,5]=.95
    notes=[dict(start=.1,end=.5,pitch=64)]
    e=assign(notes,[0.,.2,.4],p,tablature=tab,fret_weight=1.,evidence_window=.08)[0]
    assert (e['string'],e['fret'],e['start'])==(4,5,.1)


def test_window_uses_post_attack_evidence():
    p=np.full((3,6),.001);p[:,4]=.1;p[1:,5]=.99
    e=assign([dict(start=0.,end=.5,pitch=64)],[0.,.05,.1],p,evidence_window=.1)[0]
    assert e['string']==5
    assert e['start']==0.
