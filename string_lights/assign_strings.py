"""Assign pitched events to strings using acoustic evidence and a fingering beam search."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .audio import TRANSCRIPTION_FPS


def assign(notes, seconds, activity, *, tuning=(40,45,50,55,59,64), capo=0,
           max_fret=19, beam_size=128, tablature=None, evidence_window=0., fret_weight=0.):
    if len(tuning) != 6 or capo < 0 or max_fret < capo or beam_size < 1:
        raise ValueError('Use six tuning pitches, nonnegative capo, max fret >= capo, and positive beam size')
    seconds, activity = np.asarray(seconds), np.asarray(activity)
    if (seconds.ndim != 1 or len(seconds) == 0 or activity.shape != (len(seconds),6)
            or not np.isfinite(seconds).all() or not np.isfinite(activity).all()
            or np.any(np.diff(seconds) <= 0) or np.any((activity < 0) | (activity > 1))):
        raise ValueError('Provide increasing timestamps and finite (N,6) probabilities in [0,1]')
    if not np.isfinite([evidence_window, fret_weight]).all() or evidence_window < 0 or not 0 <= fret_weight <= 1:
        raise ValueError('Use a nonnegative window and fret weight in [0,1]')
    if tablature is not None:
        tablature = np.asarray(tablature)
        if (tablature.ndim != 3 or tablature.shape[:2] != activity.shape
                or tablature.shape[2] < max_fret + 2 or not np.isfinite(tablature).all()
                or np.any((tablature < 0) | (tablature > 1))):
            raise ValueError('Tablature must contain finite (N,6,fret classes plus silence) probabilities')
    elif fret_weight:
        raise ValueError('Fret evidence requires tablature probabilities')

    def evidence(values, onset):
        if evidence_window == 0:
            return float(np.interp(onset, seconds, values))
        # Preserve timestamps: only inspect acoustic evidence after the attack.
        samples = np.linspace(onset, onset + evidence_window, 7)
        return float(np.mean(np.interp(samples, seconds, values)))

    ordered = sorted(notes, key=lambda n:n['start'])
    for n in ordered:
        if not np.isfinite([n['start'],n['end'],n['pitch']]).all() or n['start'] < 0 or n['end'] <= n['start'] or n['pitch'] != int(n['pitch']):
            raise ValueError('Notes require nonnegative starts, later ends and integer MIDI pitches')
    # cost, active (pitch,end) per string, previous fretted position, assignment history
    beam = [(0., ((None,0.),)*6, 0., [])]
    groups=[]
    for note in ordered:
        if not groups or note['start']-groups[-1][0]['start'] > .04:
            groups.append([])
        groups[-1].append(note)
    for group in groups:
        candidates=[]
        for n in group:
            options=[(s,int(n['pitch']-base-capo)) for s,base in enumerate(tuning)
                     if 0 <= n['pitch']-base-capo <= max_fret-capo]
            candidates.append(options+[(None,None)])
        expanded=[]
        for choices in itertools.product(*candidates):
            strings=[s for s,f in choices if s is not None]
            if len(set(strings)) != len(strings):
                continue
            frets=[f for s,f in choices if s is not None and f>0]
            position=float(np.mean(frets)) if frets else 0.
            span=max(frets)-min(frets) if frets else 0
            acoustic=0.
            for n,(s,f) in zip(group,choices):
                if s is None:
                    acoustic+=7.
                else:
                    prob=evidence(activity[:,s], n['start'])
                    cost=-(1-fret_weight)*np.log(max(prob,.001))
                    if fret_weight:
                        # Checkpoint classes 0..19 are frets; 20 is silence.
                        fret_prob=evidence(tablature[:,s,f+capo], n['start'])
                        cost-=fret_weight*np.log(max(fret_prob,.001))
                    acoustic+=cost+.025*f
            for cost,active,previous,history in beam:
                state=list(active); events=[]; penalty=0.
                for n,(s,f) in zip(group,choices):
                    if s is not None:
                        pitch,end=state[s]
                        if end>n['start']+.04 and pitch != n['pitch']:
                            # Audio offsets include ringing/overlap; penalize replacing an occupied string.
                            penalty+=1.5*min(end-n['start'],1.)
                        state[s]=(n['pitch'],n['end'])
                    events.append(dict(n,string=s,fret=f))
                total=cost+acoustic+.12*abs(position-previous)+.5*max(0,span-4)**2+penalty
                expanded.append((total,tuple(state),position,history+events))
        expanded.sort(key=lambda item:item[0])
        beam=expanded[:beam_size]
    return beam[0][3] if beam else []


def export(events, duration, pulse=.09):
    if not np.isfinite([duration,pulse]).all() or duration <= 0 or pulse <= 0:
        raise ValueError('Duration and pulse must be finite and positive')
    result=np.zeros((int(np.ceil(duration*TRANSCRIPTION_FPS)),6),dtype=np.int64)
    for e in events:
        if e['string'] is not None:
            start=round(e['start']*TRANSCRIPTION_FPS)
            result[start:min(len(result),start+max(1,round(pulse*TRANSCRIPTION_FPS))),e['string']]=1
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notes',type=Path,help='JSON containing a notes array')
    parser.add_argument('probabilities',type=Path,help='NPZ containing seconds and activity')
    parser.add_argument('--model',default='gaps',help='Key in a multi-model notes JSON')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--duration',required=True,type=float)
    parser.add_argument('--capo',default=0,type=int)
    parser.add_argument('--tuning',nargs=6,type=int,default=[40,45,50,55,59,64])
    parser.add_argument('--pulse',default=.09,type=float)
    parser.add_argument('--evidence-window',type=float,default=0.,help='Seconds of post-attack evidence')
    parser.add_argument('--fret-weight',type=float,default=0.,help='Blend of fret evidence versus activity, 0..1')
    args=parser.parse_args()
    if args.output.suffix != '.npy': parser.error('Output must end in .npy')
    protected={args.notes.resolve(),args.probabilities.resolve()}
    if args.output.resolve() in protected or args.output.with_suffix('.json').resolve() in protected:
        parser.error('Output and sidecar must not overwrite input files')
    data=json.loads(args.notes.read_text()); notes=data.get('notes',data.get(args.model,{}).get('notes'))
    if notes is None: parser.error('No notes found')
    with np.load(args.probabilities,allow_pickle=False) as z:
        events=assign(notes,z['seconds'],z['activity'],tuning=args.tuning,capo=args.capo,tablature=z['tablature'] if 'tablature' in z else None,evidence_window=args.evidence_window,fret_weight=args.fret_weight)
    frames=export(events,args.duration,args.pulse)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,frames)
    args.output.with_suffix('.json').write_text(json.dumps(dict(evidence_window=args.evidence_window,fret_weight=args.fret_weight,tuning=args.tuning,capo=args.capo,fps=TRANSCRIPTION_FPS,pulse=args.pulse,events=events,caveat='Heuristic assignments; sounding probabilities are not pluck confidence. Fret numbers relative to capo.'),indent=2)+'\n')
    print(f'Saved {sum(e["string"] is not None for e in events)}/{len(events)} assignments to {args.output}')


if __name__=='__main__': main()
