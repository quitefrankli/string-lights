"""Read tab PDFs with Codex and align their strings to detected audio notes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import paths
from .assign_strings import export
from .transcription import write_outputs
from .transcription_models import sha256

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'capo': {'type': ['integer', 'null']},
        'issues': {'type': 'array', 'items': {'type': 'string'}},
        'events': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'page': {'type': 'integer'}, 'bar': {'type': 'integer'},
                'uncertain': {'type': 'boolean'},
                'notes': {'type': 'array', 'items': {
                    'type': 'object', 'additionalProperties': False,
                    'properties': {'string': {'type': 'integer'}, 'fret': {'type': 'integer'}},
                    'required': ['string', 'fret'],
                }},
            }, 'required': ['page', 'bar', 'uncertain', 'notes'],
        }},
    }, 'required': ['capo', 'issues', 'events'],
}


def validate_score(score):
    if not isinstance(score, dict) or set(score) != {'capo', 'issues', 'events'}:
        raise ValueError('Invalid extracted score object')
    if score['capo'] is not None and (type(score['capo']) is not int or not 0 <= score['capo'] <= 19):
        raise ValueError('Invalid extracted capo')
    if not isinstance(score['issues'], list) or any(not isinstance(i,str) for i in score['issues']):
        raise ValueError('Score issues must be strings')
    if not isinstance(score['events'], list) or not score['events']:
        raise ValueError('No tab events extracted')
    for event in score['events']:
        if set(event) != {'page', 'bar', 'uncertain', 'notes'} or type(event['uncertain']) is not bool:
            raise ValueError('Invalid score event')
        if any(type(event[k]) is not int or event[k] < 1 for k in ('page','bar')):
            raise ValueError('Invalid page/bar number')
        strings = []
        if not isinstance(event['notes'],list) or not event['notes']:
            raise ValueError('Score event needs notes')
        for note in event['notes']:
            if (set(note) != {'string','fret'} or type(note['string']) is not int
                    or type(note['fret']) is not int or not 0 <= note['string'] < 6
                    or not 0 <= note['fret'] <= 24):
                raise ValueError('Use string indices 0..5 and frets 0..24')
            strings.append(note['string'])
        if len(strings) != len(set(strings)):
            raise ValueError('A tab event cannot play two frets on the same string')
    return score


def extract_tabs(pdf, work_dir, *, pages, first_bar, last_bar, model=None, timeout=300):
    """Codex sees page images only; application code writes and validates results."""
    if pages is None:
        info = subprocess.check_output(['pdfinfo', str(pdf.resolve())], text=True)
        count = re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)
        if count is None:
            raise ValueError('Could not determine PDF page count')
        pages = list(range(1, int(count.group(1)) + 1))
    if not shutil.which('codex') or not shutil.which('pdftoppm'):
        raise ValueError('PDF refinement requires codex and pdftoppm on PATH')
    work_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output(['codex','--version'], text=True).strip()
    identity = dict(pdf_sha256=sha256(pdf), pages=pages, first_bar=first_bar,
                    last_bar=last_bar, requested_model=model, prompt_version=1,
                    schema_sha256=__import__('hashlib').sha256(json.dumps(SCHEMA,sort_keys=True).encode()).hexdigest(),
                    codex_version=version)
    cache_file, manifest = work_dir/'tabs.json', work_dir/'extraction.json'
    if cache_file.exists() and manifest.exists():
        cached = json.loads(manifest.read_text())
        if cached.get('request', cached) == identity:
            actual_model = cached.get('actual_model')
            log_path = work_dir/'codex.log'
            if actual_model is None and log_path.exists():
                found = re.search(r'^model:\s+(.+)$', log_path.read_text(), re.MULTILINE)
                actual_model = found.group(1).strip() if found else None
            return validate_score(json.loads(cache_file.read_text())), dict(identity, actual_model=actual_model)
    images = []
    for page in pages:
        stem = work_dir/f'page-{page}'
        subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-singlefile',
                        '-scale-to','2400','-png',str(pdf.resolve()),str(stem.resolve())], check=True)
        images.append(stem.with_suffix('.png').resolve())
    schema_path = work_dir/'schema.json'
    schema_path.write_text(json.dumps(SCHEMA,indent=2)+'\n')
    prompt = f'''Read the supplied guitar tablature page images. Return only the requested structured data.
Extract bars {first_bar or 'from the beginning'} through {last_bar or 'the end of the supplied pages'} inclusive, in musical playback order.
String indices ALWAYS run LOW TO HIGH: 0=low E, 1=A, 2=D, 3=G, 4=B, 5=high e.
In conventional printed tablature the TOP line is string 5 and BOTTOM line is string 0.
Frets are relative to the capo; report the printed capo separately. Do not transpose fret numbers.
Each event is one attack; simultaneous vertically stacked notes form one event, rolled strum arrows still one event.
Preserve the sequence of notes, not horizontal pixel spacing as timing. No timestamps or BPM inference.
Ignore rests and X/body percussion. Ignore chord-name text. For hammer-ons, pull-offs, bends and slides include
only the initial plucked note, unless the score explicitly marks another pluck. Mark ambiguous events uncertain.
Expand explicit repeats in playback order when unambiguous; otherwise record the ambiguity in issues.
Do not invent obscured notes. Return page numbers matching attached original PDF pages {pages}.
Use only the images. Do not inspect project files, existing annotations or audio transcriptions. Do not run tools or write files.
'''
    (work_dir/'prompt.txt').write_text(prompt)
    raw = work_dir/'codex-result.json'
    raw.unlink(missing_ok=True)
    cmd = ['codex','exec','--ephemeral','--sandbox','read-only',
           '--output-schema',str(schema_path.resolve()),'-o',str(raw.resolve())]
    if model:
        cmd += ['--model',model]
    for image in images:
        cmd += ['--image',str(image)]
    cmd += ['-']
    try:
        with (work_dir/'codex.log').open('w') as log:
            result = subprocess.run(cmd, input=prompt, text=True, stdout=log, stderr=log,
                                    cwd=work_dir.resolve(), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f'Codex extraction timed out; see {work_dir / "codex.log"}') from exc
    if result.returncode or not raw.exists():
        raise ValueError(f'Codex extraction failed; see {work_dir / "codex.log"}')
    score = validate_score(json.loads(raw.read_text()))
    if any(e['page'] not in pages or first_bar is not None and e['bar'] < first_bar
           or last_bar is not None and e['bar'] > last_bar for e in score['events']):
        raise ValueError('Extracted events fall outside the requested pages or bars')
    cache_file.write_text(json.dumps(score,indent=2)+'\n')
    log_text = (work_dir/'codex.log').read_text()
    model_match = re.search(r'^model:\s+(.+)$', log_text, re.MULTILINE)
    actual_model = model_match.group(1).strip() if model_match else None
    manifest.write_text(json.dumps(dict(request=identity, actual_model=actual_model),indent=2)+'\n')
    return score, dict(identity, actual_model=actual_model)


def align(events, score, tuning, capo, *, group_window=.08):
    """Monotonic pitch-set alignment; retain audio timing and unmatched assignments."""
    validate_score(score)
    if score['capo'] is not None and score['capo'] != capo:
        raise ValueError(f'PDF capo {score["capo"]} differs from transcription capo {capo}; regenerate with correct capo')
    groups = []
    for index in sorted(range(len(events)), key=lambda i: events[i]['start']):
        if not groups or events[index]['start']-events[groups[-1][0]]['start'] > group_window:
            groups.append([])
        groups[-1].append(index)
    tabs = [e for e in score['events'] if not e['uncertain']]
    audio_pitches = [Counter(events[i]['pitch'] for i in group) for group in groups]
    tab_pitches = [Counter(tuning[n['string']]+capo+n['fret'] for n in e['notes']) for e in tabs]
    m,n=len(groups),len(tabs)
    if not m or not n:
        raise ValueError('No alignable events')
    dp=np.full((m+1,n+1),np.inf);dp[0,0]=0
    trace=np.zeros((m+1,n+1),np.uint8)
    for i in range(m+1):
        for j in range(n+1):
            if i and dp[i-1,j]+1.05 < dp[i,j]:
                dp[i,j]=dp[i-1,j]+1.05;trace[i,j]=1
            if j and dp[i,j-1]+.65 < dp[i,j]:
                dp[i,j]=dp[i,j-1]+.65;trace[i,j]=2
            if i and j:
                common=sum((audio_pitches[i-1]&tab_pitches[j-1]).values())
                if common:
                    cost=1.5*(1-2*common/(sum(audio_pitches[i-1].values())+sum(tab_pitches[j-1].values())))
                    if dp[i-1,j-1]+cost < dp[i,j]:
                        dp[i,j]=dp[i-1,j-1]+cost;trace[i,j]=3
    # A short audio excerpt may stop before the end of the selected PDF range.
    i,j=m,int(np.argmin(dp[m]));pairs=[]
    while i or j:
        action=trace[i,j]
        if action==3:
            pairs.append((i-1,j-1));i-=1;j-=1
        elif action==1:i-=1
        elif action==2:j-=1
        else:break
    refined=[dict(e) for e in events];matches=[]
    for ai,ti in reversed(pairs):
        by_pitch={}
        for note in tabs[ti]['notes']:
            pitch=tuning[note['string']]+capo+note['fret']
            by_pitch.setdefault(pitch,[]).append(note)
        audio_counts=audio_pitches[ai]
        for index in groups[ai]:
            pitch=events[index]['pitch'];candidates=by_pitch.get(pitch,[])
            # Avoid arbitrarily assigning duplicated unison notes or reuse of one tab note.
            if len(candidates)!=1 or audio_counts[pitch]!=1:
                continue
            note=candidates[0];old=events[index]
            refined[index]=dict(old,string=note['string'],fret=note['fret'])
            matches.append(dict(event_index=index,seconds=old['start'],pitch=pitch,
                                old_string=old['string'],new_string=note['string'],
                                bar=tabs[ti]['bar'],page=tabs[ti]['page']))
    used={x['event_index'] for x in matches}
    return refined,dict(matched_notes=len(matches),changed_notes=sum(e['old_string']!=e['new_string'] for e in matches),
                        untouched_notes=len(events)-len(matches),audio_groups=m,tab_groups=n,
                        matched_groups=len(pairs),matches=matches,
                        unresolved_event_indices=[i for i in range(len(events)) if i not in used])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',help='Project name or video path')
    parser.add_argument('--tabs',required=True,type=Path)
    parser.add_argument('--transcription',type=Path,help='Existing .npy with same-stem generation JSON')
    parser.add_argument('--output',type=Path,help='Default: update project transcription and companion')
    parser.add_argument('--projects-dir',type=Path,default=Path('data/projects'))
    parser.add_argument('--project-dir',type=Path)
    parser.add_argument('--work-dir',type=Path,help='Default: project tmp/tab-refinement')
    parser.add_argument('--pages',nargs='+',type=int,default=None,help='PDF pages to read (default: all)')
    parser.add_argument('--first-bar',type=int)
    parser.add_argument('--last-bar',type=int)
    parser.add_argument('--codex-model')
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--min-coverage',type=float,default=.5,help='Minimum fraction of audio notes matched (default: .5)')
    parser.add_argument('--force',action='store_true',help='Allow replacing output and companion')
    args=parser.parse_args()
    old_root=paths.PROJECTS_DIR;paths.PROJECTS_DIR=args.projects_dir
    token=paths.project_override.set(args.project_dir)
    try:
        if (not np.isfinite(args.min_coverage) or not 0 <= args.min_coverage <= 1
                or args.pages is not None and any(p<1 for p in args.pages) or args.timeout<=0
                or not np.isfinite(args.timeout) or any(b is not None and b<1 for b in [args.first_bar,args.last_bar])
                or args.first_bar and args.last_bar and args.first_bar>args.last_bar):
            raise ValueError('Use positive pages, bars and timeout with ordered bar range')
        source=args.transcription or paths.project_dir(args.input)/'transcription.npy'
        output=args.output or paths.project_dir(args.input)/'transcription.npy'
        work=args.work_dir or paths.project_dir(args.input)/'tmp/tab-refinement'
        if output.suffix!='.npy':raise ValueError('Output must end in .npy')
        if args.tabs.resolve() in (output.resolve(),output.with_suffix('.json').resolve()):
            raise ValueError('Cannot overwrite PDF')
        if not args.force and (output.exists() or output.with_suffix('.json').exists()):
            raise ValueError('Output exists; use --output for a candidate or --force to replace')
        metadata=json.loads(source.with_suffix('.json').read_text())
        if metadata.get('array_sha256')!=sha256(source):
            raise ValueError('Companion does not match array; regenerate or reconcile editor changes before refinement')
        if metadata.get('column_order')!='low_to_high':raise ValueError('Expected low-to-high string order')
        score,provenance=extract_tabs(args.tabs,work,pages=args.pages,first_bar=args.first_bar,
                                     last_bar=args.last_bar,model=args.codex_model,timeout=args.timeout)
        refined,report=align(metadata['events'],score,metadata['tuning_midi'],metadata['capo'])
        if not report['matched_notes']:raise ValueError('No pitch matches found; outputs were not changed')
        if report['matched_notes'] / len(metadata['events']) < args.min_coverage:
            raise ValueError('Too few notes aligned; inspect the extracted tabs and bar range, or explicitly lower --min-coverage')
        frames=export(refined,metadata['duration_seconds'],metadata['pulse_seconds'])
        updated=dict(metadata,events=refined,unassigned_count=sum(e['string'] is None for e in refined),
                     refined_at=datetime.now(timezone.utc).isoformat(),
                     refinement=dict(pdf=str(args.tabs.resolve()),extraction=provenance,alignment=report,
                                     source_array_sha256=sha256(source),issues=score['issues'],min_coverage=args.min_coverage,
                                     extracted_score=score,
                                     alignment_settings=dict(version=1,group_window_seconds=.08,
                                                             skip_audio_cost=1.05,skip_tab_cost=.65),
                                     caveat='Heuristic sequence alignment; unmatched notes retain audio assignments'))
        updated['caveats']=[c for c in metadata.get('caveats',[]) if c!='PDF tab refinement has not been applied']
        write_outputs(output,frames,updated,args.force)
        print(f'Saved {output} and JSON: {report["matched_notes"]} matched, '
              f'{report["changed_notes"]} string changes, {report["untouched_notes"]} unresolved')
    except (ValueError,KeyError,TypeError,OSError,subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    finally:
        paths.project_override.reset(token);paths.PROJECTS_DIR=old_root


if __name__=='__main__':main()
