"""Post-run SFT02 analysis: no model inference or training."""
import argparse, csv, datetime as dt, hashlib, json, re
from collections import Counter
from pathlib import Path

DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
COHORTS = ['id','expression','unseen_source','unseen_offset','composition']

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, data): p.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
def csvsave(p, rows):
    with p.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(root):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.chronogeom.clock_teacher_lora01 import bootstrap
    bootstrap()
    import torch
    assert (root/'completion.json').exists() and read(root/'process_exit.json')['exit_code']==0
    out=Path('docs/chronogeom/clock_teacher_sft02')
    out.mkdir(exist_ok=False)
    c=read(root/'config.json'); done=read(root/'completion.json')
    source=read(root/'source.json'); lock=read(root/'test_lock.json')
    rows=read(root/'examples.json'); old=read(Path(c['previous_run'])/'examples.json')
    test=[x for x in rows if x['split'] in COHORTS]; byid={x['id']:x for x in rows}
    assert len(test)==240
    for split in ['train','val']:
        assert [x for x in rows if x['split']==split]==[x for x in old if x['split']==split]
    assert len({tuple(x['identity']) for x in test})==240
    assert not ({tuple(x['identity']) for x in test}&{tuple(x['identity']) for x in old})
    for row in rows:
        t=dt.datetime(2024,1,1)+dt.timedelta(days=row['day'],hours=row['hour']+row['delta'])
        assert (t.weekday(),t.hour)==(row['answer_day'],row['answer_hour'])
        if row['parts']: assert sum(row['parts'])==row['delta']
    curve=read(root/'validation_curve.json')
    chosen=min(curve,key=lambda x:(-x['correct'],x['answer_token_ce'],x['total_updates']))
    assert chosen==lock['selected']==done['selected']
    assert sha(Path(chosen['path']))==chosen['sha256']
    assert sha(Path(c['warm_adapter']))==c['warm_adapter_sha256']
    assert sha(root/'examples.json')==source['examples_sha256']
    assert sha(Path('scripts/chronogeom/clock_teacher_sft02.py'))==source['script_sha256']
    adapters=[]
    for x in curve:
        t=torch.load(x['path'],map_location='cpu',weights_only=True)
        assert len(t)==16 and sum(v.numel() for v in t.values())==262144
        assert all(bool(torch.isfinite(v).all()) for v in t.values())
        assert sha(Path(x['path']))==x['sha256']
        adapters.append(dict(total_updates=x['total_updates'],sha256=x['sha256'],finite=True))
    pred=read(root/'new_test_predictions.json')
    assert len(pred)==480
    def verify(p):
        row=byid[p['id']]
        m=re.fullmatch(r'\s*('+'|'.join(DAYS)+r')\s+(\d{1,2}):00\s*[.!]?\s*',p['text'])
        valid=m is not None and int(m.group(2))<24
        dow=bool(valid and DAYS.index(m.group(1))==row['answer_day'])
        hour=bool(valid and int(m.group(2))==row['answer_hour'])
        for k,v in dict(valid_format=bool(valid),weekday_correct=dow,hour_correct=hour,correct=dow and hour).items():assert p[k]==v
        assert p['expected']==row['answer'].strip()
    for p in pred:verify(p)
    for p in read(root/'selected_train64_predictions.json'):verify(p)
    for x in curve:
        vp=read(root/f"val_total{x['total_updates']}.json")
        for p in vp:verify(p)
        assert sum(p['correct'] for p in vp)==x['correct']
    summary=[]
    for arm in ['Warm128','Selected']:
        for group in ['all']+COHORTS+['carry','noncarry']:
            rr=[p for p in pred if p['arm']==arm and
                (group=='all' or p['split']==group or group=='carry' and p['carry'] or group=='noncarry' and not p['carry'])]
            summary.append(dict(arm=arm,group=group,n=len(rr),source_ids=len({p['source_id'] for p in rr}),
                **{k:sum(p[k] for p in rr) for k in ['correct','valid_format','weekday_correct','hour_correct']}))
    csvsave(out/'summary.csv',summary)
    warm={p['id']:p for p in pred if p['arm']=='Warm128'}
    selected={p['id']:p for p in pred if p['arm']=='Selected'}
    paired=Counter((warm[i]['correct'],selected[i]['correct']) for i in warm)
    stats=[];rng=np.random.default_rng(27102703)
    for group in ['all']+COHORTS+['carry','noncarry']:
        ids=[x['id'] for x in test if group=='all' or x['split']==group or group=='carry' and x['carry'] or group=='noncarry' and not x['carry']]
        sources=sorted({byid[i]['source_id'] for i in ids})
        num=np.array([sum(int(selected[i]['correct'])-int(warm[i]['correct']) for i in ids if byid[i]['source_id']==s) for s in sources])
        den=np.array([sum(byid[i]['source_id']==s for i in ids) for s in sources])
        ix=rng.integers(0,len(sources),size=(2000,len(sources)))
        boot=num[ix].sum(1)/den[ix].sum(1)
        lo,hi=np.quantile(boot,[.025,.975])
        stats.append(dict(group=group,n=len(ids),source_clusters=len(sources),selected_minus_warm=float(num.sum()/den.sum()),ci95_low=float(lo),ci95_high=float(hi)))
    csvsave(out/'source_cluster_bootstrap.csv',stats)
    cases=[]
    for p in pred:
        row=byid[p['id']]
        cases.append(dict(arm=p['arm'],id=p['id'],split=p['split'],source_id=p['source_id'],carry=p['carry'],prompt=row['text'],expected=p['expected'],generated=p['text'],correct=p['correct'],weekday_correct=p['weekday_correct'],hour_correct=p['hour_correct'],valid_format=p['valid_format']))
    csvsave(out/'all_test_cases.csv',cases)
    csvsave(out/'validation_curve.csv',[{k:x[k] for k in ['total_updates','additional_updates','n','correct','carry_n','carry_correct','valid_format','answer_token_ce']} for x in curve])
    for name in ['geometry.csv','dataset_counts.csv']:
        (out/name).write_bytes((root/name).read_bytes())
    assert done['active_seconds']<=600 and done['load_seconds']<=300 and done['peak_gib']<=13
    assert done['additional_updates']<=896
    disk=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert disk<=50*2**20
    assessment=dict(completion=done,dataset_and_prediction_checks=True,adapter_checks=adapters,
        pairs=dict(both_correct=paired[(True,True)],warm_only=paired[(True,False)],selected_only=paired[(False,True)],both_wrong=paired[(False,False)]),
        bootstrap=dict(draws=2000,seed=27102703,cluster='source_id (weekday,hour)',caveat='descriptive, conditional on one training seed and fixed generator'),
        actual_disk_bytes=disk,selected_train64=read(root/'selected_train64.json'),source=source)
    save(out/'verification.json',assessment)
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'figure.facecolor':'white'}):
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        xs=[x['total_updates'] for x in curve]
        axes[0].plot(xs,[100*x['correct']/64 for x in curve],color='#0072B2',marker='o')
        axes[0].scatter([chosen['total_updates']],[100*chosen['correct']/64],marker='*',s=150,color='#D55E00',label='Validation-selected')
        axes[0].set(xlabel='Cumulative optimizer updates',ylabel='Full generated answer accuracy (%)',ylim=(0,100),title='Validation: 64 fixed examples')
        axes[0].legend(fontsize=8)
        axes[1].plot(xs,[x['answer_token_ce'] for x in curve],color='#0072B2',marker='s')
        axes[1].set(xlabel='Cumulative optimizer updates',ylabel='Answer-token cross entropy',ylim=(0,max(x['answer_token_ce'] for x in curve)*1.15),title='CE is a secondary selection criterion')
        for ax in axes: ax.grid(alpha=.2);ax.set_xticks(xs);ax.tick_params(axis='x',rotation=45)
        fig.savefig(out/'validation_curve.png',dpi=180,facecolor='white')
        fig.savefig(out/'validation_curve.pdf',facecolor='white')
        plt.close(fig)
    save(out/'figure_provenance.json',dict(source=str(root/'validation_curve.json'),normalization='100 * correct / 64',smoothing=None,
        uncertainty='None: deterministic evaluation of one fixed validation set, not seed-level uncertainty',
        versions=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),export='9 x 3.5 inches; PNG 180dpi and PDF; general research report, no journal compliance claimed',
        alt_text='All validation checkpoints shown without smoothing; star marks the protocol-selected checkpoint; right panel shows answer-token CE.'))
    print(json.dumps(dict(summary=summary,paired=assessment['pairs'],bootstrap=stats,completion=done,traincheck=assessment['selected_train64']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run)
