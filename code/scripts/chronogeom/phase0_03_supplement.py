"""CPU-only rank-sensitivity intervals and optional standalone numerical diagnostics."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.chronogeom.phase0_03_geometry import identity_weights,interval,fit_spaces,subspace_error
from scripts.chronogeom.representation_probe import basis,metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--standalone',type=Path);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((args.run/'config.json').read_text());screen=json.loads((args.run/'screening.json').read_text());df=pd.read_json(args.run/'interventions.jsonl',lines=True);pairs=[p for p in screen['cohorts']['fixed'] if p['selected']];ids=[p['id'] for p in pairs];weights=identity_weights(pairs,2000,1729);rows=[]
    for rank in [1,8]:
        for l in cfg['rank_sensitivity_layers']:
            for op in ['swap','ablate_mean']:
                def choose(method):return df[(df.cohort=='fixed')&(df.layer==l)&(df.site=='a')&(df.method==method)&(df['rank']==rank)&(df.operation==op)&(df['mode']=='raw')&(df.control_seed==42)]
                h=choose('helix')
                if len(h)!=len(ids):continue
                h=h.set_index('pair_id').loc[ids]
                for method in ['pca','fourier','linear','polynomial','random','shuffled']:
                    b=choose(method)
                    if len(b)!=len(ids):continue
                    b=b.set_index('pair_id').loc[ids];ci=interval((h.effect-b.effect).to_numpy(),weights)
                    rows.append(dict(rank=rank,layer=l,operation=op,comparison='helix_minus_'+method,n=len(ids),bootstrap='shared_nodes_exploratory',**ci))
    pd.DataFrame(rows).to_csv(out/'rank_sensitivity_intervals.csv',index=False,lineterminator='\n')
    path=args.standalone or args.run/'standalone_0_360.npz'
    if not path.exists():
        (out/'fft_status.json').write_text(json.dumps({'executed':False,'reason':'no optional standalone cache; see main budget status'})+'\n');return
    d=np.load(path);x=d['values'];h=d['hidden'];lengths=d['token_lengths'];freq=np.fft.rfftfreq(len(x));spectra=[];geometry=[];pcrows=[]
    for li,l in enumerate([0,1]):
        y=h[:,li].astype(float);linear=basis(x,'linear',scale=360);detrended=y-linear@np.linalg.lstsq(linear,y,rcond=1e-10)[0]
        for label,yy in [('raw',y),('linear_detrended',detrended)]:
            power=np.mean(np.abs(np.fft.rfft(yy,axis=0))**2,axis=1)
            for f,p in zip(freq,power):spectra.append(dict(layer=l,condition=label,frequency=float(f),period=None if f==0 else float(1/f),power=float(p)))
        small=y[:100];sd=small.std(0);standardized=(small-small.mean(0))/np.where(sd>0,sd,1);u,s,_=np.linalg.svd(standardized,full_matrices=False);pc1=u[:,0]*s[0];pcrows.append(dict(layer=l,n=100,pc1_a_correlation_r2=float(np.corrcoef(np.arange(100),pc1)[0,1]**2),standardization='per hidden dimension, full standalone0..99; descriptive source-style PC1, not heldout reconstruction'))
        train_ids=cfg['fit_values'];train=np.isin(np.arange(100),train_ids);test=~train;spaces,info=fit_spaces(np.arange(100)[train],small[train])
        for method in ['helix','fourier','linear','polynomial','shuffled']:
            kind='helix' if method=='shuffled' else method;design=basis(np.arange(100),kind);pred=design@info[method]['coef'];row=dict(layer=l,method=method,n_train=int(train.sum()),n_test=int(test.sum()),native_rank=len(spaces[method]));row.update(metrics(small[test],pred[test]));geometry.append(row)
    pd.DataFrame(spectra).to_csv(out/'standalone_spectra.csv',index=False,lineterminator='\n');pd.DataFrame(geometry).to_csv(out/'standalone_numeric_fits.csv',index=False,lineterminator='\n');(out/'standalone_pc1.json').write_text(json.dumps(pcrows,indent=2)+'\n');(out/'fft_status.json').write_text(json.dumps({'executed':True,'n':361,'all_single_token':bool(np.all(lengths==1)),'token_length_counts':{str(n):int(np.sum(lengths==n)) for n in np.unique(lengths)},'grid':'all0..360; no interpolation or sample removal','heldout': '79 original fit IDs vs21 others including missing cache identity45','interpretation':'discovery diagnostic only; decimal integer frequencies are not physical time cycles'},indent=2)+'\n')
    spectra=pd.DataFrame(spectra)
    with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
        for ax,l in zip(axes,[0,1]):
            for label,color,style in [('raw','#0072B2','-'),('linear_detrended','#D55E00','--')]:
                part=spectra[(spectra.layer==l)&(spectra.condition==label)&(spectra.frequency>0)];ax.plot(part.frequency,part.power,label=label,color=color,linestyle=style)
            ax.set(title=f'Standalone numbers, block-input {l}',xlabel='Cycles per integer',ylabel='Mean |FFT(hidden)|²',yscale='log');ax.grid(alpha=.2)
            for t in [2,5,10,100]:ax.axvline(1/t,color='.6',linewidth=.6,linestyle=':')
        axes[0].legend(fontsize=8);fig.suptitle('All361 consecutive integers; periodicity diagnostic, no causal claim');fig.savefig(out/'standalone_fft.png',dpi=180);fig.savefig(out/'standalone_fft.pdf');plt.close(fig)
    print('Standalone PC1',json.dumps(pcrows),flush=True)
if __name__=='__main__':main()
