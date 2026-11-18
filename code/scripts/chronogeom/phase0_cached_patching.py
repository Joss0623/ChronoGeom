"""Replot upstream cached intervention aggregates; no new patching/inference."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True, exist_ok=False)
    data=json.loads(args.cache.read_text())
    rows=[]
    for method,layers in data.items():
        if not isinstance(layers,list) or len(layers)!=28:raise ValueError('Unexpected layer count')
        for layer,values in enumerate(layers):
            if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite metric')
            rows.append(dict(method=method,layer=layer,**values))
    with open(args.output/'all_cached_metrics.csv','x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]), lineterminator="\n");writer.writeheader();writer.writerows(rows)
    selected=[('exchange','Full activation','#000000','-'),
        ('helix_a_[2, 5, 10, 100]','Helix (all periods)','#D55E00','-'),
        ('circle_a_[2, 5, 10, 100]','Fourier / source circle','#0072B2','--'),
        ('pca_9','PCA 9D','#009E73','-.'),('poly_a_9','Polynomial degree 9','#CC79A7',':')]
    summary=[]
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
        for key,label,color,style in selected:
            values=data[key]
            x=np.arange(len(values))
            for ax,metric in zip(axes,['logit_diff','accuracy']):
                ax.plot(x,[v[metric] for v in values],label=label,color=color,linestyle=style,linewidth=1.8)
            entry={'method':key,'mean_logit_diff_over_28_layers':float(np.mean([v['logit_diff'] for v in values])),
                'mean_patched_accuracy_over_28_layers':float(np.mean([v['accuracy'] for v in values]))}
            summary.append(entry);print(json.dumps(entry),flush=True)
        axes[0].set(ylabel='Cached target-logit gain',xlabel='Upstream cached layer index')
        axes[1].set(ylabel='Cached patched-target accuracy',xlabel='Upstream cached layer index',ylim=(-.02,1.02))
        axes[0].legend(frameon=False,fontsize=7)
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle('GPT-J: author-cached intervention aggregates (no new model run)',fontsize=11)
        fig.savefig(args.output/'cached_interventions.png',dpi=180,facecolor='white')
        fig.savefig(args.output/'cached_interventions.pdf',facecolor='white')
        plt.close(fig)
    metadata={'evidence':'reanalysis_of_author_cached_intervention_aggregates',
        'source_cache_sha256':hashlib.sha256(args.cache.read_bytes()).hexdigest(),'n_methods':len(data),'n_layers':28,
        'summary':summary,'uncertainty':'Not computed: per-example raw outputs and paired sample identifiers unavailable.',
        'interpretation':'Layer means are descriptive, not independent experimental replicates. This does not reproduce an intervention execution or establish a new causal result.',
        'raw_metrics':'all_cached_metrics.csv','figure_alt':'Two panels show cached target-logit gain and patched-target accuracy across all 28 layers for full activation, helix, Fourier, PCA and polynomial patches.'}
    (args.output/'cached_patching_summary.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
