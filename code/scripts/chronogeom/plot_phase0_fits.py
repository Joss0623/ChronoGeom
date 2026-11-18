"""Plot all predeclared cached-activation probe results; general research draft."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    for name in ['heldout_fits.png', 'heldout_fits.pdf', 'figure_description.txt']:
        if (args.input/name).exists():raise FileExistsError('Refuse figure overwrite: '+str(args.input/name))
    summary=json.loads((args.input/'fit_summary.json').read_text())
    numeric=['linear','circle_T100','fourier','helix','polynomial8','shuffled_helix']
    labels=['Linear','Circle\nT=100','Fourier','Helix','Poly 8','Shuffled\nHelix']
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
        for row,layer in enumerate([0,1]):
            ax=axes[row,0]
            entries=[next(e for e in summary if e['layer']==layer and e['method']==m) for m in numeric]
            ax.errorbar(np.arange(len(entries)),[e['r2_mean'] for e in entries],
                yerr=[e['r2_sd'] for e in entries],fmt='o',color='#0072B2',capsize=3)
            ax.axhline(0,color='gray',linewidth=.6);ax.set_xticks(np.arange(len(entries)),labels)
            ax.set(title=f'Cached layer {layer}: held-out numeric identities',ylabel='Activation reconstruction R² (mean ± SD)')
            ax.grid(axis='y',alpha=.2)
            data=np.load(args.input/f'preview_layer{layer}.npz')
            test=~data['train'];xy=data['pca2'];pred=data['helix_pca2'];ax=axes[row,1]
            scatter=ax.scatter(xy[test,0],xy[test,1],c=data['a'][test],cmap='viridis',marker='^',s=20,label='Observed held-out')
            ax.scatter(pred[test,0],pred[test,1],facecolors='none',edgecolors='black',marker='o',s=24,label='Helix prediction')
            ax.set(title=f'Layer {layer}: train-only PCA display (seed 0)',xlabel='PC1',ylabel='PC2')
            ax.legend(fontsize=7,frameon=False);fig.colorbar(scatter,ax=ax,label='Operand a')
        fig.suptitle('Official cached GPT-J activations: numeric regression, five predeclared splits',fontsize=11)
        fig.savefig(args.input/'heldout_fits.png',dpi=180,facecolor='white')
        fig.savefig(args.input/'heldout_fits.pdf',facecolor='white');plt.close(fig)
    (args.input/'figure_description.txt').write_text('Left: all numeric predictor conditions, mean and sample SD across five overlapping identity splits (not five independent pretrained models). Right: observed and predicted held-out activations projected using train-only PCA. PCA reconstruction baselines are in the full table and excluded from numeric-prediction panels because they observe test activations. No points excluded or axes clipped. Draft general figure; no journal-specific formatting claim.\n')


if __name__=='__main__':main()
