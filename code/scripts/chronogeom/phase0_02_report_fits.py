"""Descriptive all-layer summaries and plots; all scores retained, no layer selection."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 rows=list(csv.DictReader((args.input/'fits.csv').open()));summary=[]
 for protocol in sorted(set(r['protocol'] for r in rows)):
  for layer in range(28):
   for method in sorted(set(r['method'] for r in rows)):
    selected=[r for r in rows if r['protocol']==protocol and int(r['layer'])==layer and r['method']==method and r['group']=='heldout']
    vals=np.array([float(r['r2']) for r in selected]);ids=np.array([float(r['identity_r2']) for r in selected])
    summary.append(dict(protocol=protocol,layer=layer,method=method,n=len(vals),r2_mean=float(vals.mean()),r2_sd=float(vals.std(ddof=1)) if len(vals)>1 else None,identity_r2_mean=float(ids.mean()),effective_rank=int(selected[0]['effective_rank'])))
 (args.output/'all_layer_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (args.output/'all_layer_summary.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
 methods=['linear','circle100','fourier','helix','poly8_scaled','shuffled_helix'];colors=['#0072B2','#CC79A7','#009E73','#D55E00','#000000','#777777'];styles=['-','--','-.','-',':','--'];markers=['o','s','^','D','v','x']
 with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
  fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
  for ax,protocol,title in zip(axes,['observed80_direct','official75_globalPCA100','range80_direct'],['80% identity / direct hidden','75% full-grid / global PCA100','Range holdout: train <80, test >=80']):
   for method,color,style,marker in zip(methods,colors,styles,markers):
    rs=[r for r in summary if r['protocol']==protocol and r['method']==method];x=[r['layer'] for r in rs];y=np.array([r['r2_mean'] for r in rs]);ax.plot(x,y,label=method,color=color,linestyle=style,marker=marker,markevery=4,markersize=3)
    if rs[0]['r2_sd'] is not None:
     sd=np.array([r['r2_sd'] for r in rs]);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.08)
   ax.axhline(0,color='.5',linewidth=.7);ax.set(xlabel='Cache block-input index',ylabel='Held-out full-hidden R²',title=title);ax.grid(alpha=.2)
  axes[2].set_yscale('symlog',linthresh=1);axes[2].set_ylim(bottom=1.2*min(r['r2_mean'] for r in summary if r['protocol']=='range80_direct' and r['method'] in methods),top=.2);axes[2].set_ylabel('Held-out R² (symlog; linear within ±1)')
  axes[0].legend(fontsize=7);fig.suptitle('Official cached GPT-J: partial replication; bands = SD over 5 overlapping splits; range split n=1')
  fig.savefig(args.output/'all_layers.png',dpi=180);fig.savefig(args.output/'all_layers.pdf');plt.close(fig)
  spec=list(csv.DictReader((args.input/'fourier_partial.csv').open()));fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
  for ax,l in zip(axes,[0,1]):
   rs=[r for r in spec if int(r['layer'])==l];ax.plot([float(r['frequency']) for r in rs],[float(r['partial_r2']) for r in rs],color='#0072B2',marker='.',markersize=3);ax.set(title=f'Cache index {l}',xlabel='Cycles per integer',ylabel='Partial R² after removing linear trend');ax.grid(alpha=.2)
   for period in [2,5,10,100]:ax.axvline(1/period,color='.6',linestyle=':',linewidth=.5)
  fig.suptitle('99 observed identity means; frequency scan, not 0–360 standalone-number FFT')
  fig.savefig(args.output/'frequency_scan.png',dpi=180);fig.savefig(args.output/'frequency_scan.pdf');plt.close(fig)
 (args.output/'figure_description.md').write_text('All28layers shown. First two panels: means±sample SD across5overlapping identity splits, not independent-model uncertainty. Thirdpanel: single predefined range split. No score clipping; range panel uses symlog with linear threshold1 to retain extreme polynomial failures and visible smaller failures. Frequency plot uses partial regression on irregular99-value grid after linear detrending; missing45 not interpolated. Raw all-method tables include raw/scaledpoly9 and all subgroup outcomes. Provisional research figures, no venue-format claim.\n')
 print('rows',len(rows),'summary',len(summary))
if __name__=='__main__':main()
