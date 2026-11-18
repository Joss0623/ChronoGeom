"""Summarize independent patch runs, keeping quality gates and failed pairs explicit."""
from pathlib import Path
import argparse,json,csv,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 meta=json.loads((args.run/'metadata.json').read_text());rows=json.loads((args.run/'per_pair_results.json').read_text());summary=json.loads((args.run/'summary.json').read_text());pairs=json.loads((args.run/'pair_screening.json').read_text());screen=json.loads((args.run/'eligibility.json').read_text())
 noops=[r for r in rows if r['method']=='noop'];full0=[r for r in rows if r['method']=='full' and r['layer']==0];err=max([r['max_logit_change'] for r in noops],default=None)
 quality={'eligible_pairs':meta['eligible_pairs'],'candidate_pairs':meta['candidate_pairs'],'screened_prompts':len(screen),'correct_prompts':sum(r.get('correct',False) for r in screen),'noop_max_logit_error':err,'noop_pass':err is not None and err<=1e-5,'full0_clean_restoration':float(np.mean([r['restores_clean'] for r in full0])) if full0 else None,'not_plumbing':not meta['config'].get('plumbing_only',False),'all_metrics_finite':all(math.isfinite(r['logit_difference_gain']) for r in rows)}
 quality['full0_pass']=bool(full0) and all(r['restores_clean'] for r in full0);quality['causal_execution_gate']=quality['not_plumbing'] and quality['eligible_pairs']>0 and quality['noop_pass'] and quality['full0_pass'] and quality['all_metrics_finite']
 (args.output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
 methods=list(dict.fromkeys(r['method'] for r in summary));layers=meta['config']['layers'];paired=[]
 bykey={(r['layer'],r['method'],r['pair_id']):r for r in rows};valid_ids=sorted(set(r['pair_id'] for r in rows))
 for l in layers:
  for pid in valid_ids:
   if (l,'helix',pid) in bykey:
    for other in ['pca','random','full']:
     h=bykey[l,'helix',pid];o=bykey[l,other,pid];paired.append(dict(layer=l,pair_id=pid,comparison='helix_minus_'+other,ld_gain_difference=h['logit_difference_gain']-o['logit_difference_gain']))
 (args.output/'paired_differences.json').write_text(json.dumps(paired,indent=2)+'\n')
 with (args.output/'summary.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
 # Examples chosen by candidate order, before inspecting effect size; all data remain in original JSON.
 selected_ids=[p['id'] for p in pairs if p['eligible']][:3];examples=[r for r in rows if r['pair_id'] in selected_ids and r['layer'] in [0,1]]
 (args.output/'first_selected_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
 colors=['#777777','#000000','#D55E00','#0072B2','#009E73'];markers=['x','o','D','s','^']
 with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
  fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
  for method,color,marker in zip(['noop','full','helix','pca','random'],colors,markers):
   ss=[r for r in summary if r['method']==method and r['n']>0]
   axes[0].plot([r['layer'] for r in ss],[r['mean_gain'] for r in ss],marker=marker,color=color,label=method,linestyle='--')
   axes[1].plot([r['layer'] for r in ss],[r['restoration'] for r in ss],marker=marker,color=color,label=method,linestyle='--')
  axes[0].axhline(0,color='.6',linewidth=.6);axes[0].set(ylabel='Mean gain in clean-vs-corrupt logit difference',xlabel='Block-input index');axes[0].legend(fontsize=8)
  axes[1].set(ylabel='Clean-answer restoration fraction',xlabel='Block-input index',ylim=(-.03,1.03))
  for ax in axes:ax.set_xticks(layers);ax.grid(alpha=.2)
  title='Plumbing only: no arithmetic causal conclusion' if meta['config'].get('plumbing_only') else f"Independent GPT-J patching: {meta['eligible_pairs']}/{meta['candidate_pairs']} jointly-correct pairs"
  fig.suptitle(title);fig.savefig(args.output/'primary_controls.png',dpi=180);fig.savefig(args.output/'primary_controls.pdf');plt.close(fig)
 (args.output/'figure_description.md').write_text('Means over the same selected pairs, one checkpoint/seed. No independent-model CI or p-values. Only six preset indices tested; connecting dashes are visual guides. All8conditions saved in summary.csv, all failures and outputs in per_pair_results.json. Correct-pair screening is conditional, not general arithmetic accuracy. Full and noop gate must pass before causal interpretation. Provisional research figure.\n')
 print(json.dumps(quality));print(json.dumps(summary))
if __name__=='__main__':main()
