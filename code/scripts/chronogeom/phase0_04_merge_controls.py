"""Replace only the invalid rescue-noop rows; preserve all original files."""
from pathlib import Path
import json,argparse,hashlib,shutil
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root;original=root/'execution';repair=root/'repair_rescue_noop';out=root/'execution_corrected';out.mkdir(exist_ok=False)
old=[json.loads(s) for s in (original/'interventions.jsonl').read_text().splitlines()];new=[json.loads(s) for s in (repair/'interventions.jsonl').read_text().splitlines()];lookup={(r['layer'],r['pair_id']):r for r in new};assert len(lookup)==492 and all(r['method']=='noop' and r['operation']=='rescue' and r['max_logit_change']==0 for r in new)
corrected=[];diffs=[]
for r in old:
 if r['method']=='noop' and r['operation']=='rescue':
  n=lookup.pop((r['layer'],r['pair_id']));diffs.append(abs(n['baseline_ld']-r['baseline_ld']));assert all(n[k]==r[k] for k in ['baseline_ld','clean_a','corrupt_a','b','baseline_text','baseline_clean_correct','damage_ld']);n['row_source']='repaired_rescue_noop';corrected.append(n)
 else:r['row_source']='original_unmodified';corrected.append(r)
assert not lookup and len(corrected)==22458
(out/'interventions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in corrected));shutil.copy2(original/'config.json',out/'config.json');meta=json.loads((original/'metadata.json').read_text());rm=json.loads((repair/'metadata.json').read_text());meta['noop_controls']=[r for r in meta['noop_controls'] if r['operation']!='rescue']+rm['noop_controls'];meta['original_seconds']=meta['seconds'];meta['repair_seconds']=rm['seconds'];meta['seconds']+=rm['seconds'];meta['subset_batchshape_by_layer']=rm['subset_batchshape_by_layer'];meta['correction']=dict(replaced_rows=492,baseline_ld_max_difference=max(diffs),repair_commit=rm['source_commit'],original_raw_sha256=hashlib.sha256((original/'interventions.jsonl').read_bytes()).hexdigest(),repair_raw_sha256=hashlib.sha256((repair/'interventions.jsonl').read_bytes()).hexdigest(),reason='rescue no-op must reapply damaged activation; no changes to other outcomes or sample selection');(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('Merged only492controls; matched baseline max error',max(diffs))
