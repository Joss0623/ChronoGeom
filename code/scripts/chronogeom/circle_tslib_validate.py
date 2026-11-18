"""Replay, integrity and descriptive waveform diagnostics; no retraining."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from models.Transformer import Model
from data_provider.chronogeom_circle_pilot import make_pack,args_for,H
from scripts.chronogeom.tslib_circle_transfer import capture_site,forward_with_patch

def main(root):
    cfg=json.loads((root/"config.json").read_text())
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    replay={}
    for s in ["train","val","fit","test"]:
        z=np.load(root/"data"/(s+".npz"));p=make_pack(s,cfg["n_"+s])
        replay[s]=all(np.array_equal(z[k],v) for k,v in p.items())
    for shift in cfg["shifts"]:
        z=np.load(root/"data"/f"test_shift_{shift}.npz");p=make_pack("test",cfg["n_test"],shift)
        replay["shift_"+str(shift)]=all(np.array_equal(z[k],v) for k,v in p.items())
    assert all(replay.values())
    checks=[]
    clean=make_pack("test",cfg["n_test"])
    for seed in cfg["seeds"]:
        model=Model(SimpleNamespace(**cfg)).cuda().eval()
        state=torch.load(root/"states"/f"best_{seed}.pt",map_location="cpu",weights_only=True);model.load_state_dict(state["state"])
        for si in [0,3]:
            site=cfg["sites"][si];q=np.load(root/"analysis/bases"/f"trained_{seed}_{si}.npz")["circle_basis"]
            cp,ca=capture_site(model,site,args_for(clean,"cuda"))
            for shift in cfg["shifts"]:
                bad=make_pack("test",cfg["n_test"],shift);bp,ba=capture_site(model,site,args_for(bad,"cuda"))
                donor=ba.clone();donor[:,-H:]=ca[:,-H:]
                # Use the direct orthogonal-subspace API independently of the saved patched activations.
                pred=forward_with_patch(model,site,args_for(bad,"cuda"),donor,q,mode="subspace").cpu().numpy()
                old=np.load(root/"analysis/predictions"/f"{seed}_{si}_{shift}_circle.npz")
                err=float(np.max(np.abs(pred-old["prediction"])))
                # Original scientific run used CPU float64 projection then FP32 casting.
                b=ba[:,-H:].cpu().numpy();d=ca[:,-H:].cpu().numpy()
                changed=(b+((d-b)@q)@q.T).astype(b.dtype)
                original_path=ba.clone();original_path[:,-H:]=torch.as_tensor(changed,device="cuda")
                exact=forward_with_patch(model,site,args_for(bad,"cuda"),original_path,mode="full").cpu().numpy()
                exacterr=float(np.max(np.abs(exact-old["prediction"])))
                assert exacterr==0.,exacterr
                # Measure the alternate FP32 coordinate path; do not replace raw scientific outputs.
                with torch.inference_mode():
                    qt=torch.as_tensor(q,dtype=ba.dtype,device=ba.device)
                    ft=(ba[:,-H:]+((ca[:,-H:]-ba[:,-H:])@qt)@qt.T).cpu().numpy()
                alternate_mse=float(np.mean((pred-old["prediction"])**2))
                checks.append(dict(seed=seed,site=site,shift=shift,
                    original_fp64_numpy_replay_max_abs=exacterr,
                    alternate_fp32_subspace_replay_max_abs=err,
                    alternate_fp32_prediction_difference_mse=alternate_mse,
                    alternate_projection_activation_max_abs=float(np.max(np.abs(ft-changed)))))
    original=subprocess.run(["git","diff","--exit-code","origin/main","--","models/Transformer.py","layers/Embed.py"],capture_output=True)
    assert original.returncode==0,original.stdout.decode()
    df=pd.read_csv(root/"analysis/interventions.csv")
    match=df[df.condition.str.contains("_norm")]
    wrong=df[(df.condition=="circle")&(df.phase_change*(-df["shift"])<0)]
    validation=dict(data_exact_replay=replay,checkpoint_replays=checks,public_tslib_model_unchanged=True,
        total_rows=len(df),finite_primary_metrics=bool(np.isfinite(df[["clean_truth_mse","clean_pred_mse","delta_norm"]].to_numpy()).all()),
        norm_matching_max_relative_error=float(match.norm_match_relative_error.max()),
        norm_matching_failure_count=int((match.norm_match_relative_error>.01).sum()),
        wrong_circle_phase_cases=wrong[["seed","site","shift","trajectory","phase_change"]].to_dict("records"),
        phase_undefined_all_conditions=int((~df.phase_defined).sum()),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        reporting_repairs=["An additional audit initially required alternate FP32 projection to match the original FP64-then-cast path within 2e-5; it failed. Original-path replay is bitwise exact; alternate path differences are measured per case, not hidden or used to change results.","summary_v2 excludes phase-undefined outputs as preregistered; original summary retained. No predictions/bases/data changed.",
        "One interactive pandas query lost quoted string through shell quoting; corrected with direct column filter. No scientific run affected."])
    # Post-hoc diagnostic prompted by visible waveform roughness, not used for model/layer selection.
    j=np.arange(1,25);z=np.column_stack([np.ones(24),j,np.sin(2*np.pi*j/24),np.cos(2*np.pi*j/24)]);pinv=np.linalg.pinv(z)
    wave=[]
    for f in sorted((root/"analysis/predictions").glob("*.npz")):
        parts=f.stem.split("_",3);seed,site,shift=int(parts[0]),int(parts[1]),int(parts[2]);condition=parts[3]
        a=np.load(f);p=a["prediction"][...,0];c=a["clean_prediction"][...,0]
        co=p@pinv.T;cc=c@pinv.T;res=p-co@z.T
        for i,ident in enumerate(a["ids"]):
            wave.append(dict(seed=seed,site=cfg["sites"][site],shift=shift,condition=condition,trajectory=int(ident),
                nonharmonic_mse=float(np.mean(res[i]**2)),clean_nonharmonic_mse=float(np.mean((c[i]-cc[i]@z.T)**2)),
                trend_error_vs_clean=float(abs(co[i,1]-cc[i,1])),offset_error_vs_clean=float(abs(co[i,0]-cc[i,0]))))
    pd.DataFrame(wave).to_csv(root/"summary_v2/waveform_diagnostic_posthoc.csv",index=False)
    w=pd.DataFrame(wave).groupby(["site","condition"])[["nonharmonic_mse","clean_nonharmonic_mse","trend_error_vs_clean","offset_error_vs_clean"]].mean().reset_index()
    w.to_csv(root/"summary_v2/waveform_diagnostic_summary_posthoc.csv",index=False)
    (root/"validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in validation.items() if k!="checkpoint_replays"},ensure_ascii=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);main(Path(p.parse_args().run))
