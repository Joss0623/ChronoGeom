"""Summarize all preregistered Circle pilot results without selecting a best layer."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def main(root,folder):
    a=root/"analysis";out=root/folder;out.mkdir(exist_ok=False);figs=out/"figures";figs.mkdir()
    df=pd.read_csv(a/"interventions.csv");g=pd.read_csv(a/"geometry.csv")
    rng=np.random.default_rng(1729);ids=np.sort(df.trajectory.unique());boot=rng.integers(len(ids),size=(1000,len(ids)))
    np.savez_compressed(out/"bootstrap_indices.npz",indices=boot,trajectory_ids=ids)
    rows=[];contrast=[]
    for (site,cond),v in df.groupby(["site","condition"],sort=False):
        t=v.groupby("trajectory",sort=True).mean(numeric_only=True)
        t=t.reindex(ids);err=t.clean_pred_mse.to_numpy();den=t.base_clean_pred_mse.to_numpy()
        samples=1-err[boot].mean(1)/den[boot].mean(1)
        recovery=1-err.mean()/den.mean()
        seedmeans=v.groupby("seed")[["clean_pred_mse","base_clean_pred_mse"]].mean()
        perseed=1-seedmeans.clean_pred_mse/seedmeans.base_clean_pred_mse
        # A phase change should reverse the known clean-to-corrupt phase shift.
        defined=v[v.phase_defined]
        directional=(defined.phase_change/(-defined["shift"]*2*np.pi/24)).mean()
        rows.append(dict(site=site,condition=cond,trajectories=len(ids),models=v.seed.nunique(),shifts=v["shift"].nunique(),records=len(v),
            recovery=recovery,recovery_low=np.quantile(samples,.025),recovery_high=np.quantile(samples,.975),
            recovery_model_sd=perseed.std(),clean_truth_mse=v.clean_truth_mse.mean(),
            corrupt_truth_mse=v.corrupt_truth_mse.mean(),delta_norm=v.delta_norm.mean(),
            relative_delta=v.relative_delta.mean(),mahal=v.mahal.mean(),phase_error_degrees=np.degrees(defined.phase_error.mean()),
            directional_fraction=directional,amplitude=v.amplitude.mean(),phase_undefined=int((~v.phase_defined).sum()),
            norm_match_failures=int((v.norm_match_relative_error>.01).sum()),max_scale=v.scale.max()))
        if cond=="circle":
            for other in ["pca","random0","random1","random2","shuffled","pca_norm","random0_norm","shuffled_norm"]:
                z=df[(df.site==site)&(df.condition==other)].groupby("trajectory").mean(numeric_only=True).reindex(ids)
                difference=z.clean_pred_mse.to_numpy()-err
                bs=difference[boot].mean(1)/den[boot].mean(1)
                contrast.append(dict(site=site,contrast="circle_minus_"+other,recovery_difference=difference.mean()/den.mean(),
                    ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975),trajectories=len(ids),bootstrap_unit="source trajectory; both shifts and fixed three models averaged"))
    summary=pd.DataFrame(rows);summary.to_csv(out/"intervention_summary.csv",index=False)
    pd.DataFrame(contrast).to_csv(out/"paired_contrasts.csv",index=False)
    g.groupby(["stage","site","cohort"]).agg(
        circle_r2=("circle_augmented_r2","mean"),incremental_r2=("incremental_r2","mean"),
        phase_decode_r2=("phase_decode_r2","mean"),axis_ratio=("axis_ratio","mean"),pca_overlap=("pca_overlap","mean"),
        model_sd_r2=("circle_augmented_r2","std"),rank_min=("effective_rank","min"),rank_max=("effective_rank","max")
    ).reset_index().to_csv(out/"geometry_summary.csv",index=False)
    noops=df[df.condition=="noop"].groupby(["seed","site","shift","trajectory"]).corrupt_truth_mse
    harm=[]
    for (site,cond),v in df[df.condition.str.contains("ablate")].groupby(["site","condition"],sort=False):
        t=v.groupby("trajectory").mean(numeric_only=True).reindex(ids)
        base=df[(df.site==site)&(df.condition=="noop")].groupby("trajectory").mean(numeric_only=True).reindex(ids)
        d=(t.corrupt_truth_mse-base.corrupt_truth_mse).to_numpy();bs=d[boot].mean(1)
        harm.append(dict(site=site,condition=cond,mse_increase=d.mean(),ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975)))
    pd.DataFrame(harm).to_csv(out/"ablation_summary.csv",index=False)
    # Fixed-order raw examples, including all failures rather than cherry-picked cases.
    fail=df[(df.condition=="circle")].copy()
    fail["recovery"]=1-fail.clean_pred_mse/fail.base_clean_pred_mse
    fail["wrong_phase_direction"]=fail.phase_change*(-fail["shift"])<0
    fail.to_csv(out/"all_circle_cases.csv",index=False)
    counts=fail.groupby(["site"]).agg(records=("trajectory","size"),negative_recovery=("recovery",lambda x:int((x<0).sum())),wrong_phase_direction=("wrong_phase_direction","sum")).reset_index()
    counts.to_csv(out/"circle_failure_counts.csv",index=False)
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    sites=list(df.site.drop_duplicates())
    fig,axs=plt.subplots(2,4,figsize=(12,6),layout="constrained")
    for k,site in enumerate(sites):
        coords=[]
        for r,stage in enumerate(["init","trained"]):
            z=np.load(a/f"geometry_{stage}_0_{k}.npz");p=z["projection"];hour=z["hour"]
            mean=np.stack([p[hour==h].mean(0) for h in range(24)]);coords.append(mean)
            ax=axs[r,k];ax.plot(np.r_[mean[:,0],mean[0,0]],np.r_[mean[:,1],mean[0,1]],color="#444444",lw=.7)
            ax.scatter(mean[:,0],mean[:,1],c=np.arange(24),cmap="twilight",s=18)
            for h in [0,6,12,18]:ax.annotate(str(h),mean[h],fontsize=8)
            ax.set_title(f"{stage}, seed 0\n{site.replace('dec_embedding.temporal_embedding.','')}",fontsize=8)
            ax.set_xlabel("Orthogonal phase axis 1");ax.set_ylabel("Orthogonal phase axis 2");ax.set_aspect("equal",adjustable="box")
        lim=np.max(np.abs(np.vstack(coords)))*1.2
        for r in range(2):axs[r,k].set(xlim=(-lim,lim),ylim=(-lim,lim))
    fig.savefig(figs/"geometry.png",dpi=180);fig.savefig(figs/"geometry.pdf");plt.close(fig)
    conditions=["noop","full_donor","circle","pca","random0","random1","random2","shuffled","pca_norm","random0_norm","shuffled_norm"]
    fig,axs=plt.subplots(1,4,figsize=(12,4.8),sharey=True,sharex=True,layout="constrained")
    for ax,site in zip(axs,sites):
        z=summary[summary.site==site].set_index("condition").loc[conditions]
        v=z.recovery.to_numpy();lo=z.recovery_low.to_numpy();hi=z.recovery_high.to_numpy()
        ax.errorbar(v,np.arange(len(z)),xerr=np.maximum(0,np.stack([v-lo,hi-v])),fmt="o",color="#0072B2",ms=4,capsize=2)
        ax.axvline(0,color="gray",lw=.7);ax.axvline(1,color="gray",ls=":",lw=.7)
        ax.set_yticks(np.arange(len(z)),conditions);ax.set_xlabel("Recovery toward clean forecast");ax.set_title(site.replace("dec_embedding.temporal_embedding.",""),fontsize=8)
    axs[0].invert_yaxis()
    fig.savefig(figs/"interventions.png",dpi=180);fig.savefig(figs/"interventions.pdf");plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout="constrained")
    styles={"noop":("#777777","--"),"full_donor":("#CC79A7",":"),"circle":("#0072B2","-"),"random0_norm":("#D55E00","-.")}
    for ax,k in zip(axs,[0,3]):
        for name,(color,ls) in styles.items():
            z=np.load(a/"predictions"/f"0_{k}_6_{name}.npz")
            ax.plot(np.arange(1,25),z["prediction"][0,:,0],color=color,ls=ls,label=name)
        ax.plot(np.arange(1,25),z["clean_truth"][0,:,0],color="black",lw=1,label="clean mean")
        ax.set(title=f"Fixed case: seed 0, trajectory 300000\n{sites[k]}",xlabel="Forecast hour",ylabel="Value")
    axs[1].legend(fontsize=8)
    fig.savefig(figs/"fixed_example.png",dpi=180);fig.savefig(figs/"fixed_example.pdf");plt.close(fig)
    dump(out/"figure_provenance.json",dict(purpose="provisional research report, no publisher compliance claim",data=["analysis/geometry*.npz","analysis/interventions.csv","analysis/predictions/*.npz"],
        geometry="hour means, raw orthogonal phase-plane coordinates; seed 0 fixed in code, no whitening; common init/trained limits within each site",
        uncertainty="95% percentile trajectory bootstrap; 1000 draws seed1729, fixed 3 models and both shifts averaged",
        example="first test trajectory, seed0, +6h, predeclared endpoint sites; not selected by outcome",
        matplotlib=matplotlib.__version__,figures=[{"file":p.name,"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(figs.iterdir())]))
    print(summary[summary.condition.isin(["circle","pca","random0_norm","shuffled","noop","full_donor"])][["site","condition","recovery","recovery_low","recovery_high","phase_error_degrees"]].to_string(index=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--folder",default="summary");a=p.parse_args();main(Path(a.run),a.folder)
