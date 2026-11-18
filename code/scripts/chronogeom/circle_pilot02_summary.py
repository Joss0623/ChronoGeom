"""Pilot02 full summaries and source-trajectory statistics; no outcome selection."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.chronogeom.circle_pilot02_math import centroids,order_metrics
from scripts.chronogeom.circle_pilot02 import metrics

def main(root):
    out=root/"summary";out.mkdir();figs=out/"figures";figs.mkdir()
    cfg=json.loads((root/"config.json").read_text());sites=cfg["sites"]
    d=pd.read_csv(root/"interventions.csv");ng=pd.read_csv(root/"native_geometry.csv")
    cal=pd.read_csv(root/"calibration.csv");sc=pd.read_csv(root/"test_space_scores.csv")
    ref=sc[sc.space.eq("circle")][["seed","site","test_phase_r2","test_capture"]].rename(columns={"test_phase_r2":"target_test_phase_r2","test_capture":"target_test_capture"})
    match=cal.merge(sc,on=["seed","site","space"]).merge(ref,on=["seed","site"])
    match["test_phase_match"]=(match.test_phase_r2-match.target_test_phase_r2).abs()<=cfg["phase_tolerance"]
    match["test_reconstruction_relative_error"]=(match.test_capture-match.target_test_capture).abs()/match.target_test_capture
    match["test_reconstruction_match"]=match.test_reconstruction_relative_error<=cfg["reconstruction_tolerance"]
    match["strict_joint_match"]=match.phase_match&match.reconstruction_match&match.test_phase_match&match.test_reconstruction_match
    match.to_csv(out/"matching_diagnostics.csv",index=False)
    ids=np.sort(d.trajectory.unique());coh=d.drop_duplicates("trajectory").set_index("trajectory").reindex(ids).cohort.to_numpy()
    rng=np.random.default_rng(1729);boot=np.concatenate([rng.choice(np.where(coh==c)[0],(cfg["bootstrap"],sum(coh==c)),replace=True) for c in np.unique(coh)],axis=1)
    np.savez_compressed(out/"bootstrap.npz",indices=boot,ids=ids,cohort=coh)
    rows=[];cohortrows=[];contrast=[]
    means={}
    for (site,condition),v in d.groupby(["site","condition"],sort=False):
        t=v.groupby("trajectory").mean(numeric_only=True).reindex(ids);means[(site,condition)]=t
        err=t.clean_pred_mse.to_numpy();den=t.base_clean_pred_mse.to_numpy()
        rec=1-err.mean()/den.mean();bs=1-err[boot].mean(1)/den[boot].mean(1)
        defined=v[v.phase_defined];sm=v.groupby("seed")[["clean_pred_mse","base_clean_pred_mse"]].mean()
        row=dict(site=site,condition=condition,n_trajectories=len(ids),models=3,records=len(v),recovery=rec,
            low=np.quantile(bs,.025),high=np.quantile(bs,.975),model_sd=(1-sm.clean_pred_mse/sm.base_clean_pred_mse).std(),
            phase_error_deg=np.degrees(defined.phase_error.mean()),true_phase_error_deg=np.degrees(defined.phase_error_true.mean()),
            phase_undefined=int((~v.phase_defined).sum()))
        for k in ["amplitude","amplitude_error","offset_error","trend_error","offset_vs_clean","trend_vs_clean","nonharmonic_mse",
            "clean_truth_mse","corrupt_truth_mse","delta_norm","relative_delta","nn_state_distance","target_centroid_distance","mahal","outside_centroid_support"]:
            row[k]=v[k].mean()
        row["norm_mismatch_rows"]=int((v.norm_match_error>.01).sum());rows.append(row)
        for c,part in v.groupby("cohort"):
            cohrow=dict(site=site,condition=condition,cohort=int(c),n=part.trajectory.nunique(),
                recovery=1-part.clean_pred_mse.mean()/part.base_clean_pred_mse.mean(),
                phase_error_deg=np.degrees(part[part.phase_defined].phase_error.mean()),
                nonharmonic_mse=part.nonharmonic_mse.mean(),amplitude_error=part.amplitude_error.mean())
            cohortrows.append(cohrow)
    s=pd.DataFrame(rows);s.to_csv(out/"intervention_summary.csv",index=False)
    pd.DataFrame(cohortrows).to_csv(out/"cohort_repeats.csv",index=False)
    for site in sites:
        c=means[(site,"circle")];den=c.base_clean_pred_mse.to_numpy()
        for other in ["residual_pca","cv_phase","cv_phase_norm","recon_match_norm","info_match_norm","joint_match_norm",
                      "conditional_full","conditional_circle","harmonic_mean_delta"]:
            z=means[(site,other)];diff=z.clean_pred_mse.to_numpy()-c.clean_pred_mse.to_numpy()
            bs=diff[boot].mean(1)/den[boot].mean(1)
            contrast.append(dict(site=site,contrast="circle_minus_"+other,recovery_difference=diff.mean()/den.mean(),
                low=np.quantile(bs,.025),high=np.quantile(bs,.975)))
    pd.DataFrame(contrast).to_csv(out/"paired_contrasts.csv",index=False)
    # Native geometry repeats on the two independent cohorts, no supervised projection.
    data=np.load(root/"data/test.npz");h=(data["t"]%24).ravel();nc=len(data["x"]);nr=[]
    for seed in cfg["seeds"]:
        for stage in ["init","trained"]:
            cache=np.load(root/"cache"/f"{seed}_{stage}_baseline_test.npz")
            for si,site in enumerate(sites):
                source=np.load(root/"native"/f"{seed}_{stage}_baseline_{si}.npz")
                for c in [2,3]:
                    mask=data["cohort"]==c;x=cache[f"site{si}"][mask].reshape(-1,16)
                    cc=centroids(x,h.reshape(nc,24)[mask].ravel())
                    m,_,_,_=order_metrics(cc,source["pca_basis"],source["center"])
                    nr.append(dict(seed=seed,stage=stage,site=site,cohort=c,n=int(mask.sum()),**m))
    pd.DataFrame(nr).to_csv(out/"native_cohort_repeats.csv",index=False)
    fields=["pca2_fraction","cyclic_distance_rho","adjacent_ratio","nearest_two_hit","angular_consistency","winding","radius_cv","harmonic1_fraction","native_circle_gate","supervised_phase_r2"]
    ng.groupby(["stage","mode","site"])[fields].mean().reset_index().to_csv(out/"native_summary.csv",index=False)
    pd.read_csv(root/"input_forecast.csv").groupby(["stage","mode"])[["mse","clean_mse"]].agg(["mean","std"]).to_csv(out/"input_summary.csv")
    # Input conflict: orientation relative to unchanged numeric-history forecast.
    zz=np.column_stack([np.ones(24),np.arange(1,25),np.sin(2*np.pi*np.arange(1,25)/24),np.cos(2*np.pi*np.arange(1,25)/24)])
    pi=np.linalg.pinv(zz);ir=[]
    for seed in cfg["seeds"]:
        base=np.load(root/"cache"/f"{seed}_trained_baseline_test.npz")["prediction"][...,0]
        co=base@pi.T;conf=np.load(root/"cache"/f"{seed}_calendar_conflict.npz")["prediction"][...,0]@pi.T
        change=np.angle(np.exp(1j*(np.arctan2(conf[:,3],conf[:,2])-np.arctan2(co[:,3],co[:,2]))))
        ir.append(dict(seed=seed,calendar_follow_fraction=float(np.mean(change)/(np.pi/2)),phase_change_degrees=float(np.degrees(change).mean())))
    pd.DataFrame(ir).to_csv(out/"calendar_follow.csv",index=False)
    fail=d[d.condition.eq("circle")].copy()
    fail["wrong_phase_direction"]=fail.phase_change*(-fail["shift"])<0
    fail["negative_recovery"]=fail.clean_pred_mse>fail.base_clean_pred_mse
    fail.to_csv(out/"all_circle_cases.csv",index=False)
    fail.groupby("site")[["wrong_phase_direction","negative_recovery"]].sum().to_csv(out/"failure_counts.csv")
    plt.rcParams.update({"font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(2,4,figsize=(12,6),layout="constrained")
    for si,site in enumerate(sites):
        coords=[]
        for row,stage in enumerate(["init","trained"]):
            a=np.load(root/"native"/f"0_{stage}_baseline_{si}.npz");xy=a["raw_projection"];coords.append(xy)
            ax[row,si].plot(np.r_[xy[:,0],xy[0,0]],np.r_[xy[:,1],xy[0,1]],color="gray",lw=.8)
            ax[row,si].scatter(xy[:,0],xy[:,1],c=np.arange(24),cmap="twilight",s=18)
            for hour in [0,6,12,18]:ax[row,si].annotate(str(hour),xy[hour],fontsize=7)
            ax[row,si].set_title(stage+"\n"+site.replace("dec_embedding.temporal_embedding.",""),fontsize=8)
            ax[row,si].set_aspect("equal",adjustable="box");ax[row,si].set_xlabel("Unsupervised centroid PC1");ax[row,si].set_ylabel("PC2")
        lim=np.max(np.abs(np.concatenate(coords)))*1.15
        for row in range(2):ax[row,si].set(xlim=(-lim,lim),ylim=(-lim,lim))
    fig.savefig(figs/"native_geometry.png",dpi=180);fig.savefig(figs/"native_geometry.pdf");plt.close(fig)
    fig,ax=plt.subplots(2,4,figsize=(11,5.4),layout="constrained")
    for si,site in enumerate(sites):
        for row,stage in enumerate(["init","trained"]):
            a=np.load(root/"native"/f"0_{stage}_baseline_{si}.npz");dm=a["distance_matrix"];dm=dm/dm[np.triu_indices(24,1)].mean()
            im=ax[row,si].imshow(dm,vmin=0,vmax=2.5,cmap="viridis",origin="lower",interpolation="nearest")
            ax[row,si].set_title(stage+"\n"+site.replace("dec_embedding.temporal_embedding.",""),fontsize=8);ax[row,si].set(xlabel="Hour",ylabel="Hour")
    fig.colorbar(im,ax=ax.ravel().tolist(),label="Raw centroid distance / mean off-diagonal distance",shrink=.65)
    fig.savefig(figs/"distances.png",dpi=180);fig.savefig(figs/"distances.pdf");plt.close(fig)
    conds=["circle","raw_pca_norm","residual_pca_norm","cv_phase_norm","recon_match_norm","info_match_norm","joint_match_norm","conditional_full","conditional_circle","harmonic_mean_delta"]
    fig,ax=plt.subplots(1,4,figsize=(12,5),sharey=True,sharex=True,layout="constrained")
    for axis,site in zip(ax,sites):
        v=s[s.site.eq(site)].set_index("condition").loc[conds]
        axis.errorbar(v.recovery,np.arange(len(v)),xerr=np.stack([v.recovery-v.low,v.high-v.recovery]),fmt="o",ms=4,color="#0072B2",capsize=2)
        axis.set_yticks(np.arange(len(v)),conds);axis.axvline(0,color="gray",lw=.6);axis.set_title(site.replace("dec_embedding.temporal_embedding.",""),fontsize=8);axis.set_xlabel("Recovery toward clean forecast")
    ax[0].invert_yaxis();fig.savefig(figs/"matched_interventions.png",dpi=180);fig.savefig(figs/"matched_interventions.pdf");plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for axis,site in zip(ax,[sites[0],sites[-1]]):
        for name,marker in [("circle","o"),("info_match_norm","s"),("cv_phase_norm","^"),("conditional_full","D"),("conditional_circle","v"),("harmonic_mean_delta","P")]:
            v=s[(s.site==site)&(s.condition==name)].iloc[0]
            axis.scatter(v.phase_error_deg,v.nonharmonic_mse,marker=marker,s=40,label=name)
        axis.set(title=site.replace("dec_embedding.temporal_embedding.",""),xlabel="Phase error (degrees)",ylabel="Nonharmonic output MSE")
    ax[1].legend(fontsize=7)
    fig.savefig(figs/"phase_fidelity.png",dpi=180);fig.savefig(figs/"phase_fidelity.pdf");plt.close(fig)
    provenance=dict(raw_source="native_geometry.csv, interventions.csv, calibration.csv, test_space_scores.csv",
        native_projection="fit hour-centroid unsupervised PCA2; no sin/cos targets, whitening, rotation to phase or sorting by fitted angle",
        uncertainty="1000 within-cohort source-trajectory bootstrap; 192 source trajectories, fixed3 models and2 shifts averaged",
        selection="fixed seed0 panels; all models/layers in tables; phase/fidelity metrics preregistered",
        missing="amplitude<1e-4 excluded only from phase summaries; all MSE retained",
        normalization="distance heatmaps divided by each matrix mean off-diagonal distance only; raw distances retained",
        supervised_matching="linear phase score is not Shannon information; see matching flags",
        source_commit="94fb6cb44da7840ced2c12fc18e8ff1e6a71ee50")
    (out/"figure_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(s[s.condition.isin(["circle","residual_pca_norm","cv_phase_norm","info_match_norm","joint_match_norm","conditional_full","conditional_circle","harmonic_mean_delta"])][["site","condition","recovery","phase_error_deg","amplitude_error","nonharmonic_mse"]].to_string(index=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);main(Path(p.parse_args().run))
