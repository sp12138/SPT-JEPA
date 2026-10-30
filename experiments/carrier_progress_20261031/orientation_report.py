"""Render the completed saved-array orientation analysis; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument("--results",required=True);a=p.parse_args()
    path=Path(a.results);out=path.parent;r=json.loads(path.read_text())
    rows=r["rows"];assert len(rows)==4 and r["contracts"]["passed"]
    refs=rows[0]["codebook_references"]
    lines=["# Frozen carrier orientation diagnostic", "",
        "The four Cube F600 checkpoints have weak held-out linear orientation readout under this fixed probe. The bounded chart does not materially improve it. This narrows a gap left by the earlier xyz probes; it does not establish that orientation information is absent, or that orientation alone causes planning failure.", "",
        "| Arm / seed | Observed invariant R² | Observed angle | Forecast angle | Initial-carry angle | Observed / forecast within 15° |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        s=row["scores"]["cube24_tensor81"];obs,forecast,carry=[s[k]for k in ["observed_future","clean_rollout","initial_carry"]]
        lines.append(f"| {row['arm']} / {row['seed']} | {obs['features']['variance_weighted_r2']:.4f} | {obs['angles']['mean_deg']:.2f}° | {forecast['angles']['mean_deg']:.2f}° | {carry['angles']['mean_deg']:.2f}° | {100*obs['angles']['fraction_within15deg']:.1f}% / {100*forecast['angles']['fraction_within15deg']:.1f}% |")
    const=refs["constant_training_invariant_mean_decode"];floor=refs["true_invariant_feature_codebook_decode"]
    lines += ["",f"The constant training-mean reference has {const['mean_deg']:.2f}° mean error and {100*const['fraction_within15deg']:.1f}% within 15°. The exact true-feature training-codebook decoder has only {floor['mean_deg']:.3f}° mean error and 100% within 15°, so the poor readout is not explained by coarse codebook coverage. This decoder control uses true orientation only to measure the diagnostic's resolution; it is not a model result.", "",
        "Forecast versus copying the initial state is mixed in R² and small in angular improvement: roughly 0.8° for seed 1 and 0.4° for seed 2. Identity versus bounded forecast angles differ by less than 0.05°. Teacher-forced and shifted-observed controls are retained in results.json; they receive new real observations and are not free dynamics predictions.", "",
        "The conventional rotation-matrix9 control also has negative observed variance-weighted R² (approximately −.148 to −.176). Unlike the primary target, its component labels distinguish physically equivalent cube orientations; it is reported as a stricter control, not evidence against the cube-symmetry-aware result.", "",
        "## Target and decoding", "",
        "The pinned evaluator uses wxyz quaternions, minimum geodesic angle over 24 proper cube rotations, and a 15° threshold. We convert quaternions with its own implementation. For rotation matrix R with columns a₁,a₂,a₃, the primary linear target is the 81-component fourth-order tensor M(R)=Σⱼ aⱼ⊗aⱼ⊗aⱼ⊗aⱼ. It is invariant to quaternion sign and to signed permutations of cube axes. For unit v, contraction M[v,v,v,v]=Σⱼ(v·aⱼ)⁴ attains its maximum 1 precisely on the six unoriented axes. Thus equal tensors identify the same cube orientation up to the evaluator's proper cube symmetries. This is an established invariant construction, not an algorithm novelty claim.", "",
        "A fixed training-quaternion codebook decodes predicted tensors by nearest tensor distance, with no validation tuning or fitted nonlinear decoder. The rotation9 control uses the nearest SO(3) matrix from SVD. Both report physical angles with the same pinned 24-symmetry function. Feature R² and angular error measure different losses; their apparent ordering need not agree.", "",
        "## Data and fairness", "",
        "Reuse only the existing metric-calibration 128 training clips ×8 frames and state-audit 64 validation clips ×8 frames; score the same 448 future frames across all conditions. Exact HDF5 rows and episode IDs are checked. Training/validation episodes are disjoint; saved embedding, model-weight, dataset, manifest and source hashes are retained. The readout is the existing benchmark protocol: training-only standardization, coordinate std floor1e−12, ridge .001 in mean-squared-error units. No encoder or GPU is used, and no model, planner, goal metric or evaluation case is changed. Privileged quaternion labels fit this diagnostic readout only.", "",
        "This fixed small linear probe can fail because of nonlinear encoding, readout estimation, or representation limitations. Negative held-out R² alone cannot distinguish these. The comparison does show that earlier strong position probes were insufficient evidence of a usable full pose representation, and that the bounded chart has not resolved this particular gap.", "",
        "## Hold is not measured", "",
        "Strict Cube requires position ≤3 cm and symmetry-aware orientation ≤15° for three consecutive physical steps. The pinned runner resets and updates a goal-dependent counter. Saved embeddings and predictions are five physical steps apart; neither the between-frame states nor this evaluation-specific counter is available. HDF5 `success` is the upstream dataset flag and is not silently treated as strict hold. We therefore do not fit a purported hold label or equate the orientation-within15 fraction with planning success.", "",
        "No extra cases, model training, hyperparameter search, encoder calls or GPU jobs were used. Raw readout weights and predicted orientation features remain on EFS; the small results JSON contains all per-frame errors, feature metrics, constants and provenance.", "",
        "![Orientation readout](orientation-readout.png)", ""]
    (out/"RESULTS.md").write_text("\n".join(lines))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout="constrained")
    fig.suptitle("Cube orientation at 600 updates · same saved 64 validation clips",fontsize=14)
    conditions=[("observed_future","Observed state","#327ba6"),("clean_rollout","Causal forecast","#cc7047"),
                ("initial_carry","Initial carry","#777777")]
    for i,row in enumerate(rows):
        for j,(condition,label,color)in enumerate(conditions):
            s=row["scores"]["cube24_tensor81"][condition]
            x=i+(j-1)*.18;marker="o"if row["seed"]==1 else "^"
            for ax,value in zip(axes,[s["features"]["variance_weighted_r2"],s["angles"]["mean_deg"]]):
                ax.scatter(x,value,s=50,color=color,marker=marker,label=label if i==0 else None,zorder=3)
    axes[0].axhline(0,color="black",ls="--",lw=.8,label="Constant train-mean R²")
    axes[1].axhline(const["mean_deg"],color="black",ls="--",lw=.8,label="Constant train mean")
    axes[1].axhline(floor["mean_deg"],color="#5b9460",ls=":",lw=1,label="True-feature codebook floor")
    axes[0].set_ylabel("Variance-weighted R² of cube24-invariant tensor")
    axes[1].set_ylabel("Mean orientation error over 24 cube symmetries (°)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xticks(np.arange(4),[f"{x['arm']}\nseed {x['seed']}"for x in rows])
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18);ax.legend(frameon=False,fontsize=8)
    fig.supxlabel("Fixed train-only linear readout; angular decoder uses training poses only. These are orientation diagnostics, not strict planning/hold success.",fontsize=9)
    for suffix in ["png","svg"]:fig.savefig(out/("orientation-readout."+suffix),dpi=180,bbox_inches="tight")
    plt.close(fig)
    (out/"report-provenance.json").write_text(json.dumps(dict(results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+"\n")


if __name__=="__main__":main()
