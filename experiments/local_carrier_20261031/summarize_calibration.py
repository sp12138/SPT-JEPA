"""CPU-only isotropic-normalization counterfactual from frozen measurements."""
import json,hashlib,math
from pathlib import Path
import numpy as np

ROOT=Path('/efs/prescottsun/SPT-JEPA/reports/local-calibration-diagnosis-20261031')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(z):
    flat=z.reshape(-1,1024);energy=(flat**2).mean(0);coordstd=flat.std(0)
    order=np.argsort(energy)[::-1]
    # At initial zero output head the native predictor is exact carry. Average
    # the original correlated Gaussian posterior noise analytically, not by RNG.
    sigma=.1;rho=.8;w=z.shape[1];d=z.shape[2];fe=3*64*64
    entry=.5*(z[:,0]**2).sum(1)+d*(.5*sigma*sigma-.5-math.log(sigma))
    delta=np.diff(z,axis=1);temporal_bycoord=(delta**2).mean((0,1))/(2*sigma*sigma)
    mean_noise=.5*d*(1-rho)**2
    covariance_kl=.5*d*(-rho*rho-math.log1p(-rho*rho))
    surprise=(delta**2).sum(2)/(2*sigma*sigma)+mean_noise+covariance_kl
    rate=float((entry+surprise.sum(1)).mean()/((w-1)*fe))
    return dict(rms=float(np.sqrt((flat**2).mean())),mean_coordinate_std=float(coordstd.mean()),
        mean_coordinate_abs_bias=float(np.abs(flat.mean(0)).mean()),max_abs=float(np.abs(flat).max()),
        top16_energy_fraction=float(energy[order[:16]].sum()/energy.sum()),
        top16_coordinates=order[:16].tolist(),top16_spatial_cells=(order[:16]//16).tolist(),
        expected_initial_carry_rate=rate,entry_rate=float(entry.mean()/((w-1)*fe)),
        temporal_signal_rate=float(temporal_bycoord.sum()/fe),
        posterior_noise_and_covariance_rate=float((mean_noise+covariance_kl)/fe),
        top16_temporal_signal_fraction=float(np.sort(temporal_bycoord)[-16:].sum()/temporal_bycoord.sum()))


def main():
    result=dict(helper_sha256=sha(__file__),source_commit='96aca13f99b25dcf880346a1e699318150139ae5',
        operation='Keep original128-frame per-coordinate mean. Replace all1024 scales by sqrt(mean of their128-frame population variances). No new images, fit, tuning, validation or model change.',
        rate_definition='Analytic expectation of native rate at exact initial carry head under rho=.8,sigma=.1,window8. Includes entry and covariance constants; not measured step2 loss.',
        all_floor_1e_6_untriggered=True,rows=[])
    for task in ('tworoom','cube'):
        for seed in (1,2):
            folder=ROOT/f'{task}-seed{seed}';report=read(folder/'report.json');cal=read(report['calibration'])
            arrays=np.load(folder/'measurements.npz')
            for arm in ('local','dense'):
                kwargs=cal['arms'][arm]['model_kwargs'];mean=np.asarray(kwargs['spatial_mean'],dtype=np.float64)
                scale=np.asarray(kwargs['spatial_scale'],dtype=np.float64);global_scale=float(np.sqrt((scale**2).mean()))
                assert cal['arms'][arm]['scale_floor_coordinates']==0
                row=dict(task=task,seed=seed,arm=arm,global_scale=global_scale,coordinate_scale_min=float(scale.min()),
                    source_report_sha256=sha(folder/'report.json'),calibration_sha256=sha(report['calibration']),
                    floor_1e_6_coordinates=0,precision={},calibration={})
                for path in ('HH','LL'):
                    raw=arrays[f'native_first_batch__{arm}__{path}'].astype(np.float64)
                    row['precision'][path]=dict(diagonal=stats((raw-mean)/scale),isotropic=stats((raw-mean)/global_scale))
                    calraw=arrays[f'calibration__{arm}__{path}'].astype(np.float64)
                    row['calibration'][path]=dict(diagonal=stats((calraw-mean)/scale),isotropic=stats((calraw-mean)/global_scale))
                h=arrays[f'native_first_batch__{arm}__HH'].astype(np.float64)
                l=arrays[f'native_first_batch__{arm}__LL'].astype(np.float64)
                row['normalized_precision_rms']=dict(diagonal=float(np.sqrt(np.mean(((l-h)/scale)**2))),
                    isotropic=float(np.sqrt(np.mean(((l-h)/global_scale)**2))))
                result['rows'].append(row)
                print(task,seed,arm,json.dumps(dict(global_scale=global_scale,
                    high_diagonal=row['precision']['LL']['diagonal'],high_isotropic=row['precision']['LL']['isotropic'],
                    highest_diagonal_rate=row['precision']['HH']['diagonal']['expected_initial_carry_rate'])),flush=True)
    (ROOT/'isotropic-counterfactual.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
