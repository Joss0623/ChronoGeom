"""CPU analysis of official cached GPT-J activations, not an inference run."""
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse, csv, hashlib, json, platform, subprocess, time
import numpy as np
from scripts.chronogeom.representation_probe import basis, fit_predict, metrics, split_values


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def analyze(a, activations, output, layers=(0, 1), seeds=(0, 1, 2, 3, 4)):
    rows, folds, previews = [], {}, {}
    configs = [('linear', 'linear', {}), ('circle_T100', 'circle', {'periods': [100.]}),
               ('fourier', 'fourier', {}), ('helix', 'helix', {}),
               ('polynomial8', 'polynomial', {}), ('shuffled_helix', 'helix', {})]
    for layer in layers:
        y = np.asarray(activations[:, layer, :], dtype=np.float64)
        for seed in seeds:
            train, test = split_values(a, seed, .8)
            folds[f'layer{layer}_seed{seed}'] = {'train_rows': np.where(train)[0].tolist(),
                'test_rows': np.where(test)[0].tolist(), 'train_values': np.unique(a[train]).tolist(),
                'test_values': np.unique(a[test]).tolist()}
            mean = y[train].mean(axis=0)
            _, singular, vt = np.linalg.svd(y[train] - mean, full_matrices=False)
            n_pca = min(100, len(vt))
            components = vt[:n_pca]
            ztrain = (y[train] - mean) @ components.T
            designs = {}
            for name, kind, kwargs in configs:
                design = basis(a, kind, **kwargs)
                xtrain = design[train]
                if name == 'shuffled_helix':
                    xtrain = xtrain[np.random.default_rng(seed + 1000).permutation(train.sum())]
                zpred, info = fit_predict(xtrain, ztrain, design)
                prediction = zpred @ components + mean
                designs[name] = prediction
                for split, mask in [('train', train), ('heldout_values', test)]:
                    row = dict(layer=layer, seed=seed, method=name, split=split,
                        evidence='official_cached_activations', evaluation='numeric_to_activation',
                        n_rows=int(mask.sum()), n_values=int(np.unique(a[mask]).size),
                        pca_fit='train_only', pca_components=n_pca,
                        columns=info['columns'], effective_rank=info['effective_rank'])
                    row.update(metrics(y[mask], prediction[mask])); rows.append(row)
            for rank in [2, 8, 9]:
                comp = vt[:rank]
                prediction = (y - mean) @ comp.T @ comp + mean
                for split, mask in [('train', train), ('heldout_values', test)]:
                    row = dict(layer=layer, seed=seed, method=f'PCA_reconstruct_{rank}', split=split,
                        evidence='official_cached_activations', evaluation='observed_activation_reconstruction',
                        n_rows=int(mask.sum()), n_values=int(np.unique(a[mask]).size),
                        pca_fit='train_only', pca_components=rank, columns=rank, effective_rank=rank)
                    row.update(metrics(y[mask], prediction[mask])); rows.append(row)
            if seed == 0:
                previews[str(layer)] = {'a': a, 'train': train, 'pca2': (y - mean) @ vt[:2].T,
                    'helix_pca2': (designs['helix'] - mean) @ vt[:2].T,
                    'linear_pca2': (designs['linear'] - mean) @ vt[:2].T}
    with open(output / 'fit_metrics.csv', 'x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    (output / 'splits.json').write_text(json.dumps(folds, indent=2) + '\n')
    for layer, values in previews.items(): np.savez(output / f'preview_layer{layer}.npz', **values)
    summary = []
    for layer in layers:
        for name in [c[0] for c in configs] + ['PCA_reconstruct_2', 'PCA_reconstruct_8', 'PCA_reconstruct_9']:
            selected = [r for r in rows if r['layer'] == layer and r['method'] == name and r['split'] == 'heldout_values']
            scores = np.array([r['r2'] for r in selected])
            entry = dict(layer=layer, method=name, r2_mean=float(scores.mean()), r2_sd=float(scores.std(ddof=1)), n=len(scores),
                evaluation=selected[0]['evaluation'], effective_rank=selected[0]['effective_rank'])
            summary.append(entry); print(json.dumps(entry), flush=True)
    (output / 'fit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(2)
    start = time.monotonic()
    obj = torch.load(args.cache, map_location='cpu', weights_only=True)
    if not isinstance(obj, dict) or not all(k in obj for k in ['a', 'b', 'a+b', 'hidden_states']):
        raise ValueError('Unexpected cache schema')
    a = obj['a'].cpu().numpy().astype(np.float64)
    hidden = obj['hidden_states']
    if hidden.ndim != 3 or len(a) != hidden.shape[0] or len(np.unique(a)) < 10:
        raise ValueError('Unexpected activation axes or label coverage')
    print('Cache shapes', {k: list(v.shape) for k, v in obj.items() if hasattr(v, 'shape')}, flush=True)
    # Only materialize selected CPU layers; do not load a model or tokenizer.
    selected = hidden[:, :2, :].float().cpu().numpy()
    analyze(a, selected, args.output)
    metadata = dict(evidence='official_cached_activations', model='EleutherAI/gpt-j-6B',
        model_revision='not specified by source cache', tokenizer_revision='not specified by source cache',
        source_repo='https://github.com/subhashk01/LLM-addition',
        source_commit=subprocess.check_output(['git', '-C', str(args.official_root), 'rev-parse', 'HEAD'], text=True).strip(),
        cache_name=args.cache.name, cache_sha256=digest(args.cache), shape=list(hidden.shape),
        source_dtype=str(hidden.dtype), observed_values=np.unique(a).tolist(), layers=[0, 1], seeds=[0, 1, 2, 3, 4],
        protocol='80% of unique numeric identities; PCA trained only on training rows; fixed periods [2,5,10,100]; scale 99',
        layer_semantics='Source run_hs captures transformer block inputs; index 1 is candidate post-block-0 state. Cache-generation revision metadata absent; report both indices without selecting by score.',
        pca_warning='PCA reconstructions see test activations; diagnostic, not a fair numerical-prediction baseline.',
        hardware='CPU only; two BLAS/torch threads; no model loaded', elapsed_seconds=time.monotonic()-start,
        python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(),
        code_sha256={p.name: digest(p) for p in [Path(__file__), ROOT/'scripts/chronogeom/representation_probe.py']})
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Elapsed CPU-analysis wall seconds', metadata['elapsed_seconds'], flush=True)


if __name__ == '__main__': main()
