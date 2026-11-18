"""Source-function conformance and synthetic Circle positive/negative controls.
No LLM or tokenizer is imported; no upstream module top-level code is executed.
"""
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import ast, argparse, csv, hashlib, json, re, logging, math, time
import numpy as np
from scripts.chronogeom.representation_probe import basis


def extract_function(source, name, namespace):
    module = ast.parse(source)
    nodes = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(nodes) != 1: raise ValueError('Expected exactly one source definition: ' + name)
    isolated = ast.Module(body=nodes, type_ignores=[])
    exec(compile(isolated, '<reviewed-official-definition>', 'exec'), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.decomposition import PCA
    torch.set_num_threads(2)
    helix_file = args.official_root/'LLM-addition/experimentation/helix_fitting.ipynb'
    notebook = json.loads(helix_file.read_text())
    cell = next(''.join(c['source']) for c in notebook['cells'] if 'def create_features(' in ''.join(c.get('source', [])))
    create_features = extract_function(cell, 'create_features', {'np': np})
    x = np.arange(100.)
    comparisons = []
    for ours, official, kwargs in [('helix', 'helix', {}), ('fourier', 'circle', {}),
                                    ('circle', 'circle', {'periods': [100.]})]:
        expected = create_features(official, np.array([x]), bases=kwargs.get('periods', [2,5,10,100]))
        observed = basis(x, ours, **kwargs)
        delta = float(np.max(np.abs(expected-observed)))
        if delta > 1e-12: raise AssertionError((ours, delta))
        comparisons.append({'basis': ours, 'max_abs_difference': delta})
    circle_file = args.official_root/'MultiDimensionalFeatures/intervention/circle_finding_utils.py'
    do_regression = extract_function(circle_file.read_text(), 'do_regression', {'torch': torch, 'np': np, 'PCA': PCA})
    a = torch.tensor([i for i in range(7) for _ in range(7)])
    b = torch.tensor([i for _ in range(7) for i in range(7)])
    c = (a+b) % 7
    nuisance = [torch.ones_like(a)] + [a==i for i in range(8)] + [b==i for i in range(8)]
    nuisance += [torch.cos(2*np.pi*(a-b)/7), torch.sin(2*np.pi*(a-b)/7)]
    answers = [torch.cos(2*np.pi*c/7), torch.sin(2*np.pi*c/7)]
    circle_rows = []
    for seed in range(5):
        torch.manual_seed(seed)
        # Explicitly synthetic acts, generated from answer circle plus nuisance.
        target = torch.stack(nuisance+answers, dim=1).float() @ torch.randn(len(nuisance)+2, 32)
        before = do_regression(None, nuisance, target, verbose=False)[0]
        after = do_regression(None, nuisance+answers, target, verbose=False)[0]
        perm = torch.randperm(len(c))
        control = do_regression(None, nuisance+[v[perm] for v in answers], target, verbose=False)[0]
        circle_rows.append({'seed': seed, 'evidence': 'synthetic_source_function_calibration',
                            'before_answer_circle_r2': before, 'after_answer_circle_r2': after,
                            'shuffled_answer_circle_r2': control})
        if after < .999: raise AssertionError('Official positive-control regression failed')
    fone_dir = args.official_root/'FoNE'
    preprocess = extract_function((fone_dir/'utils/data_utils.py').read_text(), 'preprocess', {'re': re})
    parser_rows=[]
    for text, expected in [('0',[0.]), ('3.14',[3.14]), ('-3.14',[-3.14]),
                           ('1,234.56',[1234.56]), ('3.14e-2',[.0314]), ('.5',[.5])]:
        result=preprocess({'question':text, 'label':0},False,method='fne')
        parser_rows.append({'text':text,'expected_canonical_numbers':expected,
            'observed_numbers':result['numbers'],'question_with_num':result['question_with_num'],
            'matches_canonical_parse':result['numbers']==expected})
    FNE = extract_function((fone_dir/'number_encoders/FNE.py').read_text(), 'FNE',
        {'np':np,'torch':torch,'nn':nn,'F':F,'logging':logging,'math':math,'time':time})
    encoder=FNE(embedding_dim=16,int_digit_len=3,frac_digit_len=0,period_base_list=[10.],add_linear=False,device='cpu')
    with torch.no_grad(): embedded=encoder(torch.tensor([0.,1.],dtype=torch.float64))
    results={'source_basis_conformance':comparisons,'circle_calibration':circle_rows,'fone_parser':parser_rows,
        'fone_zero_input_norm':float(embedded[0].norm()),'fone_one_input_norm':float(embedded[1].norm()),
        'fone_period_list':encoder.period_list,
        'limitations':'Circle targets are synthetic, not LLM activations. Parser failures are pinned-code behavior; not claims about all FoNE implementations.',
        'source_sha256':{str(p.relative_to(args.official_root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [helix_file,circle_file,fone_dir/'utils/data_utils.py',fone_dir/'number_encoders/FNE.py']}}
    (args.output/'source_calibration.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':main()
