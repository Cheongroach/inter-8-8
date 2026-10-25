"""Replay eight measured cases through the public calibration loader, not a new accuracy test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from evaluate import CRITERIA, EXPECTED_REVISION, read_dataset
from metrics import probabilities


def main():
    parser = argparse.ArgumentParser()
    for key in ('upstream','model','data','results','calibration','output'):
        parser.add_argument(f'--{key}',type=Path,required=True)
    args=parser.parse_args()
    revision=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    if revision!=EXPECTED_REVISION: raise ValueError('Source revision changed')
    protocol=json.loads((args.results/'protocol.json').read_text())
    selected=json.loads((args.results/'results.json').read_text())
    records=[json.loads(line) for line in (args.results/'predictions.jsonl').read_text().splitlines()]
    records=[row for row in records if row['split']=='test'][:8]
    source={row['id']:row for row in read_dataset(args.data/'test.csv','test')}
    if any(source[row['id']]['text_sha256']!=row['text_sha256'] for row in records):
        raise ValueError('Replay input changed')
    sys.path.insert(0,str(args.upstream))
    import torch
    import laya
    torch.set_num_threads(8)
    agent=laya.load(str(args.model),device='cpu',backend='eager',calibration=str(args.calibration),
                    expected_sha256={'model.safetensors':protocol['model_sha256']})
    questions={'topic':{'type':'choice','instructions':'What is the topic of `article`?','criteria':CRITERIA}}
    outputs=agent.predict_batch([{'article':source[row['id']]['state']} for row in records],questions,batch_size=8)
    expected=probabilities([row['logits'] for row in records],selected['selected_parameters']['temperature'])
    actual=np.asarray([[result['answers']['topic']['probabilities'][key] for key in CRITERIA] for result in outputs])
    error=float(np.max(np.abs(actual-expected)))
    same=bool(np.array_equal(actual.argmax(axis=1),expected.argmax(axis=1)))
    result={'rows':8,'max_abs_probability_difference':error,'choices_identical':same,
            'tolerance':0.00011,'passed':same and error<=0.00011,
            'scope':'runtime serialization/parity replay of already measured inputs; not new quality evidence',
            'temperature_loaded':agent.temperature_by_options['choice:3-5'],'checkpoint_verified':True}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
