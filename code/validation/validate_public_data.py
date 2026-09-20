"""Verify the immutable Shanghai demo and anonymous social-sensing tables."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from model.preprocessing import load_matrix, file_sha256, FEATURES


def validate(root=ROOT):
    root = Path(root)
    path = root / 'demo/data/shanghai_buildings_anonymized_10k.csv'
    frame, X, y = load_matrix(path, demo=True)
    expected = json.loads((root / 'metadata/demo_sample.json').read_text(encoding='utf-8'))
    if file_sha256(path) != expected['sha256']:
        raise ValueError('The fixed Shanghai file has changed.')
    predictor = pd.read_csv(root/'metadata/predictor_dictionary.csv')
    if predictor.predictor.tolist() != FEATURES or len(predictor) != 9:
        raise ValueError('The predictor dictionary must contain exactly the nine ordered predictors.')
    events = pd.read_csv(root/'code/data/social_sensing/shanghai_weibo_anonymized.csv')
    if len(events) != 420 or not events.public_event_id.is_unique:
        raise ValueError('Expected 420 unique anonymous Shanghai event aggregates.')
    manual = pd.read_csv(root/'code/data/social_sensing/shanghai_manual_validation.csv')
    if manual[['consensus_judgement','valid_fire_case']].isna().any().any():
        raise ValueError('Manual consensus and binary labels must not contain missing values.')
    if not manual.consensus_judgement.map({'\u662f':1,'\u5426':0}).equals(manual.valid_fire_case):
        raise ValueError('Manual consensus labels and binary fire labels disagree.')
    if len(manual)!=446 or manual.public_record_id.isna().any() or manual.valid_fire_case.sum()!=427 or manual.valid_fire_case.eq(0).sum()!=19:
        raise ValueError('Manual-validation composition changed.')
    result = dict(status='passed',demo_rows=len(frame),positives=int(y.sum()),negatives=int(y.eq(0).sum()),
                  predictors=9,event_aggregates=len(events),manual_records=len(manual),manual_valid=427,manual_rejected=19,
                  manual_distinct_anonymous_keys=int(manual.public_record_id.nunique()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    print(json.dumps(validate(args.root),indent=2))


if __name__=='__main__':
    main()
