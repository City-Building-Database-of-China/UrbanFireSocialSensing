"""Public package invariants, documented entry points and privacy regressions."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from model.preprocessing import FEATURES, load_matrix
from validation.validate_public_data import validate
from validation.validate_figure_source_data import validate_figures
from validation.privacy_check import scan, BLOCKED, normalized
from validation.official_statistics import compute


class PublicReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.X, cls.y = load_matrix(ROOT/'demo/data/shanghai_buildings_anonymized_10k.csv', demo=True)
        cls.readme = (ROOT/'README.md').read_text(encoding='utf-8')

    def test_01_sample_rows(self):
        self.assertEqual(len(self.frame), 10000)

    def test_02_positive_labels(self):
        self.assertEqual(int(self.y.sum()), 191)

    def test_03_negative_labels(self):
        self.assertEqual(int(self.y.eq(0).sum()), 9809)

    def test_04_nine_predictors(self):
        self.assertEqual(list(self.X), FEATURES)
        self.assertEqual(len(FEATURES), 9)
        self.assertEqual(pd.read_csv(ROOT/'metadata/predictor_dictionary.csv').predictor.tolist(), FEATURES)

    def test_05_binary_fire_label(self):
        self.assertIn('fire', self.frame)
        self.assertEqual(set(self.y), {0,1})

    def test_06_public_privacy_scope(self):
        result = scan(ROOT)
        self.assertEqual(result['findings'], [])

    def test_07_events(self):
        self.assertEqual(validate()['event_aggregates'], 420)

    def test_08_manual_validation(self):
        r = validate()
        self.assertEqual((r['manual_records'], r['manual_valid'], r['manual_rejected']), (446,427,19))

    def test_09_age_window(self):
        table = pd.read_csv(ROOT/'figure/figure3/a/Fig3a_age_fire_relationship.csv')
        self.assertEqual(set(table.observation_start_year), {2016})
        self.assertEqual(set(table.observation_end_year), {2025})

    def test_10_main_figure_manifest(self):
        labels = set(pd.read_csv(ROOT/'metadata/figure_data_manifest.csv').Figure)
        self.assertTrue({f'Figure {i}' for i in range(1,7)} <= labels)

    def test_11_supplementary_manifest(self):
        labels = set(pd.read_csv(ROOT/'metadata/figure_data_manifest.csv').Figure)
        self.assertTrue({f'Figure S{i}' for i in range(1,10)} <= labels)

    def test_12_tables_and_numerical_sources(self):
        r = validate_figures()
        self.assertEqual(r['supplementary_tables'], 8)
        self.assertEqual(r['fig6_nodes'], 45)

    def test_13_readme_commands_and_links(self):
        blocks = re.findall(r'```(?:bash|powershell)\n(.*?)```', self.readme, re.S)
        for block in blocks:
            for line in block.strip().splitlines():
                if line.startswith('python '):
                    args=line.split()[1:]
                    if args[0].endswith('.py'):
                        self.assertTrue((ROOT/args[0]).is_file(), line)
                    elif args[:2] == ['-m','unittest']:
                        self.assertTrue((ROOT/'tests/test_public_release.py').is_file())
                    elif args[:2] == ['-m','pip']:
                        self.assertTrue((ROOT/args[-1]).is_file())
                    else:
                        self.assertEqual(args[:2], ['-m','venv'])
                else:
                    self.assertIn(line, [r'.venv\Scripts\Activate.ps1','source .venv/bin/activate'])
        for path in ROOT.rglob('*.md'):
            for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if not link.startswith(('http:', 'https:', '#')):
                    self.assertTrue((path.parent/link.split('#')[0]).exists(), f'{path.name}: {link}')

    def test_14_demo_and_documented_analysis_commands_execute(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'demo'
            commands = [
                ['demo/run_demo.py','--output-dir',str(output)],
                ['code/run_all.py','--output-dir',str(Path(tmp)/'general')],
                ['demo/run_demo.py','--stage','check'],
                ['demo/run_demo.py','--stage','oof','--output-dir',str(output)],
                ['demo/run_demo.py','--stage','shap','--output-dir',str(output)],
                ['code/validation/official_statistics.py','--output-dir',str(Path(tmp)/'official')],
                ['code/validation/validate_public_data.py'],
                ['code/validation/validate_figure_source_data.py'],
                ['code/run_all.py','--validate-only'],
            ]
            for args in commands:
                result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            oof = pd.read_csv(output/'oof_predictions.csv')
            self.assertEqual(len(oof),10000)
            self.assertEqual(set(oof.fold),{1,2,3,4,5})
            self.assertTrue(oof.oof_probability.between(0,1).all())
            self.assertTrue(oof.public_building_id.is_unique)
            self.assertEqual(len(list((output/'dependence').glob('*.csv'))),9)
            shap = pd.read_csv(output/'shap_values_or_public_summary.csv')
            self.assertEqual(shap.filter(like='SHAP_').shape,(10000,9))
            self.assertTrue(np.isfinite(shap.filter(like='SHAP_')).all().all())
            generic=pd.read_csv(Path(tmp)/'general/oof_predictions.csv')
            np.testing.assert_array_equal(oof.oof_probability,generic.oof_probability)
            self.assertEqual(set(pd.read_csv(output/'model_metrics.csv').metric),{'AUROC','AUPRC','Brier score','Log loss'})

    def test_15_current_title(self):
        self.assertTrue(self.readme.startswith('# Coupled socio-physical systems govern urban fire risk'))
        obsolete = 'Coupled socio-physical systems govern urban fire ' + 'vulnerability'
        self.assertNotIn(obsolete.lower(), self.readme.lower())

    def test_16_public_readme_language(self):
        excluded = ['manuscript'+'0908', 'Supplementary'+'0902', 'release'+' candidate',
                    'under'+' review', 'remaining alignment'+' questions', 'TO BE'+' CONFIRMED']
        for phrase in excluded:
            self.assertNotIn(phrase.lower(),self.readme.lower())

    def test_17_no_full_model_matrix(self):
        matrices=[]
        for path in ROOT.rglob('*.csv'):
            headers=pd.read_csv(path,nrows=0).columns
            if set(FEATURES)<=set(headers):
                matrices.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(matrices,['demo/data/shanghai_buildings_anonymized_10k.csv'])

    def test_18_privacy_scanner_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame({'public_event_id':['anonymous_1'],'has_address':[1]}).to_csv(root/'safe.csv',index=False)
            self.assertEqual(scan(root)['findings'],[])
            pd.DataFrame({'latitude':[31.123456],'user_id':['private']}).to_csv(root/'blocked.csv',index=False)
            (root/'blocked.json').write_text(json.dumps({'nested':{'longitude':121.12345}}))
            pd.DataFrame({'address':['precise location']}).to_excel(root/'blocked.xlsx',index=False)
            (root/'blocked.md').write_text('Workstation: '+chr(67)+':'+chr(92)+'Users'+chr(92)+'private')
            result=scan(root)
            self.assertEqual({f['file'] for f in result['findings']},{'blocked.csv','blocked.json','blocked.xlsx','blocked.md'})

    def test_19_official_2020_statistics(self):
        table, coefficient=compute(ROOT/'figure/supplementary/figureS7')
        np.testing.assert_allclose(table.pearson_r,[.7268161865504065,.811915172167098])
        self.assertEqual(coefficient['n'],29)
        self.assertAlmostEqual(coefficient['coefficient'],471.452206,places=5)

    def test_20_dictionary_coverage_and_package_size(self):
        dictionary=pd.read_csv(ROOT/'metadata/data_dictionary.csv')
        for path in ROOT.rglob('*.csv'):
            if 'output' in path.parts or 'metadata' in path.parts:
                continue
            defined=dictionary[dictionary.Public_file==path.relative_to(ROOT).as_posix()]
            self.assertEqual(set(defined.Field),set(pd.read_csv(path,nrows=0).columns))
        self.assertFalse([p for p in ROOT.rglob('*') if p.is_file() and p.stat().st_size>=100*1024**2])

    def test_21_user_supplied_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'code/run_all.py','--input',
                'demo/data/shanghai_buildings_anonymized_10k.csv','--stage','oof','--output-dir',tmp],
                cwd=ROOT,text=True,capture_output=True,timeout=120,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(set(pd.read_csv(Path(tmp)/'model_metrics.csv').scope),{'user-supplied matrix OOF'})


if __name__=='__main__':
    unittest.main()
