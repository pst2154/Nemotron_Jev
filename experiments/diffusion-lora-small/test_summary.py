"""CPU-only checks for report arithmetic and incomplete-run rejection."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name)
        self.write('audit.json', {'steps':1,'training_rows':16,'training_tokens':[200,400],
                                 'heldout_rows':4,'jevbench_rows':2})
        self.write('selected.json', {'step':1})
        self.lines('training.jsonl', [dict(step=1, examples_in_update=16, seconds=10,
                                        peak_gib=40,loss=.5,grad_norm=1)])
        self.lines('validation-selection.jsonl', [dict(step=0,correct=1,cross_entropy=1),
                                                  dict(step=1,correct=2,cross_entropy=.5)])
        for mode in ('base','trained'):
            self.lines(f'{mode}-heldout.jsonl', [dict(id=prefix+'example',correct=True)
                for prefix in ('google/boolq:','google-research-datasets/paws:','scale-','other-synthetic-')])
            self.lines(f'{mode}-jevbench.jsonl', [dict(id=str(i),correct=bool(i) or mode=='trained',
                strict_valid=True,renormalized=False,probs={'a':.7,'b':.3}) for i in range(2)])
        self.lines('paired-latency.jsonl', [dict(id=str(i),repeat=r,adapter_enabled=enabled,
             latency_ms=100 if enabled else 80) for i in range(18) for r in range(3)
             for enabled in (False,True)])

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, value):
        (self.run/name).write_text(json.dumps(value))

    def lines(self, name, values):
        (self.run/name).write_text(''.join(json.dumps(value)+'\n' for value in values))

    def summarize(self):
        return subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_small.py')),
                               '--run',str(self.run)],capture_output=True,text=True)

    def test_counts_and_paired_latency(self):
        result = self.summarize()
        self.assertEqual(result.returncode,0,result.stderr)
        summary = json.loads((self.run/'summary.json').read_text())
        self.assertEqual(summary['accuracy']['trained/jevbench/all'],dict(correct=2,total=2))
        self.assertEqual(summary['jevbench_transitions'],dict(wrong_to_correct=1,correct_to_wrong=0))
        self.assertEqual(summary['warm_latency_median_ms'],{'False':80,'True':100})
        self.assertEqual(summary['accuracy']['trained/heldout/synthetic'],dict(correct=2,total=2))

    def test_rejects_incomplete_training(self):
        self.lines('training.jsonl',[])
        self.assertNotEqual(self.summarize().returncode,0)

    def test_rejects_wrong_selected_checkpoint(self):
        self.write('selected.json',{'step':0})
        self.assertNotEqual(self.summarize().returncode,0)

    def test_rejects_invalid_probabilities(self):
        self.lines('trained-jevbench.jsonl',[dict(id=str(i),correct=True,strict_valid=True,
            renormalized=False,probs={'a':.7,'b':.7}) for i in range(2)])
        self.assertNotEqual(self.summarize().returncode,0)

    def test_rejects_missing_latency_pair(self):
        self.lines('paired-latency.jsonl',[dict(id='same',repeat=0,adapter_enabled=True,
                                              latency_ms=100)]*108)
        self.assertNotEqual(self.summarize().returncode,0)


if __name__ == '__main__':
    unittest.main()
