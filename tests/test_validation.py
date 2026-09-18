import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from compat_gateway import validate_request


class ValidationTests(unittest.TestCase):
    def test_all_primitives(self):
        validate_request({'model':'jev-latest','state':{'text':'example'},'questions':{
            'choice':{'type':'choice','instructions':{'task':'pick'},'criteria':{'a':{'meaning':'one'},'b':'two'}},
            'noul':{'type':'noul','instructions':'yes?'},
            'score':{'type':'score','instructions':'degree','criteria':['low','high']}}})

    def test_missing_fields(self):
        for body in (None, [], {}, {'model':'x','state':'x','questions':{}},
                     {'model':'x','state':'x','questions':{'bad':{'type':'unknown','instructions':'x'}}}):
            with self.subTest(body=body), self.assertRaises(ValueError): validate_request(body)

    def test_invalid_options(self):
        for question in ({'type':'choice','instructions':'x','criteria':{'one':'one'}},
                         {'type':'score','instructions':'x','criteria':['one']},
                         {'type':'noul','instructions':'x','criteria':[]}):
            with self.subTest(question=question), self.assertRaises(ValueError):
                validate_request({'model':'x','state':'x','questions':{'q':question}})


if __name__ == '__main__': unittest.main()
