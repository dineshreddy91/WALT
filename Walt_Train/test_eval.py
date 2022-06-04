import numpy as np
from code_local.datasets.cocoeval import COCOeval
import pickle

with open(f'testGt.pickle','rb') as file:
    cocoGt = pickle.load(file)

with open(f'testDt.pickle','rb') as file:
    result_files = pickle.load(file)
cocoDt = cocoGt.loadRes(result_files['segm'])
cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
cocoEval.params.useCats = 0
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

cocoDt = cocoGt.loadRes(result_files['bbox'])
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.useCats = 0
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


