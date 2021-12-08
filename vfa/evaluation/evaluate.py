import numpy as np
import sys, os
from .pyeval.evaluateAPAOS import evaluateDetectionAPAOS
proj_path = os.getcwd()
sys.path.append(proj_path)

def evaluate_rcll_prec_moda_modp(res_fpath, gt_fpath, dataset='wildtrack', eval='matlab'):
    if eval == 'matlab':
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.cd(r'vfa\evaluation\motchallenge-devkit')    
        # relative path -> absolute path 
        res_fpath = proj_path + '\\' + res_fpath.split('\\', 1)[1] 
        gt_fpath = proj_path + '\\' + gt_fpath.split('\\', 1)[1] 
        res = eng.evaluateDetection(res_fpath, gt_fpath, dataset) #experiments\Wildtrack\evaluation\pr_dir_gt.txt
        recall, precision, moda, modp = np.array(res['detMets']).squeeze()[[0, 1, -2, -1]]
    elif eval == 'python':
        from vfa.evaluation.pyeval.evaluateDetection import evaluateDetection_py

        recall, precision, moda, modp = evaluateDetection_py(res_fpath, gt_fpath, dataset)
    else:
        raise ValueError('eval only has two mode: `python` and `matlab`. ')
    return recall, precision, moda, modp

def evaluate_ap_aos(res_fpath, gt_fpath):
    AP_75, AOS_75, OS_75, AP_50, AOS_50, OS_50, AP_25, AOS_25, OS_25 = evaluateDetectionAPAOS(res_fpath, gt_fpath)
    return AP_75, AOS_75, OS_75, AP_50, AOS_50, OS_50, AP_25, AOS_25, OS_25

if __name__ == "__main__":


    res_fpath = os.path.abspath('.\\evaluation\\test-demo.txt')
    gt_fpath = os.path.abspath('.\\evaluation\\gt-demo.txt')
    os.chdir('../..')
    print(os.path.abspath('.')) 

    # recall, precision, moda, modp = matlab_eval(res_fpath, gt_fpath, 'Wildtrack')
    # print(f'matlab eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
    # recall, precision, moda, modp = python_eval(res_fpath, gt_fpath, 'Wildtrack')
    # print(f'python eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')

    recall, precision, moda, modp = evaluate_rcll_prec_moda_modp(res_fpath, gt_fpath, 'Wildtrack')
    print(f'eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
