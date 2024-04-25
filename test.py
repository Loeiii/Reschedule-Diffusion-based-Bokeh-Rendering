from pathlib import Path
from core.metrics import calc_metric
import argparse
from core.interpolate import interpolate_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/defocus_240424_165954')
    parser.add_argument('-gt', '--gtpath', type=str, default='datasets/validation/FNumber_2')
    
    args = parser.parse_args()
    
    interpolate_result('{}/results'.format(args.path))
    calc_metric("{}/output".format(args.path), args.gtpath)