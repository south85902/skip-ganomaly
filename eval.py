"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model
from line_notify import sent_message
import yaml
import os
import pandas as pd
import math

##

def get_threshold(opt):
    save_path = os.path.join(opt.outf, opt.name, opt.phase)
    df = pd.read_csv(os.path.join(save_path, 'fpr_tpr_thre.csv'))
    data_num = df.shape[0]
    thre_index = 3
    nor_recall_index = 4
    abn_recall_index = 6
    minimum = 1
    minimum_index = 0
    for i in range(0, data_num):
        if i+1 < data_num:
            if (math.fabs(df.iat[i, abn_recall_index] - df.iat[i, nor_recall_index]) < minimum):
                minimum = math.fabs(df.iat[i, abn_recall_index] - df.iat[i, nor_recall_index])
                minimum_index = i

    return df.iat[minimum_index, thre_index]



def main():
    """ Testing
    """
    opt = Options().parse()
    data = load_data(opt)
    model = load_model(opt, data)
    test_set = opt.phase

    save_path = os.path.join(opt.outf, opt.name, test_set)

    # get value.yaml
    with open(os.path.join(save_path, 'value.yaml'), 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    yaml.dump(data)
    opt.min = float(data['min'])
    opt.max = float(data['max'])
    opt.pixel_min = float(data['pixel_min'])
    opt.pixel_max = float(data['pixel_max'])
    opt.threshold = get_threshold(opt)
    model.eval(plot_hist=True, test_set=test_set, min=opt.min, max=opt.max, pixel_min=opt.pixel_min, pixel_max=opt.pixel_max, threshold=opt.threshold)
    sent_message('eval done')

if __name__ == '__main__':
    main()
