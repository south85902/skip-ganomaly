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
##
def main():
    """ Testing
    """
    opt = Options().parse()
    data = load_data(opt)
    model = load_model(opt, data)
    test_set = opt.phase
    model.eval(plot_hist=True, test_set=test_set, min=opt.min, max=opt.max, threshold=opt.threshold)
    sent_message('eval done')

if __name__ == '__main__':
    main()
