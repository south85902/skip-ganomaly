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
    model.eval(plot_hist=True, test_set=test_set, min=0.1263, max=0.3753, threshold=0.040829003)
    sent_message('eval done')

if __name__ == '__main__':
    main()
