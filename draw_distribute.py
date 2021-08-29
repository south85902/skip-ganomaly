import matplotlib.pyplot as plt
from options import Options
import pandas as pd
import seaborn as sns
import os
save_path = 'C:/Users/nannan/Desktop/Skip-Ganomaly/skip-ganomaly/output/skipganomaly/AnomalyDetectionData/'
#plt.ion()
# Create data frame for scores and labels.
opt = Options().parse()
save_path = os.path.join(opt.outf, opt.name, opt.phase)
hist = pd.read_csv(os.path.join(save_path, 'histogram.csv'))

# Filter normal and abnormal scores.
abn_scr = hist.loc[hist.labels == 1]['scores']
nrm_scr = hist.loc[hist.labels == 0]['scores']
# print(abn_scr)
# print(nrm_scr)
# abn_scr = abn_scr.cpu()
# nrm_scr = nrm_scr.cpu()
# Create figure and plot the distribution.
#fig, ax = plt.subplots(figsize=(4,4))
abn_scr = abn_scr.values
nrm_scr = nrm_scr.values
#nrm_scr = [1, 2, 3, 4, 5, 6]
#abn_scr = [1, 2, 3, 4, 5, 6, 7, 3, 2, 4, 5]
sns.distplot(nrm_scr, label=r'Normal Scores', kde=False)
sns.distplot(abn_scr, label=r'Abnormal Scores', kde=False)
#plt.scatter(nrm_scr, color='b', linestyle='dotted', label=r'Normal Scores')
#plt.scatter(abn_scr, color='r', linestyle='dotted', label=r'Abnormal Scores')
plt.legend()
#plt.yticks([])
plt.xlabel(r'Anomaly Scores')
#plt.show()
plt.savefig(os.path.join(save_path, 'Anomaly_scores.png'))