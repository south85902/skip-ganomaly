import os
import subprocess
return_code = subprocess.call("python train.py --dataset AnomalyDetectionData --name AnomalyDetectionData_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con ssim", shell=True)
return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con ssim", shell=True)
return_code = subprocess.call("python train.py  --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con ssim", shell=True)
#return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_ssim --dataroot ../dataSet/AnomalyDetectionData --name test1 --isize 64 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)

print('done')