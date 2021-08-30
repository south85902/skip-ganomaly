import os
import subprocess
#return_code = subprocess.call("python train.py --dataset AnomalyDetectionData --name AnomalyDetectionData_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con ssim ", shell=True)
#return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con ssim", shell=True)
#return_code = subprocess.call("python train.py  --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con ssim", shell=True)

#return_code = subprocess.call("python test.py --dataset AnomalyDetectionData --name AnomalyDetectionData_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)
#return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData --name AnomalyDetectionData_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)

#return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)
#return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)

#return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)
#return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_ssim --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)

#return_code = subprocess.call("python train.py --dataset AnomalyDetectionData --name AnomalyDetectionData_ssim_k3 --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con ssim ", shell=True)
#return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_noNoise --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1", shell=True)
#return_code = subprocess.call("python train.py  --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_ssim_k3 --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con ssim", shell=True)

#return_code = subprocess.call("python test.py --dataset AnomalyDetectionData --name AnomalyDetectionData_ssim_k3 --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)
#return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData --name AnomalyDetectionData_ssim_k3 --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)

# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_noNoise --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --save_test_images --load_weights", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_noNoise --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --save_test_images --load_weights", shell=True)

#return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_ssim_k3 --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)
#return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_ssim_k3 --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con ssim --save_test_images --load_weights", shell=True)

# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --DFR --nc 1472 --batchsize 8", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --nc 1472 --batchsize 8", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --nc 1472 --batchsize 8", shell=True)

# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_CAE_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --DFR --nc 1472 --batchsize 4 --no_discriminator --netg CAE", shell=True)
#return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_CAE_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --nc 1472 --batchsize 4 --no_discriminator --netg CAE", shell=True)
#return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_CAE_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --nc 1472 --batchsize 4 --no_discriminator --netg CAE", shell=True)

# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --DFR --nc 1472 --batchsize 4 --netg CAE --verbose", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --nc 1472 --batchsize 4 --netg CAE --verbose", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --nc 1472 --batchsize 4 --netg CAE --verbose", shell=True)

return_code = subprocess.call("python test.py --dataset AnomalyDetectionData --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --min 0.0012 --max 0.1316 --threshold 0.0431", shell=True)
return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --min 0.0002 --max 0.0633 --threshold 0.0424", shell=True)
return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.5 --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --min 0.000013516 --max 0.0211 --threshold 0.0077", shell=True)

print('done')