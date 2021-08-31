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

# return_code = subprocess.call("python eval.py --dataset AnomalyDetectionData --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --min 0.001 --max 0.132 --threshold 0.0431", shell=True)
# return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData/val/images_abn_error_og.zip ./output/skipganomaly/AnomalyDetectionData/val/images_abn_error/", shell=True)
# return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData/val/images_nor_error_og.zip ./output/skipganomaly/AnomalyDetectionData/val/images_nor_error/", shell=True)
#
# return_code = subprocess.call("python eval.py --dataset AnomalyDetectionData_train0.1 --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --min 0.0002 --max 0.0633 --threshold 0.0424", shell=True)
# return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_abn_error_0.1.zip ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_abn_error/", shell=True)
# return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_nor_error_0.1.zip ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_nor_error/", shell=True)
#
# return_code = subprocess.call("python eval.py --dataset AnomalyDetectionData_train0.5 --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --min 0.000013516 --max 0.0211 --threshold 0.0077", shell=True)
# return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData_train0.5/val/images_abn_error_train0.5.zip ./output/skipganomaly/AnomalyDetectionData_train0.5/val/images_abn_error/", shell=True)
# return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData_train0.5/val/images_nor_error_train0.5.zip ./output/skipganomaly/AnomalyDetectionData_train0.5/val/images_nor_error/", shell=True)
#
# return_code = subprocess.call("python eval.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_ssim --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --l_con ssim --min 0.0027 --max 0.1767 --threshold 0.0107406", shell=True)
# return_code = subprocess.call("zip -r ./output/AnomalyDetectionData_train0.1_ssim/val/images_abn_error_train0.1_ssim.zip ./output/AnomalyDetectionData_train0.1_ssim/val/images_abn_error/", shell=True)
# return_code = subprocess.call("zip -r ./output/AnomalyDetectionData_train0.1_ssim/val/images_nor_error_train0.1_ssim.zip ./output/AnomalyDetectionData_train0.1_ssim/val/images_nor_error/", shell=True)
#
# return_code = subprocess.call("python eval.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_CAE_noDis --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --DFR --nc 1472 --batchsize 4 --no_discriminator --netg CAE --min 0.0049 --max 0.6050 --threshold 0.1195507", shell=True)
# return_code = subprocess.call("zip -r ./output/AnomalyDetectionData_train0.1_DFR_CAE_noDis/val/images_abn_error_train0.1_DFR_CAE_noDis.zip ./output/AnomalyDetectionData_train0.1_DFR_CAE_noDis/val/images_abn_error/", shell=True)
# return_code = subprocess.call("zip -r ./output/AnomalyDetectionData_train0.1_DFR_CAE_noDis/val/images_nor_error_train0.1_DFR_CAE_noDis.zip ./output/AnomalyDetectionData_train0.1_DFR_CAE_noDis/val/images_nor_error/", shell=True)
#
# return_code = subprocess.call("python eval.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001 --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --DFR --nc 1472 --batchsize 4 --netg CAE --min 0.1757 --max 1.1719 --threshold 0.1825777", shell=True)
# return_code = subprocess.call("zip -r ./output/AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001/val/images_abn_error_train0.1_DFR_CAE_lr0.00001.zip ./output/AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001/val/images_abn_error/", shell=True)
# return_code = subprocess.call("zip -r ./output/AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001/val/images_nor_error_train0.1_DFR_CAE_lr0.00001.zip ./output/AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001/val/images_nor_error/", shell=True)
#
# return_code = subprocess.call("python eval.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --DFR --nc 1472 --batchsize 8 --min 0.2369 --max 1.3216 --threshold 0.1978176", shell=True)
# return_code = subprocess.call("zip -r ./output/AnomalyDetectionData_train0.1_DFR/val/images_abn_error_train0.1_DFR.zip ./output/AnomalyDetectionData_train0.1_DFR/val/images_abn_error/", shell=True)
# return_code = subprocess.call("zip -r ./output/AnomalyDetectionData_train0.1_DFR/val/images_nor_error_train0.1_DFR.zip ./output/AnomalyDetectionData_train0.1_DFR/val/images_nor_error/", shell=True)
#
# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --DFR --nc 1472 --batchsize 8 --verbose --no_discriminator", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --nc 1472 --batchsize 8  --verbose --no_discriminator", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_noDis --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --nc 1472 --batchsize 8 --verbose --no_discriminator", shell=True)

return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_DFR --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --DFR --batchsize 4 --verbose --netg Unet_DFR", shell=True)
return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_DFR --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --batchsize 4 --verbose --netg Unet_DFR", shell=True)
return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_DFR", shell=True)


print('done')