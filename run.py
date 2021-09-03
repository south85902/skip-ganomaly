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

# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_DFR --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --DFR --batchsize 4 --verbose --netg Unet_DFR", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_DFR --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --DFR --batchsize 4 --verbose --netg Unet_DFR", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_DFR_Unet_DFR", shell=True)

# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --batchsize 64 --verbose", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --batchsize 64 --verbose", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1", shell=True)
#
# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1_ngf32_ndf32 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --batchsize 64 --verbose --ngf 32 --ndf 32", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1_ngf32_ndf32 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --batchsize 64 --verbose --ngf 32 --ndf 32", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1_ngf32_ndf32", shell=True)
#
# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l2 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l2 --batchsize 64 --verbose", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l2 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l2 --load_weights --batchsize 64 --verbose", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l2", shell=True)
#
# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l2_ngf32_ndf32 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l2 --batchsize 64 --verbose --ngf 32 --ndf 32", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l2_ngf32_ndf32 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l2 --load_weights --batchsize 64 --verbose --ngf 32 --ndf 32", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l2_ngf32_ndf32", shell=True)

# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1_ngf16_ndf16 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --batchsize 64 --verbose --ngf 16 --ndf 16", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1_ngf16_ndf16 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --batchsize 64 --verbose --ngf 16 --ndf 16", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --name AnomalyDetectionData_train0.1_l1_ngf16_ndf16 --phase val", shell=True)
#
# return_code = subprocess.call("python train.py --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_l1_ngf32_ndf32 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase train --l_con l1 --batchsize 64 --verbose --ngf 32 --ndf 32", shell=True)
# return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_l1_ngf32_ndf32 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l1 --load_weights --batchsize 64 --verbose --ngf 32 --ndf 32", shell=True)
# return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.5 --name AnomalyDetectionData_train0.5_l1_ngf32_ndf32 --phase val", shell=True)

def testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks):
    return_code = subprocess.call(
        "python test.py --dataset %s --name %s --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l2 --load_weights --batchsize %d --verbose %s %s %s %s %s %s %s" % (
        dataset, name, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks), shell=True)
    return_code = subprocess.call("python draw_distribute.py --dataset %s --name %s --phase val" % (dataset, name),
                                  shell=True)
    return_code = subprocess.call(
        "python eval.py --dataset %s --name %s --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights %s %s %s %s %s %s %s" % (
        dataset, name, dfr, netg, l_con, discriminator, ndf, ngf, ks), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_abn_error.zip ./output/%s/val/images_abn_error/" % (name, name), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_nor_error.zip ./output/%s/val/images_nor_error/" % (name, name), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_all.zip ./output/%s/val/images_all/" % (name, name), shell=True)

try:
    #return_code = subprocess.call("python test.py --dataset AnomalyDetectionData_train0.1 --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l2 --load_weights --batchsize 64 --verbose", shell=True)
    #return_code = subprocess.call("python draw_distribute.py --dataset AnomalyDetectionData_train0.1 --phase val", shell=True)
    return_code = subprocess.call("python eval.py --dataset AnomalyDetectionData_train0.1 --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l2 --save_test_images --load_weights", shell=True)
    return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_abn_error_train0.1.zip ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_abn_error/", shell=True)
    return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_nor_error_train0.1.zip ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_nor_error/", shell=True)
    return_code = subprocess.call("zip -r ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_all.zip ./output/skipganomaly/AnomalyDetectionData_train0.1/val/images_all/", shell=True)
except:
    from line_notify import sent_message
    sent_message('error AnomalyDetectionData_train0.1')

try:
    dataset = 'AnomalyDetectionData_train0.5'
    name = 'AnomalyDetectionData_train0.5'
    batchsize = 64
    return_code = subprocess.call("python test.py --dataset %s --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l2 --load_weights --batchsize %d --verbose" % (dataset, batchsize), shell=True)
    return_code = subprocess.call("python draw_distribute.py --dataset %s --phase val" % dataset, shell=True)
    return_code = subprocess.call("python eval.py --dataset %s --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --l_con l2" % dataset, shell=True)
    return_code = subprocess.call("zip -r ./output/skipganomaly/%s/val/images_abn_error_train0.1.zip ./output/skipganomaly/%s/val/images_abn_error/" % (name, name), shell=True)
    return_code = subprocess.call("zip -r ./output/skipganomaly/%s/val/images_nor_error_train0.1.zip ./output/skipganomaly/%s/val/images_nor_error/" % (name, name), shell=True)
    return_code = subprocess.call("zip -r ./output/skipganomaly/%s/val/images_all.zip ./output/skipganomaly/%s/val/images_all/" % (name, name), shell=True)
except:
    from line_notify import sent_message
    sent_message('error AnomalyDetectionData_train0.5')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_ssim_k3'
    batchsize = 64
    return_code = subprocess.call("python test.py --dataset %s --name %s --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l2 --load_weights --batchsize %d --verbose" % (dataset, name, batchsize), shell=True)
    return_code = subprocess.call("python draw_distribute.py --dataset %s --name %s --phase val" % (dataset, name), shell=True)
    return_code = subprocess.call("python eval.py --dataset %s --name %s --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --l_con l2" % (dataset, name), shell=True)
    return_code = subprocess.call("zip -r ./output/%s/val/images_abn_error_train0.1.zip ./output/%s/val/images_abn_error/" % (name, name), shell=True)
    return_code = subprocess.call("zip -r ./output/%s/val/images_nor_error_train0.1.zip ./output/%s/val/images_nor_error/" % (name, name), shell=True)
except:
    from line_notify import sent_message
    sent_message('error AnomalyDetectionData_train0.1_ssim_k3')

try:
    dataset = 'AnomalyDetectionData_train0.5'
    name = 'AnomalyDetectionData_train0.5_ssim_k3'
    batchsize = 64
    dfr = ''
    netg = ''
    l_con = ''
    discriminator = ''
    ndf = ''
    ngf = ''
    return_code = subprocess.call("python test.py --dataset %s --name %s --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --l_con l2 --load_weights --batchsize %d --verbose" % (dataset, name, batchsize), shell=True)
    return_code = subprocess.call("python draw_distribute.py --dataset %s --name %s --phase val" % (dataset, name), shell=True)
    return_code = subprocess.call("python eval.py --dataset %s --name %s --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --l_con l2" % (dataset, name), shell=True)
    return_code = subprocess.call("zip -r ./output/%s/val/images_abn_error_train0.1.zip ./output/%s/val/images_abn_error/" % (name, name), shell=True)
    return_code = subprocess.call("zip -r ./output/%s/val/images_nor_error_train0.1.zip ./output/%s/val/images_nor_error/" % (name, name), shell=True)
except:
    from line_notify import sent_message
    sent_message('error AnomalyDetectionData_train0.5_ssim_k3')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_DFR'
    batchsize = 8
    dfr = '--DFR'
    netg = ''
    l_con = '--l_con l2'
    discriminator = ''
    ndf = ''
    ngf = ''
    ks = ''
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message
    sent_message('error AnomalyDetectionData_train0.5_DFR')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_DFR_CAE_noDis'
    batchsize = 4
    dfr = '--DFR'
    netg = '--netg CAE'
    l_con = '--l_con l2'
    discriminator = '--no_discriminator'
    ndf = ''
    ngf = ''
    ks = ''
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_DFR_CAE_noDis')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001'
    batchsize = 4
    dfr = '--DFR'
    netg = '--netg CAE'
    l_con = '--l_con l2'
    discriminator = ''
    ndf = ''
    ngf = ''
    ks = ''
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_DFR_Unet_noDis'
    batchsize = 4
    dfr = '--DFR'
    netg = '--netg Unet'
    l_con = '--l_con l2'
    discriminator = '--no_discriminator'
    ndf = ''
    ngf = ''
    ks = ''
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, '')
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_DFR_Unet_noDis')

# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_Unet'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg Unet'
#     l_con = '--l_con l2'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_DFR_Unet')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_DFR_Unet_DFR'
    batchsize = 4
    dfr = '--DFR'
    netg = '--netg Unet_DFR'
    l_con = '--l_con l2'
    discriminator = ''
    ndf = ''
    ngf = ''
    ks = '--ks 3'
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_DFR_Unet_DFR')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_l1'
    batchsize = 64
    dfr = ''
    netg = ''
    l_con = '--l_con l1'
    discriminator = ''
    ndf = ''
    ngf = ''
    ks = '--ks 3'
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_l1')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_l1_ngf32_ndf32'
    batchsize = 64
    dfr = ''
    netg = ''
    l_con = '--l_con l1'
    discriminator = ''
    ndf = '--ndf 32'
    ngf = '--ngf 32'
    ks = '--ks 3'
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_l1_ngf32_ndf32')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_l2'
    batchsize = 64
    dfr = ''
    netg = ''
    l_con = '--l_con l2'
    discriminator = ''
    ndf = ''
    ngf = ''
    ks = '--ks 3'
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_l2')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_l2_ngf32_ndf32'
    batchsize = 64
    dfr = ''
    netg = ''
    l_con = '--l_con l2'
    discriminator = ''
    ndf = '--ndf 32'
    ngf = '--ngf 32'
    ks = '--ks 3'
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_l2_ngf32_ndf32')

try:
    dataset = 'AnomalyDetectionData_train0.1'
    name = 'AnomalyDetectionData_train0.1_l1_ngf16_ndf16'
    batchsize = 64
    dfr = ''
    netg = ''
    l_con = '--l_con l1'
    discriminator = ''
    ndf = '--ndf 16'
    ngf = '--ngf 16'
    ks = '--ks 3'
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message

    sent_message('error AnomalyDetectionData_train0.1_l1_ngf16_ndf16')

try:
    dataset = 'AnomalyDetectionData_train0.5'
    name = 'AnomalyDetectionData_train0.5_l1_ngf32_ndf32'
    batchsize = 64
    dfr = ''
    netg = ''
    l_con = '--l_con l1'
    discriminator = ''
    ndf = '--ndf 32'
    ngf = '--ngf 32'
    ks = '--ks 3'
    testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
except:
    from line_notify import sent_message
    sent_message('AnomalyDetectionData_train0.5_l1_ngf32_ndf32')

print('done')