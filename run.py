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

# ==================================================all test need code here ==========================================================================
def testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks):
    return_code = subprocess.call(
        "python test.py --dataset %s --name %s --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --load_weights --batchsize %d --verbose %s %s %s %s %s %s %s" % (
        dataset, name, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks), shell=True)
    return_code = subprocess.call("python draw_distribute.py --dataset %s --name %s --phase val" % (dataset, name),
                                  shell=True)
    return_code = subprocess.call(
        "python eval.py --dataset %s --name %s --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --batchsize %d %s %s %s %s %s %s %s" % (
        dataset, name, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_abn_error.zip ./output/%s/val/images_abn_error/" % (name, name), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_nor_error.zip ./output/%s/val/images_nor_error/" % (name, name), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_all.zip ./output/%s/val/images_all/" % (name, name), shell=True)

def testAndeval_tempfornoname(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks):
    return_code = subprocess.call(
        "python test.py --dataset %s %s --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --load_weights --batchsize %d --verbose %s %s %s %s %s %s %s" % (
        dataset, name, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks), shell=True)
    return_code = subprocess.call("python draw_distribute.py --dataset %s %s --phase val" % (dataset, name),
                                  shell=True)
    return_code = subprocess.call(
        "python eval.py --dataset %s %s --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --batchsize %d %s %s %s %s %s %s %s" % (
        dataset, name, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/skipganomaly/%s/val/images_abn_error.zip ./output/skipganomaly/%s/val/images_abn_error/" % (dataset, dataset), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/skipganomaly/%s/val/images_nor_error.zip ./output/skipganomaly/%s/val/images_nor_error/" % (dataset, dataset), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/skipganomaly/%s/val/images_all.zip ./output/skipganomaly/%s/val/images_all/" % (dataset, dataset), shell=True)

def train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks,wgan):
    return_code = subprocess.call(
        "python train.py --dataset %s --name %s --isize 128 --niter 50 --display --save_image_freq 1 --print_freq 1 --phase train --batchsize %d --verbose %s %s %s %s %s %s %s %s" % (
            dataset, name, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan), shell=True)

# def testAndeval_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned):
#     return_code = subprocess.call(
#         "python test.py --dataset %s --name %s --isize 128 --niter 100 --display --save_image_freq 1 --print_freq 1 --phase val --load_weights --batchsize %d --verbose %s %s %s %s %s %s %s %s" % (
#         dataset, name, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned), shell=True)
#     return_code = subprocess.call("python draw_distribute.py --dataset %s --name %s --phase val" % (dataset, name),
#                                   shell=True)
#     return_code = subprocess.call(
#         "python eval.py --dataset %s --name %s --isize 128 --niter 1 --display --save_image_freq 1 --print_freq 1 --phase val --save_test_images --load_weights --batchsize %d %s %s %s %s %s %s %s %s" % (
#         dataset, name, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned), shell=True)
#     return_code = subprocess.call(
#         "zip -r ./output/%s/val/images_abn_error.zip ./output/%s/val/images_abn_error/" % (name, name), shell=True)
#     return_code = subprocess.call(
#         "zip -r ./output/%s/val/images_nor_error.zip ./output/%s/val/images_nor_error/" % (name, name), shell=True)
#     return_code = subprocess.call(
#         "zip -r ./output/%s/val/images_all.zip ./output/%s/val/images_all/" % (name, name), shell=True)
#
# def train_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned, niter):
#     return_code = subprocess.call(
#         "python train.py --dataset %s --name %s --isize 128 %s --display --save_image_freq 1 --print_freq 1 --phase train --batchsize %d --verbose %s %s %s %s %s %s %s %s %s" % (
#             dataset, name, niter, batchsize, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned), shell=True)

def testAndeval_eft(cmds):
    return_code = subprocess.call(
        "python test.py %s %s %s %s --display --save_image_freq 1 --print_freq 1 --load_weights %s --verbose %s %s %s %s %s %s %s %s %s %s" % (
        cmds['dataset'], cmds['name'], cmds['isize'], cmds['phase'], cmds['batchsize'], cmds['dfr'], cmds['netg'], cmds['l_con'], cmds['discriminator'], cmds['ndf'], cmds['ngf'], cmds['ks'], cmds['extractor_fine_tuned'], cmds['no_padding'], cmds['resize_same']), shell=True)
    return_code = subprocess.call("python draw_distribute.py %s %s %s" % (cmds['dataset'], cmds['name'], cmds['phase']),
                                  shell=True)
    return_code = subprocess.call(
        "python eval.py %s %s %s --niter 1 --display --save_image_freq 1 --print_freq 1 %s --save_test_images --load_weights %s %s %s %s %s %s %s %s %s %s %s" % (
        cmds['dataset'], cmds['name'], cmds['isize'], cmds['phase'], cmds['batchsize'], cmds['dfr'], cmds['netg'], cmds['l_con'], cmds['discriminator'], cmds['ndf'], cmds['ngf'], cmds['ks'], cmds['extractor_fine_tuned'], cmds['no_padding'], cmds['resize_same']), shell=True)

    name = cmds['name']
    name = name.split(' ', 1)
    name = name[1]
    # del the exit file
    return_code = subprocess.call(
        "rm -r ./output/%s/val/images_abn_error.zip" % (name), shell=True)
    return_code = subprocess.call(
        "rm -r ./output/%s/val/images_nor_error.zip" % (name), shell=True)
    return_code = subprocess.call(
        "rm -r ./output/%s/val/images_all.zip" % (name), shell=True)

    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_abn_error.zip ./output/%s/val/images_abn_error/" % (name, name), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_nor_error.zip ./output/%s/val/images_nor_error/" % (name, name), shell=True)
    return_code = subprocess.call(
        "zip -r ./output/%s/val/images_all.zip ./output/%s/val/images_all/" % (name, name), shell=True)

def train_eft(cmds):
    return_code = subprocess.call(
        "python train.py %s %s %s %s --display --save_image_freq 1 --print_freq 1 --phase train %s --verbose %s %s %s %s %s %s %s %s %s %s %s %s" % (
            cmds['dataset'], cmds['name'], cmds['isize'], cmds['niter'], cmds['phase'], cmds['batchsize'], cmds['dfr'], cmds['netg'], cmds['l_con'], cmds['discriminator'], cmds['ndf'], cmds['ngf'], cmds['ks'], cmds['wgan'], cmds['extractor_fine_tuned'], cmds['no_padding'], cmds['resize_same']), shell=True)

def train_vgg19(cmds):
    return_code = subprocess.call(
        "python train_vgg19.py %s %s %s %s %s %s --verbose %s %s %s %s" % (
            cmds['dataroot'], cmds['name'], cmds['isize'], cmds['niter'], cmds['lr'], cmds['model_name'], cmds['device'], cmds['outf'], cmds['batchsize'], cmds['weight_name']), shell=True)

# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = ''
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l2'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     #train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan)
#     testAndeval_tempfornoname(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1')


# try:
#     dataset = 'AnomalyDetectionData_train0.5'
#     name = ''
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l2'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     #train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan)
#     testAndeval_tempfornoname(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5')

# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_ssim_k3'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con ssim'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     #train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan)
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_ssim_k3')

# try:
#     dataset = 'AnomalyDetectionData_train0.5'
#     name = 'AnomalyDetectionData_train0.5_ssim_k3'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con ssim'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     #train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan)
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_ssim_k3')

# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR'
#     batchsize = 8
#     dfr = '--DFR'
#     netg = ''
#     l_con = '--l_con l2'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('error AnomalyDetectionData_train0.5_DFR')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_CAE_noDis'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l2'
#     discriminator = '--no_discriminator'
#     ndf = ''
#     ngf = ''
#     ks = ''
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_DFR_CAE_noDis')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l2'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_DFR_CAE_lr0.00001')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_Unet_noDis'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg Unet'
#     l_con = '--l_con l2'
#     discriminator = '--no_discriminator'
#     ndf = ''
#     ngf = ''
#     ks = ''
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, '')
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_DFR_Unet_noDis')
#
# # try:
# #     dataset = 'AnomalyDetectionData_train0.1'
# #     name = 'AnomalyDetectionData_train0.1_DFR_Unet'
# #     batchsize = 4
# #     dfr = '--DFR'
# #     netg = '--netg Unet'
# #     l_con = '--l_con l2'
# #     discriminator = ''
# #     ndf = ''
# #     ngf = ''
# #     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf)
# # except:
# #     from line_notify import sent_message
# #
# #     sent_message('error AnomalyDetectionData_train0.1_DFR_Unet')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_Unet_DFR'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg Unet_DFR'
#     l_con = '--l_con l2'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = '--ks 3'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_DFR_Unet_DFR')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_l1'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = '--ks 3'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_l1')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_l1_ngf32_ndf32'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = '--ndf 32'
#     ngf = '--ngf 32'
#     ks = '--ks 3'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_l1_ngf32_ndf32')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_l2'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l2'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = '--ks 3'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_l2')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_l2_ngf32_ndf32'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l2'
#     discriminator = ''
#     ndf = '--ndf 32'
#     ngf = '--ngf 32'
#     ks = '--ks 3'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_l2_ngf32_ndf32')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_l1_ngf16_ndf16'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = '--ndf 16'
#     ngf = '--ngf 16'
#     ks = '--ks 3'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#
#     sent_message('error AnomalyDetectionData_train0.1_l1_ngf16_ndf16')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.5'
#     name = 'AnomalyDetectionData_train0.5_l1_ngf32_ndf32'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = '--ndf 32'
#     ngf = '--ngf 32'
#     ks = '--ks 3'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_l1_ngf32_ndf32')

# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_CAE_ndf16'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = '--ndf 16'
#     ngf = ''
#     ks = ''
#     #train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
#     l_con = '--l_con l2'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_DFR_CAE_ndf16')

# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_CAE_wgan'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = '--WGAN'
#     train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan)
#     l_con = '--l_con l2'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_DFR_CAE_wgan')

# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_CAE_ndf8'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = '--ndf 8'
#     ngf = ''
#     ks = ''
#     wgan = ''
#     train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan)
#     l_con = '--l_con l2'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_DFR_CAE_ndf8')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_CAE_ep50'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan)
#     l_con = '--l_con l2'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_DFR_CAE_ep50')
#
# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_DFR_Unet'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg Unet'
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     train(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan)
#     l_con = '--l_con l2'
#     testAndeval(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_DFR_Unet')

# try:
#     dataset = 'AnomalyDetectionData_train0.1_vgg'
#     name = 'AnomalyDetectionData_train0.1_DFR_CAE_noDis_eft'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l1'
#     discriminator = '--no_discriminator'
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     extractor_fine_tuned = '--extractor_fine_tuned'
#     #train_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned)
#     l_con = '--l_con l2'
#     testAndeval_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_DFR_CAE_noDis_eft')

# try:
#     dataset = 'AnomalyDetectionData_newdata_train0.1_vgg'
#     name = 'AnomalyDetectionData_newdata_train0.1'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     extractor_fine_tuned = ''
#     niter = '--niter 300'
#     train_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned, niter)
#     l_con = '--l_con l2'
#     testAndeval_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.1')

# try:
#     dataset = 'AnomalyDetectionData_newdata_train0.1'
#     name = 'AnomalyDetectionData_newdata_train0.1_DFR_CAE'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l1'
#     discriminator = '--no_discriminator'
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     extractor_fine_tuned = ''
#     niter = '--niter 300'
#     train_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned, niter)
#     l_con = '--l_con l2'
#     testAndeval_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.1_DFR_CAE')

# try:
#     dataset = 'AnomalyDetectionData_newdata_train0.1_vgg'
#     name = 'AnomalyDetectionData_newdata_train0.1_DFR_CAE_noDis_eft'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l1'
#     discriminator = '--no_discriminator'
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     extractor_fine_tuned = '--extractor_fine_tuned'
#     niter = '--niter 300'
#     train_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned, niter)
#     l_con = '--l_con l2'
#     testAndeval_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.1_DFR_CAE_noDis_eft')

# try:
#     dataset = 'AnomalyDetectionData_newdata_train0.5'
#     name = 'AnomalyDetectionData_newdata_train0.5'
#     batchsize = 64
#     dfr = ''
#     netg = ''
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     extractor_fine_tuned = ''
#     niter = '--niter 300'
#     # train_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned, niter)
#     l_con = '--l_con l2'
#     testAndeval_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.5')

# try:
#     dataset = 'AnomalyDetectionData_newdata_train0.5'
#     name = 'AnomalyDetectionData_newdata_train0.5_DFR_CAE_noDis'
#     batchsize = 4
#     dfr = '--DFR'
#     netg = '--netg CAE'
#     l_con = '--l_con l1'
#     discriminator = '--no_discriminator'
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     extractor_fine_tuned = '--extractor_fine_tuned'
#     niter = '--niter 300'
#     train_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned, niter)
#     l_con = '--l_con l2'
#     testAndeval_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.5_DFR_CAE_noDis')


# try:
#     dataset = 'AnomalyDetectionData_train0.1'
#     name = 'AnomalyDetectionData_train0.1_Unet_noSkipConnection'
#     batchsize = 64
#     dfr = ''
#     netg = '--netg Unet_noSkipConnection'
#     l_con = '--l_con l1'
#     discriminator = ''
#     ndf = ''
#     ngf = ''
#     ks = ''
#     wgan = ''
#     extractor_fine_tuned = ''
#     niter = '--niter 100'
#     train_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, wgan, extractor_fine_tuned, niter)
#     l_con = '--l_con l2'
#     testAndeval_eft(dataset, batchsize, name, dfr, netg, l_con, discriminator, ndf, ngf, ks, extractor_fine_tuned)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection')

# cmds['dataset'], cmds['name'], cmds['isize'], cmds['niter'], cmds['phase'], cmds['batchsize'], cmds['dfr'], cmds['netg'], cmds['l_con'],
#  cmds['discriminator'], cmds['ndf'], cmds['ngf'], cmds['ks'], cmds['wgan'], cmds['extractor_fine_tuned']

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_noSkipConnection_noPadding'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = '--no_padding'
#     # train_eft(cmd)
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection_no_padding')
#
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_Unet_noSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     # train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_Unet_noSkipConnection')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     # train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_noSkipConnection_pad128'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     #testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection_pad128')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_Unet_noSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_Unet_noSkipConnection')

    
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_DFR_Unet_noSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_DFR_Unet_noSkipConnection')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_DFR_Unet_noSkipConnection_noDis'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_DFR_Unet_noSkipConnection_noDis')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_DFR_Unet'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg Unet'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_DFR_Unet')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_DFR_Unet_noDis'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg Unet'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_DFR_Unet')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_DFR_CAE'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_DFR_CAE')
#
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_DFR_CAE_noDis'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_DFR_CAE_noDis')

# au
#return_code = subprocess.call("python augument.py --ogf ../train_vgg/1.abnormal --outf ./train_vgg_au/1.abnormal", shell=True)

# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_train0.1_vgg/train_vgg_au'
#     cmd['name'] = '--name vgg_weights'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_DFR_CAE_noDis_eft_au'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 70'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_train0.1_DFR_CAE_noDis_eft_au error')


# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_noSkipConnection_ssim'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con ssim'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con ssim'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection_ssim')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_Unet_noSkipConnection_ssim'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con ssim'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con ssim'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_train0.5_Unet_noSkipConnection_ssim')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_ssim'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con ssim'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con ssim'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_ssim')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_Unet_noSkipConnection_ssim'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con ssim'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con ssim'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_Unet_noSkipConnection_ssim')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_DFR_Unet_noDis_eft_au'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 70'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg Unet'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_train0.1_DFR_Unet_noDis_eft_au error')

# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_newdata_train0.5_vgg/train_vgg_au'
#     cmd['name'] = '--name vgg_weights_newData'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_DFR_CAE_noDis_eft_au'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 70'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.5_DFR_CAE_noDis_eft_au error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_fewSkipConnection_del_some_file'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 80'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_fewSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_fewSkipConnection error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_del_some_file_ssim_k17'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 80'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con ssim'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l1'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_del_some_file_ssim_k17 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_del_some_file_ssiml1_k11'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 80'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con ssiml1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_del_some_file_ssim_k11 error')


# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_del_some_file_ssiml1_k11_ep300'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con ssiml1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con ssim'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_del_some_file_ssim_k11_ep300 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_Unet_noSkipConnection_del_some_file_ssiml1_k11_ep300'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con ssiml1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con ssim'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.9_Unet_noSkipConnection_del_some_file_ssim_k11_ep300 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_Unet_fewSkipConnection_2'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_fewSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_Unet_fewSkipConnection error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_fewSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_fewSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_train0.1_Unet_fewSkipConnection error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_Unet_fewSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_fewSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_train0.5_Unet_fewSkipConnection error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_del_some_file_ep300'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_del_some_file_ep300 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_ep300'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.9_ep300 error')


# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_noSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_Unet_noSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_Unet_noSkipConnection error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_Unet_noSkipConnection error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     #train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_Unet_fewSkipConnection'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_fewSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_Unet_fewSkipConnection')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_DFR_CAE_noDis_eft_au'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.5_DFR_CAE_noDis_eft_au')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_wgan'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = ''
#     cmd['wgan'] = '--WGAN'
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_wgan error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_res_2'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 32'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection_res'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.5_Unet_noSkipConnection_res_2 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.9'
#     cmd['name'] = '--name AnomalyDetectionData_newdata_train0.9_Unet_noSkipConnection_res_2'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 300'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 32'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection_res'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('AnomalyDetectionData_newdata_train0.9_Unet_noSkipConnection_res_2 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_noSkipConnection_res_2'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 32'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection_res'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection_res_2 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_Unet_noSkipConnection_res_2'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 32'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection_res'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     l_con = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_Unet_noSkipConnection_res_2 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_noSkipConnection_resize_same_w16'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     #train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection_resize_same_w16 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_noSkipConnection_resize_same_w32_delSomeConv'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l1'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection_resize_same_w32_delSomeConv error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_Unet_noSkipConnection_res_leakyRelu_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection_res'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_Unet_noSkipConnection_res_leakyRelu_resize_same_w32 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_Unet_noSkipConnection_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_Unet_noSkipConnection_resize_same_w32 error')


# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_balance_vgg/train_vgg_au'
#     cmd['name'] = '--name vgg_weights'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['weight_name'] = '--weight_name train5404_0.5_au'
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_balance_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_balance_vgg_Unet_noSkipConnection_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_balance_vgg_Unet_noSkipConnection_resize_same_w32 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_balance_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_balance_vgg_Unet_noSkipConnection_res_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection_res'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_balance_vgg_Unet_noSkipConnection_res_resize_same_w32 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_balance_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_balance_vgg_DFR_CAE_noDis_eft_au_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_balance_vgg_DFR_CAE_noDis_eft_au_resize_same_w32 error')


# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train5404_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train5404_vgg_Unet_noSkipConnection_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     # train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train5404_vgg_Unet_noSkipConnection_resize_same_w32 error')

# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_train5404_vgg/train_vgg_0.5_au'
#     cmd['name'] = '--name vgg_weights'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['weight_name'] = ''
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train5404_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train5404_vgg_DFR_CAE_noDis_eft_au0.5_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     # train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train5404_vgg_DFR_CAE_noDis_eft_au0.5_resize_same_w32 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_balance_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_balance_vgg_DFR_CAE_noDis_eft_au_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     # train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_balance_vgg_DFR_CAE_noDis_eft_au_resize_same_w32 error')
#
# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_train5404_vgg/train_vgg_all_au'
#     cmd['name'] = '--name vgg_weights'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['weight_name'] = '--weight_name _train5404_all_au'
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')


# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_train0.1_vgg/train_vgg_0.5_au'
#     cmd['name'] = '--name vgg_weights'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['weight_name'] = ''
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_vgg_DFR_CAE_noDis_eft_au0.5_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     # train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_vgg_DFR_CAE_noDis_eft_au0.5_resize_same_w32 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train0.1_vgg_Unet_fewSkipConnection_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_fewSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.1_vgg_Unet_fewSkipConnection_resize_same_w32 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_vgg_Unet_noSkipConnection_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_vgg_Unet_noSkipConnection_resize_same_w32 error')
#
# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_vgg_Unet_fewSkipConnection_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 100'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 64'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_fewSkipConnection'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_vgg_Unet_fewSkipConnection_resize_same_w32 error')
#
# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_train0.5_vgg/train_vgg_0.5_au'
#     cmd['name'] = '--name vgg_weights'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['weight_name'] = ''
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5_vgg'
#     cmd['name'] = '--name AnomalyDetectionData_train0.5_vgg_DFR_CAE_noDis_eft_au0.5_resize_same_w32'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 4'
#     cmd['dfr'] = '--DFR'
#     cmd['netg'] = '--netg CAE'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = '--no_discriminator'
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = '--extractor_fine_tuned'
#     cmd['no_padding'] = ''
#     cmd['resize_same'] = '--resize_same'
#     # train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#     sent_message('AnomalyDetectionData_train0.5_vgg_DFR_CAE_noDis_eft_au0.5_resize_same_w32 error')

try:
    cmd = {}
    cmd['dataset'] = '--dataset AnomalyDetectionData_train0.1_vgg'
    cmd['name'] = '--name AnomalyDetectionData_train0.1_vgg_Unet_twoSkipConnection_resize_same_w32'
    cmd['isize'] = '--isize 128'
    cmd['niter'] = '--niter 100'
    cmd['phase'] = '--phase train'
    cmd['batchsize'] = '--batchsize 64'
    cmd['dfr'] = ''
    cmd['netg'] = '--netg Unet_fewSkipConnection'
    cmd['l_con'] = '--l_con l1'
    cmd['discriminator'] = ''
    cmd['ndf'] = ''
    cmd['ngf'] = ''
    cmd['ks'] = '--ks 3'
    cmd['wgan'] = ''
    cmd['extractor_fine_tuned'] = ''
    cmd['no_padding'] = ''
    cmd['resize_same'] = '--resize_same'
    train_eft(cmd)

    cmd['l_con'] = '--l_con l2'
    cmd['phase'] = '--phase val'
    testAndeval_eft(cmd)
except:
    from line_notify import sent_message
    sent_message('AnomalyDetectionData_train0.1_vgg_Unet_twoSkipConnection_resize_same_w32 error')

try:
    cmd = {}
    cmd['dataset'] = '--dataset AnomalyDetectionData_train0.5_vgg'
    cmd['name'] = '--name AnomalyDetectionData_train0.5_vgg_Unet_twoSkipConnection_resize_same_w32'
    cmd['isize'] = '--isize 128'
    cmd['niter'] = '--niter 100'
    cmd['phase'] = '--phase train'
    cmd['batchsize'] = '--batchsize 64'
    cmd['dfr'] = ''
    cmd['netg'] = '--netg Unet_fewSkipConnection'
    cmd['l_con'] = '--l_con l1'
    cmd['discriminator'] = ''
    cmd['ndf'] = ''
    cmd['ngf'] = ''
    cmd['ks'] = '--ks 3'
    cmd['wgan'] = ''
    cmd['extractor_fine_tuned'] = ''
    cmd['no_padding'] = ''
    cmd['resize_same'] = '--resize_same'
    train_eft(cmd)

    cmd['l_con'] = '--l_con l2'
    cmd['phase'] = '--phase val'
    testAndeval_eft(cmd)
except:
    from line_notify import sent_message
    sent_message('AnomalyDetectionData_train0.5_vgg_Unet_twoSkipConnection_resize_same_w32 error')

# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_2PIN_train0.5_vgg/train_vgg_all_au'
#     cmd['name'] = '--name vgg_weights'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['weight_name'] = '--weight_name _2PIN_train0.5_all_au'
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')
#
# try:
#     cmd = {}
#     cmd['dataroot'] = '--dataroot ../dataSet/AnomalyDetectionData_2PIN_train0.5_vgg/train_vgg_0.5_au'
#     cmd['name'] = '--name vgg_weights'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 50'
#     cmd['lr'] = '--lr 0.0001'
#     cmd['model_name'] = '--model_name vgg'
#     cmd['weight_name'] = '--weight_name _2PIN_train0.5_0.5_au'
#     cmd['device'] = ''
#     cmd['outf'] = '--outf ../vgg_weights'
#     cmd['batchsize'] = '--batchsize 16'
#     train_vgg19(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('train vgg19 error')

# try:
#     cmd = {}
#     cmd['dataset'] = '--dataset AnomalyDetectionData_newdata_train0.5'
#     cmd['name'] = '--name del'
#     cmd['isize'] = '--isize 128'
#     cmd['niter'] = '--niter 3'
#     cmd['phase'] = '--phase train'
#     cmd['batchsize'] = '--batchsize 32'
#     cmd['dfr'] = ''
#     cmd['netg'] = '--netg Unet_noSkipConnection_res'
#     cmd['l_con'] = '--l_con l1'
#     cmd['discriminator'] = ''
#     cmd['ndf'] = ''
#     cmd['ngf'] = ''
#     cmd['ks'] = '--ks 3'
#     cmd['wgan'] = ''
#     cmd['extractor_fine_tuned'] = ''
#     cmd['no_padding'] = ''
#     train_eft(cmd)
#
#     cmd['l_con'] = '--l_con l2'
#     cmd['phase'] = '--phase val'
#     #testAndeval_eft(cmd)
# except:
#     from line_notify import sent_message
#
#     sent_message('del error')
print('done')