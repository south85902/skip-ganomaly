from PIL import Image
import cv2
import numpy as np
import os

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def rotate(image, degree):
    return image.rotate(degree, Image.NEAREST, expand=1)

def flip_left_right(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_top_down(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def randomScale(image, jitter, scaled_lower_limit, scaled_upper_limit):
    iw, ih = image.size
    new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * ih)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * iw)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)
    return image

def randomHSV(image, hue, sat, val):
    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1
    image_data *= 255

    # print(new_image)
    image = Image.fromarray(image_data.astype(np.uint8), 'RGB')
    return image

def clahe(image, cliplimit):
    open_cv_image = np.array(image)
    # Convert RGB to BGR
    bgr = open_cv_image[:, :, ::-1].copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=cliplimit)

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    #img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(bgr)

    return im_pil

def makeDir(f):
    import shutil
    try:
        if not os.path.isdir(f):
            os.makedirs(f)
        else:
            shutil.rmtree(f)
            os.makedirs(f)
    except OSError:
        print("Creation of the directory %s failed" % f)
    else:
        print("Successfully created the directory %s " % f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ogf', default='./', help='')
    parser.add_argument('--outf', default='./', help='')
    opt = parser.parse_args()
    ogf = opt.ogf
    outf = opt.outf
    makeDir(outf)
    # 列出指定路徑底下所有檔案(包含資料夾)
    allFileList = os.listdir(ogf + '/')
    for file in allFileList:
        print(file)
        if file[-4:] == '.bmp':
            image_root = ogf + '/' + file[:-4]
            image_saved_path = outf+'/' + file[:-4]
            img = Image.open(image_root+'.bmp')
            img.save(image_saved_path+'.bmp')


            img_flip_left_right = flip_left_right(img)
            img_flip_left_right.save(image_saved_path + '_flip_l_r.bmp')

            img_flip_top_down = flip_top_down(img)
            img_flip_top_down.save(image_saved_path + '_flip_t_d.bmp')

            # img_randScaled = randomScale(img, 0.3, 0.25, 2)
            # img_randScaled.save(image_saved_path + '_scaled.bmp')
            #
