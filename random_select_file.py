# author by LYS 2017/5/24
# for Deep Learning course
'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil

def check_dir(tar_dir):
    if not os.path.isdir(tar_dir):
        os.makedirs(tar_dir)

def copyFile(fileDir):
    # 1
    pathDir = os.listdir(fileDir)

    n = int(47869 * 0.5)
    # 2
    sample = random.sample(pathDir, 0)

    check_dir(tarDir_first)
    check_dir(tarDir_less)
    
    # 3
    for name in pathDir:
        if name in sample:
            shutil.copyfile(fileDir + name, tarDir_first + name)
        else:
            shutil.copyfile(fileDir + name, tarDir_less + name)


if __name__ == '__main__':
    fileDir = "../dataSet/AnomalyDetectionData_train0.5/val/1.abnormal/"
    tarDir_first = '../dataSet/AnomalyDetectionData_train0.5/val/1.abnormal/'
    tarDir_less = '../dataSet/AnomalyDetectionData_train0.5/test/1.abnormal/'
    copyFile(fileDir)