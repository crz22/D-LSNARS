import os
from libtiff import TIFF
import numpy as np
import yaml

def read_configuration(configuration_path):
    # print(configuration_path)
    assert os.path.exists(configuration_path) and configuration_path.endswith('.yaml')
    with open(configuration_path) as f:
        configure = yaml.load(f, Loader=yaml.Loader)
    return configure

def read_image(image_path):
    images = []
    tif = TIFF.open(image_path, mode='r')
    for image in tif.iter_images():
        images.append(image)
    return np.array(images)

def save_image(img, savepath,filp_flag=False):
    tif = TIFF.open(savepath, mode='w')
    num = img.shape[0]
    if img.max() <= 1:
        img = img*255
    for i in range(num):
        image = img[i]
        if filp_flag:
            image = np.flipud(image)
        tif.write_image(((image).astype(np.uint8)), compression=None)
    tif.close()
    return

def save_log(argsDict,save_path,file_name):
    save_path = save_path+'/'+file_name
    if not os.path.exists(save_path):
        #os.makedirs(save_path)
        os.mkdir(save_path)
    with open(save_path+'/log.txt', 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

def save_loss(save_path,loss_list):
    txt_path = save_path+'/loss.txt'
    with open(txt_path, 'a') as f:
        #Write loss name
        if os.path.getsize(txt_path) == 0:
            for key, value in loss_list.items():
                f.write(key)
                f.write(' ')
            f.write('\n')
        #Write loss value
        for key, value in loss_list.items():
            print(key, value)
            f.write(str(value))
            f.write(' ')
        f.write('\n')
        f.close()

def saveswc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]

    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)
