import numpy as np
import yaml
from PIL import Image,ImageEnhance,ImageOps
from skimage import measure
from sklearn.cluster import DBSCAN
import os
from libtiff import TIFF


def read_configuration(configuration_path):
    # print(configuration_path)
    assert os.path.exists(configuration_path) and configuration_path.endswith('.yaml')
    with open(configuration_path) as f:
        configure = yaml.load(f, Loader=yaml.Loader)
    return configure

def readtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""
    tiff = TIFF.open(filepath, mode='r')
    stack = []
    for sample in tiff.iter_images():
        stack.append((sample))  # stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.array(stack)
    tiff.close()

    return out

def writetiff3d(filepath, block):
    try:
        os.remove(filepath)
    except OSError:
        pass

    tiff = TIFF.open(filepath, mode='w')
    # block = np.swapaxes(block, 0, 1)
    # print(block.shape)
    for z in range(block.shape[0]):
        # tiff.write_image(np.flipud(block[:, :, z]), compression=None)
        tiff.write_image(((block[z, :, :]).astype(np.uint8)), compression=None)
    tiff.close()


def contrast(image, level):
    image = Image.fromarray(image.astype('uint8'))
    if level !='auto':
        image = ImageEnhance.Contrast(image).enhance(level)
    else:
        image = ImageOps.autocontrast(image)
    # print('image',image)

    return np.array(image)

def image_to_MIP(image,level=1.6):
    MIP1 = np.uint8(np.max(image, axis=0))
    MIP1 = contrast(MIP1, level=level)

    MIP2 = Image.fromarray(np.uint8(MIP1)).convert('RGB')
    MIP2 = MIP2.resize((256, 256))
    return MIP2

def read_marker(filepath):
    marker = []
    with open(filepath,'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            num = list(map(float,line[0:3]))
            marker.append(num)
    return np.array(marker)

def save_block_marker(filepath, marker):
    with open(filepath, 'w') as f:
        for i in range(marker.shape[0]):
            markerp = [marker[i, 0], marker[i, 1], marker[i, 2], 0, 1, ' ', ' ']

            print('%.3f, %.3f, %.3f, %d, %d, %s, %s' % (markerp[0], markerp[1], markerp[2], markerp[3],
                                                        markerp[4], markerp[5], markerp[6]), file=f)
def save_marker_list(filepath,marker):
    with open(filepath, 'a') as f:
        for i in range(marker.shape[0]):
            markerp = [marker[i, 0], marker[i, 1], marker[i, 2], 0, 1, ' ', ' ']

            print('%.3f, %.3f, %.3f, %d, %d, %s, %s' % (markerp[0], markerp[1], markerp[2], markerp[3],
                                                        markerp[4], markerp[5], markerp[6]), file=f)
def writetiff2d(filepath, block):
    try:
        os.remove(filepath)
    except OSError:
        pass
    tiff = TIFF.open(filepath, mode='w')
    # block = np.swapaxes(block, 0, 1)
#    for z in range(block.shape[2]):
    tiff.write_image((block[:, :]), compression=None)
    tiff.close()

def get_original_position(image_name):
    # eg: x0_y0_z7936.tif
    image_name_rx = image_name[:-4]
    # eg: x0_y0_z7936
    image_name_list = image_name_rx.split("_")
    # eg: ['x0','y0','z7936']
    x = int(image_name_list[0][1:])
    y = int(image_name_list[1][1:])
    z = int(image_name_list[2][1:])
    #print(x,y,z)

    return x,y,z

def savemarker(filepath,image_name,marker):
    original_x,original_y,original_z = get_original_position(image_name)
    with open(os.path.join(filepath,"predict_somalist.marker"), mode='a') as f:
        for i in range(marker.shape[0]):
            #print(marker[i])
            markerp = [marker[i, 0]+original_x, marker[i, 1]+original_y, marker[i, 2]+original_z, 0, 1, ' ', ' ']

            print('%.3f, %.3f, %.3f, %d, %d, %s, %s' % (markerp[0], markerp[1], markerp[2], markerp[3],
                                                        markerp[4], markerp[5], markerp[6]), file=f)

def savecandidateblock(filepath,cadidatelist):
    cadidate_num = len(cadidatelist)
    with open(os.path.join(filepath, "candidate.marker"), mode='a') as f:
        for i in range(cadidate_num):
            x,y,z = get_original_position(cadidatelist[i])
            markerp = [x, y, z, 0, 1, ' ', ' ']
            print('%.3f, %.3f, %.3f, %d, %d, %s, %s' % (markerp[0], markerp[1], markerp[2], markerp[3],
                                                        markerp[4], markerp[5], markerp[6]), file=f)

def connected_component(image, img_name,save_marker_path,need_label=False,scale_factor=1):
    # Mark the input 3D image
    label, num = measure.label(image, connectivity=1, return_num=True)  # 1代表４连通，２代表８连通
    image_size = image.shape
    if os.path.exists(save_marker_path) == 0:
        os.mkdir(save_marker_path)
    if num < 1:
        print("segment fail: ",img_name)
        return
    # obtain soma location
    marker_list = []
    region = measure.regionprops(label)
    num_list = [i for i in range(1, num + 1)]
    for i in num_list:
        if region[i-1].area>1000:
            marker_list.append(region[i - 1].centroid)
            if need_label:
                label[label == i] = 1.0
        else:
            num = num-1
            if need_label:
                label[label == i] = 0
    #print('marker_list', marker_list) #[z,y,x]
    marker_r = np.ones((num, 3)) #[x,y,z]
    for j in range(0, num):  # x和y互换，y变成128-y
        marker_r[j][0] = marker_list[j][2]*scale_factor + 1
        marker_r[j][1] = marker_list[j][1]*scale_factor + 1
        marker_r[j][2] = marker_list[j][0]*scale_factor + 1
    savemarker(save_marker_path, img_name, marker_r)
    if need_label:
        return label

def DBScan(marker_r,eps = 256,min_samples = 50):
    X = marker_r
    #eps = 2000
    #min_samples = 290
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    clu = db.labels_
    clu_res = np.c_[X, clu]
    delete_list = []
    for clu_index in range(0, len(clu_res[:, 3])):
        label = clu_res[:, 3][clu_index]
        #if (label == 1 or label == 0):
        if (label >= 0):
            delete_list.append(clu_index)
    marker1 = np.delete(clu_res, delete_list, axis=0)  # axis=0代表按行删除
    return  marker1

def read_txt(filepath):
    txt_list = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            txt_list.append(line.strip('\n'))
    return txt_list