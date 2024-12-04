import math
import torch
import numpy as np
from skimage.morphology import skeletonize_3d
import torch.nn.functional as F
from io1 import save_image

class Node(object):
   def __init__(self, position, conf, radius, node_type=3):
       self.position = position
       self.conf = conf
       self.radius = radius
       self.node_type = node_type
       self.nbr = []

def cut_image(img, block_size=(32, 64, 64), step=(32, 64, 64), pad_model='reflect'):
    z_size, x_size, y_size = block_size
    z_step, x_step, y_step = step
    z_img, x_img, y_img = img.shape[2:5]

    z_max = math.ceil((z_img - z_size) / z_step) + 1
    x_max = math.ceil((x_img - x_size) / x_step) + 1
    y_max = math.ceil((y_img - y_size) / y_step) + 1

    max_num = [z_max, x_max, y_max]

    z_pad = (z_max - 1) * z_step + z_size - z_img
    x_pad = (x_max - 1) * x_step + x_size - x_img
    y_pad = (y_max - 1) * y_step + y_size - y_img

    if pad_model == 'constant':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'constant', value=-1)
    elif pad_model == 'reflect':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'reflect')

    image_blockes = []
    block_num = 0
    for xx in range(x_max):
        for yy in range(y_max):
            for zz in range(z_max):
                img_block = img[:, :, zz * z_step:zz * z_step + z_size, xx * x_step:xx * x_step + x_size,
                            yy * y_step:yy * y_step + y_size]
                image_blockes.append(img_block)
                block_num = block_num + 1

    return image_blockes, block_num, max_num

def splice_image(img_block, block_num, max_num, image_size=(100, 1024, 1024), step=(32, 64, 64)):
    z_img, x_img, y_img = image_size
    z_num, x_num, y_num = max_num
    z_step, x_step, y_step = step

    zz = 0
    yy = 0
    xx = 0

    for i in range(block_num):
        img_block[i] = img_block[i][:, :, 0:z_step, 0:x_step, 0:y_step]
        if zz == 0:
            image_z = img_block[i]
        else:
            image_z = torch.cat([image_z, img_block[i]], dim=2)
        zz = zz + 1
        if zz == z_num:
            zz = 0
            if yy == 0:
                image_y = image_z
            else:
                image_y = torch.cat([image_y, image_z], dim=4)
            yy = yy + 1

        if yy == y_num:
            # print(image_x)
            yy = 0
            if xx == 0:
                image_x = image_y
            else:
                image_x = torch.cat([image_x, image_y], dim=3)
            xx = xx + 1

    image = image_x[:, :, 0:z_img, 0:y_img, 0:x_img]
    # print(image.shape)
    return image
'''
def cut_image_overlap(img, block_size, step, pad_model='reflect'):
    z_size, x_size, y_size = block_size
    z_step, x_step, y_step = step
    z_img, x_img, y_img = img.shape[2:5]

    z_max = math.ceil((z_img - z_size) / z_step) + 1
    x_max = math.ceil((x_img - x_size) / x_step) + 1
    y_max = math.ceil((y_img - y_size) / y_step) + 1

    max_num = [z_max, x_max, y_max]

    z_pad = (z_max - 1) * z_step + z_size - z_img
    x_pad = (x_max - 1) * x_step + x_size - x_img
    y_pad = (y_max - 1) * y_step + y_size - y_img

    if pad_model == 'constant':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'constant', value=-1)
    elif pad_model == 'reflect':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'reflect')
    #print(img.shape)
    image_blockes = []
    block_num = 0
    for xx in range(x_max):
        for yy in range(y_max):
            for zz in range(z_max):
                img_block = img[:, :, zz * z_step:zz * z_step + z_size, xx * x_step:xx * x_step + x_size,
                            yy * y_step:yy * y_step + y_size]
                image_blockes.append(img_block)
                block_num = block_num + 1

    return image_blockes, block_num, max_num

def splice_image_overlap(img_block, block_num, max_num, image_size, step):
    z_img, x_img, y_img = image_size
    z_num, x_num, y_num = max_num
    z_step, x_step, y_step = step
    z_block_size, x_block_size, y_block_size = img_block[0].shape[2:5]
    thz = z_block_size - z_step
    thx = x_block_size - x_step
    thy = y_block_size - y_step

    zz = 0
    yy = 0
    xx = 0

    for i in range(block_num):
        if zz == 0:
            image_z = img_block[i]
        else:
            cur_z = image_z.shape[2]
            image_z[:, :, -thz:cur_z, :, :] = (image_z[:, :, -thz:cur_z, :, :] + img_block[i][:, :, 0:thz, :, :]) / 2
            image_z = torch.cat([image_z, img_block[i][:, :, thz:z_block_size, :, :]], dim=2)
        zz = zz + 1
        if zz == z_num:
            zz = 0
            if yy == 0:
                image_y = image_z
            else:
                cur_y = image_y.shape[4]
                image_y[:, :, :, :, -thy:cur_y] = (image_y[:, :, :, :, -thy:cur_y] + image_z[:, :, :, :, 0:thy]) / 2
                image_y = torch.cat([image_y, image_z[:, :, :, :, thy:y_block_size]], dim=4)
            yy = yy + 1

        if yy == y_num:
            # print(image_x)
            yy = 0
            if xx == 0:
                image_x = image_y
            else:
                cur_x = image_x.shape[3]
                image_x[:, :, :, -thx:cur_x, :] = (image_x[:, :, :, -thx:cur_x, :] + image_y[:, :, :, 0:thx, :]) / 2
                image_x = torch.cat([image_x, image_y[:, :, :, thx:x_block_size, :]], dim=3)
            xx = xx + 1

    image = image_x[:, :, 0:z_img, 0:y_img, 0:x_img]
    return image

def splice_image_overlap(img_block, block_num, max_num, image_size, step):
    z_img, x_img, y_img = image_size
    z_num, x_num, y_num = max_num
    z_step, x_step, y_step = step
    z_block_size, x_block_size, y_block_size = img_block[0].shape[2:5]
    thz = z_block_size - z_step
    thx = x_block_size - x_step
    thy = y_block_size - y_step
    zz = 0
    yy = 0
    xx = 0
    for i in range(block_num):
        if zz == 0:
            image_z = img_block[i][:, :, 0:-thz//2, :, :]
        else:
            image_z = torch.cat([image_z, img_block[i][:, :, thz//2:z_block_size-thz//2, :, :]], dim=2)
        zz = zz + 1
        if zz == z_num:
            zz = 0
            image_z = torch.cat([image_z, img_block[i][:, :, z_block_size-thz//2:, :, :]], dim=2)
            if yy == 0:
                image_y = image_z[:, :, :, :, 0:-thy//2]
            else:
                image_y = torch.cat([image_y, image_z[:, :, :, :, thy//2:y_block_size-thy//2]], dim=4)
            yy = yy + 1

        if yy == y_num:
            yy = 0
            image_y = torch.cat([image_y, image_z[:, :, :, :, y_block_size - thy // 2: ]], dim=4)
            if xx == 0:
                image_x = image_y[:, :, :, 0:-thx//2, :]
            else:
                image_x = torch.cat([image_x, image_y[:, :, :, thx//2:x_block_size-thy//2, :]], dim=3)
            xx = xx + 1
        if xx == x_num:
            image_x = torch.cat([image_x, image_y[:, :, :, x_block_size - thy // 2 : , :]], dim=3)
    image = image_x[:, :, 0:z_img, 0:y_img, 0:x_img]
    return image
'''

def cut_image_overlap(img, block_size, step, pad_model='reflect'):
    z_size, x_size, y_size = block_size
    z_step, x_step, y_step = step
    thz,thx,thy = z_size-z_step,x_size-x_step,y_size-y_step
    img1 = F.pad(img, (thx//2, thx//2, thy//2, thy//2, thz//2, thz//2), 'reflect')
    #img1 = F.pad(img, (thx // 2, 0, thy // 2,0, thz // 2, 0), 'reflect')
    #print(img1.shape)
    z_img, x_img, y_img = img1.shape[2:5]
    z_max = math.ceil((z_img - z_size) / z_step) + 1
    x_max = math.ceil((x_img - x_size) / x_step) + 1
    y_max = math.ceil((y_img - y_size) / y_step) + 1

    max_num = [z_max, x_max, y_max]

    z_pad = (z_max - 1) * z_step + z_size - z_img
    x_pad = (x_max - 1) * x_step + x_size - x_img
    y_pad = (y_max - 1) * y_step + y_size - y_img
    #print(max_num, z_pad)
    if pad_model == 'constant':
        img1 = F.pad(img1, (0, x_pad, 0, y_pad, 0, z_pad), 'constant', value=-1)
    elif pad_model == 'reflect':
        img1 = F.pad(img1, (0, x_pad, 0, y_pad, 0, z_pad), 'reflect')
    # print(img.shape)
    image_blockes = []
    block_num = 0
    for xx in range(x_max):
        for yy in range(y_max):
            for zz in range(z_max):
                img_block = img1[:, :, zz * z_step:zz * z_step + z_size, xx * x_step:xx * x_step + x_size,
                            yy * y_step:yy * y_step + y_size]
                image_blockes.append(img_block)
                block_num = block_num + 1

    return image_blockes, block_num, max_num

def splice_image_overlap(img_block, block_num, max_num, image_size, step):
    z_img, x_img, y_img = image_size
    z_num, x_num, y_num = max_num
    z_step, x_step, y_step = step
    z_block_size, x_block_size, y_block_size = img_block[0].shape[2:5]
    thz = z_block_size - z_step
    thx = x_block_size - x_step
    thy = y_block_size - y_step
    zz = 0
    yy = 0
    xx = 0
    for i in range(block_num):
        if zz == 0:
            image_z = img_block[i][:, :, thz//2:-thz//2, :, :]
        else:
            image_z = torch.cat([image_z, img_block[i][:, :, thz//2:z_block_size-thz//2, :, :]], dim=2)
        zz = zz + 1
        if zz == z_num:
            zz = 0
            image_z = torch.cat([image_z, img_block[i][:, :, z_block_size-thz//2:, :, :]], dim=2)
            if yy == 0:
                image_y = image_z[:, :, :, :, thy//2:-thy//2]
            else:
                image_y = torch.cat([image_y, image_z[:, :, :, :, thy//2:y_block_size-thy//2]], dim=4)
            yy = yy + 1

        if yy == y_num:
            yy = 0
            image_y = torch.cat([image_y, image_z[:, :, :, :, y_block_size - thy // 2: ]], dim=4)
            if xx == 0:
                image_x = image_y[:, :, :, thx//2:-thx//2, :]
            else:
                image_x = torch.cat([image_x, image_y[:, :, :, thx//2:x_block_size-thy//2, :]], dim=3)
            xx = xx + 1
        if xx == x_num:
            image_x = torch.cat([image_x, image_y[:, :, :, x_block_size - thy // 2 : , :]], dim=3)
    image = image_x[:, :, 0:z_img, 0:y_img, 0:x_img]
    return image

def extract_seed_points(seg_image,image_path):
    save_path = image_path+'_seedmap.tif'
    th = 0.5
    seg_image_th = np.where(seg_image<255*th,0,seg_image)
    seed_image = skeletonize_3d(seg_image_th) #[0,255]
    #print("seed: ",seed_image.max(),seed_image.min())
    save_image(seed_image,save_path)
    candidate_sup = local_max(seed_image, wsize=3)
    candidate_points = np.array(candidate_sup)
    return candidate_points

def local_max(Im, wsize=3, thre=128):
    nZ, nY, nX = Im.shape
    Im_copy = Im.copy()
    pad = wsize//2
    potential_points = np.where(Im > thre)
    num_points = len(potential_points[0])
    coordinates = []
    for i in range(num_points):
        z = potential_points[0][i]
        y = potential_points[1][i]
        x = potential_points[2][i]

        if x < 1 or y < 1 or z < 1 or x > nX - 2 or y > nY - 2 or z > nZ - 2:
            continue
        img_patch = Im_copy[z-pad:z+pad+1, y-pad:y+pad+1, x-pad:x+pad+1]
        if img_patch.max() == img_patch[1, 1, 1]:
            # suppress[z, y, x] = 255
            coordinates.append([y,x,z, Im[z, y, x]/255.0])
            Im_copy[z-pad:z+pad+1, y-pad:y+pad+1, x-pad:x+pad+1] = np.zeros_like(img_patch)
    return coordinates

def extract_seed_points2(seg_image,raw_image,image_path):
    save_path = image_path+'_seedmap.tif'
    th = 0.5
    seg_image_th = np.where(seg_image<255*th,0,seg_image)
    seed_image = skeletonize_3d(seg_image_th) #[0,255]
    save_image(seed_image,save_path)
    candidate_sup = local_max2(seed_image,raw_image,wsize=3)
    candidate_points = np.array(candidate_sup)
    return candidate_points

def local_max2(Seed, Im, wsize=3, thre=200):
    nZ, nY, nX = Im.shape
    Im_copy = Im.copy()
    Seed_copy = Seed.copy()
    pad = wsize//2
    potential_points = np.where(Seed > thre)
    num_points = len(potential_points[0])
    coordinates = []
    for i in range(num_points):
        z = potential_points[0][i]
        y = potential_points[1][i]
        x = potential_points[2][i]

        if x < pad or y < pad or z < pad or x > nX - pad-1 or y > nY - pad-1 or z > nZ - pad-1:
            continue
        img_patch = Im_copy[z-pad:z+pad+1, y-pad:y+pad+1, x-pad:x+pad+1]
        seed_patch = Seed_copy[z-pad:z+pad+1, y-pad:y+pad+1, x-pad:x+pad+1]
        if seed_patch.max() == seed_patch[pad, pad, pad] :#:and seed_patch.max()>0
            seed_adjust = np.where(img_patch == img_patch.max())
            #print(seed_adjust)
            z1 = int(z-pad + seed_adjust[0][0])
            y1 = int(y-pad + seed_adjust[1][0])
            x1 = int(x-pad + seed_adjust[2][0])
            #print(y1,x1,z1)
            # suppress[z, y, x] = 255
            coordinates.append([y1,x1,z1, Seed[z, y, x]/255.0])
            Seed_copy[z-pad:z+pad+1, y-pad:y+pad+1, x-pad:x+pad+1] = np.zeros_like(seed_patch)

    return coordinates

def within_the_boundary(point,boundary):
    # point [z,y,x]
    # boundary [lower_z,uper_z,lower_y,uper_y,lower_x,uper_x]
    if point[0] <= boundary[0] or point[0] >= boundary[1]:
        return False
    if point[1] <= boundary[2] or point[1] >= boundary[3]:
        return False
    if point[2] <= boundary[4] or point[2] >= boundary[5]:
        return False
    return True

def generate_sphere(Ma,Mp):
    #generate 3d sphere
    m1=np.arange(1,Ma+1,1).reshape(-1,Ma)
    m2=np.arange(1,Mp+1,1).reshape(-1,Mp)
    alpha=2*(np.pi)*m1/Ma
    phi=-(np.arccos(2*m2/(Mp+1)-1)-(np.pi))
    xm=(np.cos(alpha).reshape(Ma,1))*np.sin(phi)
    ym=(np.sin(alpha).reshape(Ma,1))*np.sin(phi)
    zm=np.cos(phi)
    zm=np.tile(zm,(Mp,1))
    sphere_core=np.concatenate([xm.reshape(-1,1), ym.reshape(-1,1), zm.reshape(-1,1)],axis=1) #y_axis=alpha[0:Ma],x_axis=phi[0:Mp]
    return sphere_core #, alpha, phi

def Spherical_Patches_Extraction(img2, position, SP_N, SP_core, SP_step=1):
    x = position[0]
    y = position[1]
    z = position[2]
    radius = 1
    j = np.arange(radius, SP_N*SP_step + radius, SP_step).reshape(-1, SP_N)
    ray_x = x + (SP_core[:, 0].reshape(-1, 1)) * j
    ray_y = y + (SP_core[:, 1].reshape(-1, 1)) * j
    ray_z = z + (SP_core[:, 2].reshape(-1, 1)) * j

    Rray_x = np.rint(ray_x).astype(int)
    Rray_y = np.rint(ray_y).astype(int)
    Rray_z = np.rint(ray_z).astype(int)

    Spherical_patch_temp = img2[Rray_z, Rray_x, Rray_y]
    Spherical_patch = Spherical_patch_temp[:, 1:SP_N]

    SP = np.asarray(Spherical_patch)
    pmax = SP.max()

    if pmax > 0:
        SP = SP / pmax

    if SP.max()-SP.min()>0:
        SP = (SP-SP.min()) / (SP.max()-SP.min())
    '''  '''
    return SP

def Overlap_point_determination(node,branch,r):
    branch = np.array(branch)
    dist = np.sqrt(np.sum(np.square(node-branch[:,0:3]),axis=1))
    min_dist = np.min(dist)

    dist = dist-r-branch[:,3]
    if np.sum(dist<=0)>3 or min_dist==0:
        #print(np.sum(min_dist))
        return True
    return False

def constrain_range(min, max, minlimit, maxlimit):
    return list(
        range(min if min > minlimit else minlimit, max
              if max < maxlimit else maxlimit))

def compute_trees(n0, del_node=[], soma_mark=None, distance_transform=None):
    n0_size = len(n0)
    treecnt = 0
    q = []  # bfs queue
    PointsInSoma = []
    n1 = []
    dist = np.ones([n0_size, 1], dtype=np.int32) * 100000
    nmap = np.ones([n0_size, 1], dtype=np.int32) * -1  # index in output tree n1
    parent = np.ones([n0_size, 1], dtype=np.int32) * -1  # parent index in current tree n0

    print('Search for Soma')
    if soma_mark is not None:
        soma_candidate, radius = soma_point(distance_transform,soma_mark.shape[0:3])
        n = Node(soma_candidate,conf=0,radius=radius,node_type=treecnt+2)
        n1.append(n)
        for i in range(n0_size):
            if n0[i].node_type == 1:
                PointsInSoma.append(i)
                q.append(i)
                dist[i] = 0
            elif n0[i].node_type == -1:
                del_node.append(i)
        # BFS
        while len(q) > 0:
            curr = q.pop()
            if curr in del_node:
                continue
            n = Node(n0[curr].position, n0[curr].conf, n0[curr].radius, treecnt + 2)
            if parent[curr] >= 0:
                n.nbr.append(nmap[parent[curr]])
            elif curr in PointsInSoma:
                n.nbr.append(np.array([1]))
            n1.append(n)
            nmap[curr] = len(n1)
            for j in range(len(n0[curr].nbr)):
                adj = n0[curr].nbr[j]
                if dist[adj] == 100000:
                    dist[adj] = dist[curr] + 1
                    parent[adj] = curr
                    q.append(adj)
            #print(curr, n.position - 9, n.nbr)


    while (get_undiscover(dist)>= 0):
        treecnt += 1
        seed = get_undiscover(dist)
        dist[seed] = 0
        q.append(seed)
        while len(q) > 0:
            curr = q.pop()
            if curr in del_node:
                continue
            n = Node(n0[curr].position, n0[curr].conf, n0[curr].radius, treecnt + 2)
            if parent[curr] >= 0:
                n.nbr.append(nmap[parent[curr]])
            n1.append(n)
            nmap[curr] = len(n1)
            for j in range(len(n0[curr].nbr)):
                adj = n0[curr].nbr[j]
                if dist[adj] == 100000:
                    dist[adj] = dist[curr] + 1
                    parent[adj] = curr
                    q.append(adj)
            #if treecnt+2 == 145:
            #    print(curr,n.position-9,n.nbr)
    return n1

def soma_point(distance_transform,image_size):
    x,y,z = image_size
    Image_center_point = [x//2,y//2,z//2]
    maxr = np.max(distance_transform)
    # print(Image_center_point)
    position = np.argwhere(distance_transform == maxr)
    #print("maxr",maxr)
    #print(position)
    dist = np.sum(np.square(position-Image_center_point),axis=1)
    point_id = np.argmin(dist)
    #print(point_id)
    #print(position)
    soma_position = position[point_id]
    print("soma_position: ",soma_position,maxr)
    #temp = position[0] - soma_point
    #tmp1 = np.linalg.norm(temp, axis=1)
    #radius = np.max(tmp1)
    #radius = 1
    #print(radius)
    return soma_position, maxr

def get_undiscover(dist):
    for i in range(dist.shape[0]):
        if dist[i] == 100000:
            return i
    return -1

def build_nodelist(tree):
    _data = np.zeros((1, 7))
    cnt_recnodes = 0
    for i in range(len(tree)):
        #print(i)
        if len(tree[i].nbr) == 0:
            cnt_recnodes += 1
            pid = -1
            new_node = np.asarray([cnt_recnodes,
                                   tree[i].node_type,
                                   tree[i].position[1],
                                   tree[i].position[0],
                                   tree[i].position[2],
                                   tree[i].radius,
                                   pid])
            _data = np.vstack((_data, new_node))

        else:
            # if len(tree[i].nbr) > 2:
            #    print(len(tree[i].nbr))
            for j in range(len(tree[i].nbr)):
                cnt_recnodes += 1
                pid = tree[i].nbr[j].squeeze()
                new_node = np.asarray([cnt_recnodes,
                                       tree[i].node_type,
                                       tree[i].position[1],
                                       tree[i].position[0],
                                       tree[i].position[2],
                                       tree[i].radius,
                                       pid])
                _data = np.vstack((_data, new_node))
    _data = _data[1:, :]
    return _data