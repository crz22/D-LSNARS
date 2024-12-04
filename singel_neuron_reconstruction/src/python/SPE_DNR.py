import sys
sys.path.append('.')
import argparse
import torch
import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from model.DNRNET import CenterlineNet_Discrimintor_2D_Radii_32
from io1 import read_image,save_image,read_configuration,saveswc
from soma_segment import soma_segment,find_soma_with_mark
from utils import *

code_path = r"F:\neuron_reconstruction_system\D_LSNARS\singel_neuron_reconstruction\src\python"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Neuron_Reconstructor():
    def __init__(self,input_path,configuration_path,isStart = False):
        self.image_path = input_path
        self.seg_image_path = input_path+'_seg.tif'
        self.isStart = isStart
        confi = read_configuration(configuration_path)
        #self.model_name = confi['segmentMethod']
        #self.model_path = os.path.join(code_path,'weight',self.model_name)

        self.SP_N = 10                       # number of spheres
        self.SP_step = 1                     # spherical spacing
        self.psize = self.SP_N*self.SP_step  # pad of image
        self.Ma = 32
        self.Mp = 32
        self.SP_core = generate_sphere(self.Ma,self.Mp)
        self.DI_N = 1024                     # number of trace direction
        self.aph = 0.5
        Mc = int(np.sqrt(self.DI_N))
        self.SP_core_label = generate_sphere(Mc, Mc)
        self.node_step = confi['node_step']
        self.branch_MAXL =confi['branch_MAXL']
        self.Angle_T = confi['Angle_T']
        # self.Angle_T = 1.57 #1.39
        self.Lamd = confi['Lamd']
        self.seed_MAX = 30000
        #model load
        self.model = CenterlineNet_Discrimintor_2D_Radii_32(NUM_ACTIONS=self.DI_N, n=self.SP_N)
        checkpoint_path = code_path + '/weight/SPE_DNR_swc2img.pkl'
        #checkpoint_path = code_path +'/weight/CenterlineNet2D_Real_1030_90022_CorrectLoss_1024K_32Ma_10n_Preradii_model_s1.pkl'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['net_dict'])
        self.model.to(device).eval()

        self.relu = torch.nn.ReLU(inplace=True)

        #
        self.pt_index = 0
        self.nodelist = []
        self.node_to_branch = []  # Record the connection relationship between points
        self.overlapnodelist = []

    def load_sample(self):
        # load sample
        raw_image = read_image(self.image_path)  # [0,255]
        seg_image = read_image(self.seg_image_path)  # [0,255]

        # adjust raw image
        image = raw_image.astype('float')
        #######################################
        # image = image / image.max()
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.power(image, 1.0)  # 1-1.3

        seg_image1 = seg_image / seg_image.max()
        image1 = image * 0.8 + seg_image1 * 0.2
        #######################################
        #############################################
        soma_mark = soma_segment(image * 255.0, self.image_path)  # [0,1]
        if self.isStart:
            seg_image = np.where(soma_mark == False, seg_image, 255)
        seed_points = extract_seed_points(seg_image, self.image_path)  # [n,x,y,z,Im([0,1])]
        # seed_points = extract_seed_points(seg_image,self.image_path) #[n,x,y,z,Im([0,1])]
        # seed_points = extract_seed_points2(seg_image, raw_image, self.image_path)
        #############################################

        image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
        image1 = np.pad(image1, ((self.psize, self.psize), (self.psize, self.psize), (self.psize, self.psize)), 'reflect')
        Zmax, Ymax, Xmax = image1.shape
        self.IMAGE_BOUNDARY = [self.psize, Xmax-self.psize-1, self.psize, Ymax-self.psize-1, self.psize, Zmax-self.psize-1]
        #adjust soma mark
        soma_mark1 = find_soma_with_mark(soma_mark,self.image_path)
        soma_mark1 = np.pad(soma_mark1, ((self.psize, self.psize), (self.psize, self.psize), (self.psize, self.psize)), 'reflect')
        #adjust seed points
        if seed_points.shape[0] > 0:
            seed_points[:, :3] += self.psize
        return image1,soma_mark1,seed_points

    def Node_caculate(self, position):
        Spherical_patch = Spherical_Patches_Extraction(self.image, position, self.SP_N, self.SP_core, self.SP_step)
        SP = Spherical_patch.reshape([1, self.Ma, self.Mp, self.SP_N - 1]).transpose([0, 3, 1, 2])
        # print("SP: ",SP.shape,SP.max())
        # Calculate tracking direction, radius, and determine if it is background
        data = torch.from_numpy(SP)
        inputs = data.type(torch.FloatTensor).to(device)
        outputs, stop_flag = self.model(inputs)

        # radius
        if stop_flag.shape[1] > 2:
            radii = self.relu(stop_flag[:, -1, :, :]).detach().cpu().numpy().squeeze() + 1
            stop_flag = stop_flag[:, :2, :, :]
        else:
            radii = 1
        # reached the background?
        stop_flag = torch.nn.functional.softmax(stop_flag, 1)
        stop_flag = stop_flag.detach().cpu().numpy().squeeze().argmax()
        if stop_flag == 1:
            # print('terminated by discriminater', self.pt_index,position)
            return

        ### Direction
        outputs = torch.nn.functional.softmax(outputs, 1)
        direction_vector = outputs.detach().cpu().numpy().reshape([self.DI_N, 1])
        return direction_vector, radii

    def soma_area_determination(self,position,_confidence,radii):
        soma_reached = self.soma_mark[position[2], position[0], position[1]]  #
        if soma_reached:
            # print('Soma Reached', 0)
            if _confidence==0:
                nd = Node(position, _confidence, radii, soma_reached)
                self.pt_index += 1  # current value of pt_index = len(ndlist)
                self.nodelist.append(nd)
                #self.node_to_branch.append(0)
            return True
        return False

    def connect_node(self, track_neg):
        if track_neg == False:
            self.nodelist[self.pt_index - 2].nbr.append(self.pt_index - 1)
            self.nodelist[self.pt_index - 1].nbr.append(self.pt_index - 2)
        else:
            self.nodelist[self.branch_start_id - 1].nbr.append(self.pt_index - 1)
            self.nodelist[self.pt_index - 1].nbr.append(self.branch_start_id - 1)
        return

    def joint_decision(self,position,radii,max_id,_confidence,direction_last):
        correct_flag = 0
        # not joint decision
        # direction = self.sphere_core_label[max_id, :]
        # joint decision
        dist_position_seed = np.sqrt(np.sum(np.square(position - self.candidate_point[:, :3]), axis=1))
        seed_remain = np.where((dist_position_seed < (self.Lamd * radii)) & (dist_position_seed >= 2))  # find the closest seed point to current position
        remain_size = seed_remain[0].size
        # remain_size = 0
        if remain_size == 0:
            direction = self.SP_core_label[max_id, :]
        else:
            seed_remain_position = self.candidate_point[seed_remain[0], :3]
            # vector from current positon to the closest seed
            ds_vectors = (seed_remain_position - position) / np.linalg.norm(seed_remain_position - position,axis=1).reshape([remain_size, 1])
            ds_cos = np.sum(ds_vectors * direction_last, axis=1)
            ds_cos[ds_cos > 1] = 1
            ds_cos[ds_cos < -1] = -1
            ds_angle = np.arccos(ds_cos)
            ds_angle_min = ds_angle.min()
            if ds_angle_min > self.Angle_T:
                direction = self.SP_core_label[max_id, :]
            else:
                vid = np.argmin(ds_angle)
                if _confidence * 2 < self.candidate_point[seed_remain[0][vid], 3]:
                    direction = ds_vectors[vid]
                    correct_flag = 1
                    next_position = seed_remain_position[vid, :]
                    last_step_size = dist_position_seed[seed_remain[0][vid]]
                else:
                    # print(_confidence)
                    direction = self.SP_core_label[max_id, :]

        if correct_flag == 0:  # next_position is determined by seed points if correct_flag==1
            next_position = position + direction * np.max([radii, self.node_step])
            last_step_size = np.max([radii, self.node_step])
        return direction,next_position,last_step_size

    def _Track_Pos(self, position, direction1, track_neg, radii):
        cc = 0  # steps counter
        next_position = position + direction1 * np.max([radii, self.node_step])
        last_step_size = self.node_step
        while cc < self.branch_MAXL:
            cc += 1
            #correct_flag = 0
            if not within_the_boundary(next_position, self.IMAGE_BOUNDARY):
                # print('reached boundary', next_position, cc)
                break

            position = next_position.copy()
            # Prevent duplicate reconstruction on a branch
            if Overlap_point_determination(position, self.cur_branch, last_step_size):  # radii
                # print("current point overlap with current branch ",self.pt_index,position)
                break;

            position_1 = np.round(position).astype(int)
            if self.indx_map[position_1[2], position_1[0], position_1[1]] > 0:
                # print('Meet Traced Region',self.pt_index,position_1)
                # biuld connection between the met points
                nd = Node(position, 0, 1)
                self.overlapnodelist.append(self.pt_index)
                self.pt_index += 1
                self.nodelist.append(nd)
                self.connect_node(track_neg)  # biuld connection between the met points
                break

            if self.soma_area_determination(position_1,_confidence=0,radii=1):
                self.connect_node(track_neg)
                # print('Soma Reached1', self.pt_index,position_1)
                break

            node1 = self.Node_caculate(position)
            if node1 == None:
                break

            direction_vector, radii = node1
            cos_angle = np.sum(direction1 * self.SP_core_label, axis=1)
            cos_angle[cos_angle > 1] = 1
            cos_angle[cos_angle < -1] = -1
            angle = np.arccos(cos_angle).reshape([self.DI_N, 1])
            direction_vector[angle > self.Angle_T] = 0
            max_id = np.argmax(direction_vector)
            # direction_vector = direction_vector/direction_vector.sum()  ##################1
            _confidence = direction_vector[max_id]

            nd = Node(position, _confidence, radii)
            self.nodelist.append(nd)
            self.pt_index += 1
            self.connect_node(track_neg)  # biuld connection between the met points
            track_neg = False
            self.cur_branch.append(np.append(position, radii))

            direction1,next_position,last_step_size = self.joint_decision(position,radii,max_id,_confidence,direction1)

    def _mask_point(self, node_position, radii, index=0):
        n = np.rint(node_position).astype(int)
        radii = np.rint(radii).astype(int)
        X, Y, Z = np.meshgrid(
            constrain_range(n[0]-radii, n[0]+radii+1, self.IMAGE_BOUNDARY[0], self.IMAGE_BOUNDARY[1]),
            constrain_range(n[1]-radii, n[1]+radii+1, self.IMAGE_BOUNDARY[2], self.IMAGE_BOUNDARY[3]),
            constrain_range(n[2]-radii, n[2]+radii+1, self.IMAGE_BOUNDARY[4], self.IMAGE_BOUNDARY[5]))
        if index == 0:
            self.indx_map[Z, X, Y] = self.pt_index
        else:
            self.indx_map[Z, X, Y] = index
    '''
    def MergeAdjacentPoints(self):
        node_position = []
        for i in range(len(self.nodelist)):
            node_position.append(self.nodelist[i].position)
        node_position = np.array(node_position)
        remain_overlapnodelist = []
        for i in self.overlapnodelist:
            # print(i,self.nodelist[i].position)
            dist = np.sqrt(np.sum(np.square(self.nodelist[i].position - node_position), axis=1))
            dist[i] = 10000

            min_dist = 0
            while (min_dist <= 4):
                min_dist = np.min(dist)
                min_index = np.argmin(dist)

                if self.node_to_branch[i] != self.node_to_branch[min_index] and len(self.nodelist[min_index].nbr) < 3:
                    # max_connect point num two child+one parents
                    self.nodelist[min_index].nbr.append(i)
                    self.nodelist[i].nbr.append(min_index)
                    if self.node_to_branch[i] < self.node_to_branch[min_index]:
                        self.node_to_branch = np.where(self.node_to_branch == self.node_to_branch[min_index],
                                                       self.node_to_branch[i], self.node_to_branch)
                    else:
                        self.node_to_branch = np.where(self.node_to_branch == self.node_to_branch[i],
                                                       self.node_to_branch[min_index], self.node_to_branch)
                    remain_overlapnodelist.append(i)
                    if min_index in self.overlapnodelist:
                        remain_overlapnodelist.append(min_index)
                    break
                # print(min_index)
                np.put(dist, min_index, 10000)

        self.overlapnodelist = [i for i in self.overlapnodelist if i not in remain_overlapnodelist]

    '''
    def MergeAdjacentPoints(self):
        node_position = []
        for i in range(len(self.nodelist)):
            node_position.append(self.nodelist[i].position)
        node_position = np.array(node_position)
        for i in self.overlapnodelist:
            # print(i,self.nodelist[i].position)
            dist = np.sqrt(np.sum(np.square(self.nodelist[i].position - node_position), axis=1))
            dist[i] = 10000

            min_dist = 0
            while (min_dist <= 4):
                min_dist = np.min(dist)
                min_index = np.argmin(dist)
                if min_index in self.overlapnodelist:
                    np.put(dist, min_index, 10000)
                    continue
                if self.node_to_branch[i] != self.node_to_branch[min_index] and len(self.nodelist[min_index].nbr) < 3:
                    # max_connect point num two child+one parents
                    for onbr_index in self.nodelist[i].nbr:
                        self.nodelist[min_index].nbr.append(onbr_index)
                        self.nodelist[onbr_index].nbr.append(min_index)
                        self.nodelist[onbr_index].nbr.remove(i)
                    self.nodelist[i].nbr.clear()

                    if self.node_to_branch[i] < self.node_to_branch[min_index]:
                        self.node_to_branch = np.where(self.node_to_branch == self.node_to_branch[min_index],
                                                       self.node_to_branch[i], self.node_to_branch)
                    else:
                        self.node_to_branch = np.where(self.node_to_branch == self.node_to_branch[i],
                                                       self.node_to_branch[min_index], self.node_to_branch)
                    break
                np.put(dist, min_index, 10000)

    def start(self):
        self.image, self.soma_mark, self.candidate_point = self.load_sample()
        self.indx_map = np.zeros_like(self.image, dtype=np.int64)  # index map to label the index of traced point
        ################################# traced neuron points #############################################
        lent = self.candidate_point.shape[0]
        branch_num = 0
        if lent <= self.seed_MAX and lent>0:
            for i in tqdm(range(lent)):
                self.cur_branch = []
                position = self.candidate_point[i, 0:3]  # [x,y,z]
                # check if the current point has been tracked
                position = np.round(position).astype(int)
                #print(position)
                if self.indx_map[position[2], position[0], position[1]] > 0:
                    # print('Traced Seed', self.pt_index ,position)
                    continue
                # Check if the point is within the boundary
                if not within_the_boundary(position, self.IMAGE_BOUNDARY):
                    # print('out of boundary', position,i)
                    continue
                # Extract spherical feature maps near seed points
                node0 = self.Node_caculate(position)
                if node0 == None:
                    continue
                direction_vector, radii = node0

                # determine two initial direction
                # first direction
                max_id = np.argmax(direction_vector)
                direction1 = self.SP_core_label[max_id, :]
                cos_angle = np.sum(direction1 * self.SP_core_label, axis=1)
                cos_angle[cos_angle > 1] = 1
                cos_angle[cos_angle < -1] = -1
                # second direction
                angle = np.arccos(cos_angle).reshape([self.DI_N, 1])
                direction_vector[angle <= np.pi / 2] = 0
                max_id = np.argmax(direction_vector)
                direction2 = self.SP_core_label[max_id, :]
                _confidence = direction_vector[max_id]

                # Determine if it has reached the Soma area
                if self.soma_area_determination(position,_confidence,radii):
                    # print('Soma Reached', self.pt_index,position)
                    continue

                self.cur_branch.append(np.append(position, radii))
                nd = Node(position, _confidence, radii)
                self.pt_index += 1  # current value of pt_index = len(ndlist)
                self.nodelist.append(nd)
                self.branch_start_id = self.pt_index  # used for masking position location after bidirectional tracking
                previous_nd_len = len(self.nodelist)  # length of ndlist in previous iteration

                # trace towards direction1
                track_neg = False
                self._Track_Pos(position, direction1, track_neg, radii)

                # trace towards direction2
                #position = self.candidate_point[i, 0:3]
                track_neg = True
                self._Track_Pos(position, direction2, track_neg, radii)

                # label the traced branches
                branch_num += 1
                self._mask_point(position, radii, self.branch_start_id)
                self.node_to_branch.append(branch_num)
                len_branch = len(self.nodelist) - previous_nd_len  # length of new added branches
                for j in range(len_branch):
                    self.node_to_branch.append(branch_num)
                    position_m = self.nodelist[previous_nd_len + j].position
                    self._mask_point(position_m, self.nodelist[previous_nd_len + j].radius, previous_nd_len + j + 1)
                #print(branch_num, len(self.nodelist), len(self.node_to_branch))
            self.MergeAdjacentPoints()

        ############################# build soma shape and neuron tree ########################################
        print("node_num1: ",len(self.nodelist))
        if self.isStart:
            soma_mark1 = self.soma_mark.copy()
            soma_mark1[soma_mark1<0] = 0
            distance_transform = distance_transform_edt(soma_mark1)
            neuron_tree = compute_trees(self.nodelist,self.overlapnodelist,soma_mark1, distance_transform)
            swc = build_nodelist(neuron_tree)
            #swc = self.connected_soma(swc, distance_transform)

        else:
            neuron_tree = compute_trees(self.nodelist,self.overlapnodelist)
            swc = build_nodelist(neuron_tree)
            #swc = self.delete_nosy_point(swc)

        # use this result for multi-neuron reconstruction
        swc[:, 2:5] = swc[:, 2:5] - self.psize
        swc[:, 2:5] = swc[:, 2:5] + 1  # Vaa3d starts from 1 but python from 0
        save_swc_path = self.image_path + '.swc'

        #print(swc)
        saveswc(save_swc_path, swc)
        #saveswc(self.image_path+'_not_connect.swc',swc)


if __name__ == '__main__':
    print('start reconstruct')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default='', help='image path')
    parser.add_argument('--configuration_path', '-c', type=str, default='', help='configuration path')
    parser.add_argument('--isstart', '-s', type=str, default=' ', help='is start')
    args = parser.parse_args()
    if args.isstart == '1':
        isstart = True
    else:
        isstart = False
    print("isstart: ",isstart)
    reconstructor = Neuron_Reconstructor(args.input_path, args.configuration_path, isstart)
    #input_path = r"F:\neuron_reconstruction_system\test\reconstruct_result\2024-07-22\21_40_26\mouse17302_teraconvert_tmp\x16106_y15377_z2935.tif"
    #configuration_path = r"F:\neuron_reconstruction_system\test\reconstruct_result\2024-07-22\21_40_26\configuration.yaml"
    #output_path =
    #reconstructor = Neuron_Reconstructor(input_path, configuration_path,isStart=True)
    reconstructor.start()