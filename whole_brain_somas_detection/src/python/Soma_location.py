import sys
sys.path.append('.')
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from model.SFSNet3 import S_UNet
from model.Unet_3D import UNet_3D
from model.ATT_UNet import Att_UNet
from base_function import *
import torchvision.transforms as transforms

code_path = r'F:\neuron_reconstruction_system\D_LSNARS\whole_brain_somas_detection\src\python'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_images = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

def nametopath(name):
    # candidate_name： 500100_273280_029580
    # image path: \RES(27300x17206x4923)\500100\500100_273280\500100_273280_029580.tif
    file_list = name.split('_')
    file1 = file_list[0]
    file2 = file_list[0]+'_'+file_list[1]
    file3 = name+'.tif'
    return os.path.join(file1,file2,file3)

class test_dataset(Dataset):
    def __init__(self,dataset_path,candidate_list,transform=None):
        super(test_dataset, self).__init__()
        self.dataset_path = dataset_path
        self.candidate_list = candidate_list
        self.transform = transform

    def __len__(self):
        return len(self.candidate_list)

    def load_img(self,path):
        image = readtiff3d(path)
        image = image / 4096 * 255.0
        image = np.uint8(image)
        image = ((image - image.min()) / (image.max() - image.min())).astype(np.float32)
        # image = (image/ image.max()).astype(np.float32)

        #resize image to [256,256,256]
        original_img_size = np.array(image.shape)
        # Adjust the size of the block to ensure that it can be partitioned

        z_pad = 256-image.shape[0]
        y_pad = 256-image.shape[1]
        x_pad = 256-image.shape[2]

        image1 = np.pad(image, ((0, z_pad), (0, y_pad), (0, x_pad)), 'reflect')
        image1 = transform_images(image1)
        image1 = image1.permute(1, 2, 0)
        return image1,original_img_size

    def __getitem__(self, item):
        # candidate_name： 500100_273280_029580
        # image path: \RES(27300x17206x4923)\500100\500100_273280\500100_273280_029580.tif
        candidate_name = self.candidate_list[item]
        filepath = nametopath(candidate_name)
        image,image_size = self.load_img(os.path.join(self.dataset_path,filepath))
        image = image.unsqueeze(dim=0)
        return image,image_size,candidate_name

class whole_brain_soma_segment():
    def __init__(self,input_file_path,configuration_path,output_path,batch_size=1,transform=transform_images):
        self.input_file_path = input_file_path
        self.result_out_path = output_path
        self.candidate_path = output_path + '/candidate_block.txt'
        candidate_list = read_txt(self.candidate_path)

        confi = read_configuration(configuration_path)
        # print(confi)
        self.SMA = confi['SMA']
        self.DB_eps = confi['DB_eps'];
        self.DB_MP = confi['DB_MP']
        self.model_name = 'SFSUNet12'

        TEST_DATASET = test_dataset(input_file_path,candidate_list,transform)
        self.dataset = DataLoader(dataset=TEST_DATASET, batch_size=batch_size)

        if 'SFS' in self.model_name:
            self.Segment_model = S_UNet(in_dim=1, out_dim=2, num_filters=16)
        elif 'UNet_3D' in self.model_name:
            self.Segment_model = UNet_3D(in_dim=1, out_dim=2, num_filters=16)
        elif 'ATT_UNet' in self.model_name:
            self.Segment_model = Att_UNet(in_dim=1, out_dim=2, num_filters=16)


        Segment_MODEL_PATH = code_path+'/weight/'+ self.model_name+'/'+self.model_name+'_200.pt'
        self.Segment_model.load_state_dict(torch.load(Segment_MODEL_PATH,map_location='cpu')['model'])
        self.Segment_model.eval().to(device)

    def segment(self,image,raw_image_size):
        seg_block_size = [64, 128, 128]
        image = image.to(device)
        batch, c, w, h, l = image.shape
        num = np.array([w, h, l]) // seg_block_size
        buf_0 = np.zeros([batch, w, h, l], dtype=np.uint8)
        for i in range(num[2]):
            for j in range(num[1]):
                for k in range(num[0]):
                    image_block = image[:, :, seg_block_size[0] * k:seg_block_size[0] * (k + 1),
                                  seg_block_size[1] * j:seg_block_size[1] * (j + 1),
                                  seg_block_size[2] * i:seg_block_size[2] * (i + 1)]
                    #print(image_block.shape)
                    if image_block.max() == 0 or image_block.max() == image_block.min():
                        continue
                    output = self.Segment_model(image_block)  # (batch*2*64*128*128)
                    output_argmax = torch.argmax(output, dim=1)  # (batch*64*128*128)
                    #print(output.shape, output_argmax.shape)
                    ppl = output_argmax.float()
                    ppl = ppl.detach().cpu().numpy()

                    buf_0[:, seg_block_size[0] * k: seg_block_size[0] * (k + 1),
                    seg_block_size[1] * j: seg_block_size[1] * (j + 1),
                    seg_block_size[2] * i: seg_block_size[2] * (i + 1)] = ppl
                    del output
                    del output_argmax

        buf_1 = []
        for b in range(batch):
            buf_1.append(buf_0[b,0:raw_image_size[b,0],0:raw_image_size[b,1],0:raw_image_size[b,2]])
        return buf_1

    def connected_component(self,image, img_name, scale_factor=1):
        soma_list_path = self.result_out_path+'/soma_blocks'
        if not os.path.exists(soma_list_path):
            os.mkdir(soma_list_path)
        # Mark the input 3D image
        label, num = measure.label(image, connectivity=2, return_num=True)  # 1代表４连通，２代表８连通
        if num < 1:
            print("segment fail: ", img_name)
            #marker_r = np.ones((0, 3))
            #block_marker_path = os.path.join(soma_list_path, img_name + '.marker')
            #save_block_marker(block_marker_path, marker_r)
            return

        # obtain soma location
        region = measure.regionprops(label)
        num_list = [i for i in range(1, num + 1)]
        candidate_marker_list = [region[i - 1].centroid for i in num_list]
        area_list = [region[i - 1].area for i in num_list]
        marker_list = []
        for i in range(0, num):
            # if region[i - 1].area > 1000:
            if area_list[i] >= self.SMA:
                marker_list.append(candidate_marker_list[i])

        if len(marker_list) == 0:
            print("segmented soma not satisfiable: ", img_name)
        # print('marker_list', marker_list) #[z,y,x]
        marker_r = np.ones((len(marker_list), 3))  # [x,y,z]
        for j in range(0, len(marker_list)):
            marker_r[j][0] = marker_list[j][2] * scale_factor + 1
            marker_r[j][1] = marker_list[j][1] * scale_factor + 1
            marker_r[j][2] = marker_list[j][0] * scale_factor + 1

        block_marker_path = os.path.join(soma_list_path,img_name+'.marker')
        save_block_marker(block_marker_path,marker_r)

    def Convert_coordinates(self):
        block_marker_list = os.listdir(self.result_out_path+'/soma_blocks')
        save_path = os.path.join(self.result_out_path,'predict_somalist.marker')
        print('save_path: ',save_path)
        for file in block_marker_list:
            marker = read_marker(os.path.join(self.result_out_path+'/soma_blocks',file))
            if marker.shape[0]<1:
                continue
            #print(file,marker.shape)
            marker_r = np.zeros_like(marker)
            # 005120_318820_083700.tif.marker
            file = file.split('.',1)[0]
            # 005120_318820_083700
            block_y,block_x,block_z = list(map(int,file.split('_')))
            block_x = block_x/20*2
            block_y = block_y/20*2
            block_z = block_z/20*2
            # print(block_x,block_y,block_z)
            marker_r[:,0] = marker[:,0]*2 + block_x
            marker_r[:,1] = marker[:,1]*2 + block_y
            marker_r[:,2] = marker[:,2]*2 + block_z
            # print(marker)
            save_marker_list(save_path,marker_r)
            #break
        whole_marker = read_marker(save_path)
        marker1 = DBScan(whole_marker,self.DB_eps,self.DB_MP)
        save_marker_list(save_path+'DBSCAN.marker',marker1)

    def start(self):
        for data in tqdm(self.dataset):
            image,raw_image_size,file_name = data
            prdict_label = self.segment(image,raw_image_size)
            for b in range(len(prdict_label)):
                self.connected_component(prdict_label[b], file_name[b])

        self.Convert_coordinates()


if __name__ == '__main__':
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default='', help='image path')
    parser.add_argument('--configuration_path', '-c', type=str, default='', help='configuration path')
    parser.add_argument('--output_path', '-o', type=str, default=' ', help='output path')
    args = parser.parse_args()

    Segmentor = whole_brain_soma_segment(args.input_path, args.configuration_path, args.output_path)
    Segmentor.start()
