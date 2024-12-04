import sys
sys.path.append('.')
import argparse
import os
from scipy.ndimage import grey_closing,grey_opening,median_filter
from io1 import read_image,save_image,read_configuration
#from model.DTANET1 import DTANET
from model.DTANET import DTANET
from utils import *

code_path = r"F:\neuron_reconstruction_system\D_LSNARS\singel_neuron_reconstruction\src\python"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Neuron_Segment():
    def __init__(self,input_path, configuration_path, output_path):
        self.input_path = input_path
        self.result_out_path = input_path + '_seg.tif'

        self.block_size = [64, 64, 64]
        self.step = [48, 48, 48]

        confi = read_configuration(configuration_path)
        self.model_name = confi['segmentMethod']
        self.model_path = os.path.join(code_path,'weight',self.model_name)

        if "DTANET" in self.model_name:
            self.model = DTANET(in_dim=1, class_num=2, num_filters=32)
            self.model_path = self.model_path + '_1_48_20.pkl'
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(checkpoint)
        self.model.to(device).eval()

    def load_image(self):
        image = read_image(self.input_path)
        ###########################
        #image = grey_closing(image, [3, 3, 2])
        image = image/image.max()
        image = np.power(image, 1.0)
        # image = image / 255.0
        ##########################

        image = torch.from_numpy(image).float()
        image = image.unsqueeze(dim=0).unsqueeze(dim=0)
        return image

    def start(self):
        image = self.load_image()
        image_size = image.shape[2:5]
        block_list,block_num,max_num = cut_image_overlap(image,self.block_size,self.step)
        output_list = []
        for i in range(block_num):
            image_block = block_list[i].to(device)
            #print(image_block.shape)
            predict_label = self.model(image_block)
            predict_label = predict_label.detach()
            output_list.append(predict_label)
        output = splice_image_overlap(output_list,block_num,max_num,image_size,self.step)
        output = output.squeeze().squeeze()
        output = output.detach().cpu().numpy()
        #################
        #output = grey_opening(output, [2, 2, 2])
        output = median_filter(output, [3, 3, 3])  #17454
        #output = grey_closing(output, [3, 3, 3])   #17454
        output = grey_closing(output, [2, 2, 2])  #17302
        #################
        save_image(output, self.result_out_path)

if __name__ == '__main__':
    print('start segment')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default='', help='image path')
    parser.add_argument('--configuration_path', '-c', type=str, default='', help='configuration path')
    parser.add_argument('--output_path', '-o', type=str, default=' ', help='output path')
    args = parser.parse_args()

    segmentor = Neuron_Segment(args.input_path, args.configuration_path, args.output_path)
    segmentor.start()

