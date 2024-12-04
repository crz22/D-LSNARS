import gc
import sys
sys.path.append('.')
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'MAX_SPLIT_SIZE_MB=128'
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from model.MCCNet_model import ClassNet
from model.VGG16_model import VGG16
from base_function import *
import torchvision.transforms as transforms

code_path = r'F:\neuron_reconstruction_system\D_LSNARS\whole_brain_somas_detection\src\python'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_images = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

class test_dataset(Dataset):
    def __init__(self,dataset_path,contrast_level,transform=None):
        super(test_dataset, self).__init__()
        self.dataset_path = dataset_path
        self.contrast_level = contrast_level
        self.transform = transform
        self.filename = os.listdir(self.dataset_path)

    def __len__(self):
        return len(self.filename)

    def load_img(self,path,filename):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = np.array(img)
            a = contrast(img,self.contrast_level)
            image = Image.fromarray(np.uint8(a))
            image = image.resize((256, 256), Image.Resampling.BILINEAR)
        return image.convert('RGB')

    def __getitem__(self, item):
        image = self.load_img(os.path.join(self.dataset_path,self.filename[item]),self.filename[item])
        if self.transform is not None:
            image = self.transform(image)
        return image,self.filename[item]

class whole_brain_soma_classifyer():
    def __init__(self,input_file_path,configuration_path,output_path,batch_size=64,transform=transform_images):
        self.input_file_path = input_file_path
        self.result_out_path = output_path
        self.transform = transform_images
        self.batch_size = batch_size
        self.save_file = 'candidate_block.txt'
        confi = read_configuration(configuration_path)
        #print(confi)
        self.CE = confi['CE'];
        self.model_name = 'MCCNet8'

        if 'MCCNet' in self.model_name:
            self.Classify_model = ClassNet(num_classes=3)
        elif 'VGG16' in self.model_name:
            self.Classify_model = VGG16()

        Classify_MODEL_PATH =code_path+'/weight/'+self.model_name+'/'+self.model_name+'300.ckpt'
        self.Classify_model.load_state_dict(torch.load(Classify_MODEL_PATH))
        self.Classify_model.eval().to(device)

    def classify(self,image):
        image = image.to(device)
        with torch.no_grad():
            outputs = self.Classify_model(image)
            outputs = torch.softmax(outputs, dim=1)
            predict_class = torch.argmax(outputs, dim=1)

        return predict_class.cpu().numpy()

    def start(self):
        candidate_block = []
        TEST_DATASET = test_dataset(self.input_file_path, self.CE, self.transform)
        dataset = DataLoader(dataset=TEST_DATASET, batch_size=self.batch_size)
        for data in dataset:
            image,file_name = data
            predict = self.classify(image)
            candiate_index = np.where(predict==0)
            for index in candiate_index[0]:
                candidate_block.append(file_name[index][:-4])

        write_in_txt(os.path.join(self.result_out_path,self.save_file),candidate_block,mode='a+')
        gc.collect()
        #assert len(candidate_block) == 0


def write_in_txt(file_path,content,mode='w+'):
    file = open(file_path, mode)
    for line in content:
        file.write(line)
        file.write('\n')
    file.close()

def read_mip(path,CE):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = np.array(img)
        a = contrast(img, CE)
        image = Image.fromarray(np.uint8(a))
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
    return image.convert('RGB')

if __name__ == '__main__':
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default='', help='image path')
    parser.add_argument('--configuration_path', '-c', type=str, default='', help='configuration path')
    parser.add_argument('--output_path', '-o', type=str, default=' ', help='output path')
    args = parser.parse_args()
    #print(args.input_path,args.configuration_path,args.output_path)
    classifyer = whole_brain_soma_classifyer(args.input_path, args.configuration_path, args.output_path)
    classifyer.start()
