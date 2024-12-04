import sys
sys.path.append('.')
import numpy as np
import torch
from skimage import measure,morphology
from scipy.ndimage import binary_dilation
from model.SFSNet3 import S_UNet
import torchvision.transforms as transforms
from io1 import save_image

code_path = r"F:\neuron_reconstruction_system\D_LSNARS\singel_neuron_reconstruction\src\python"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_images = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

def adjust_img(image):
        # image = ((image - image.min()) / (image.max() - image.min())).astype(np.float32)
        image = (image/ image.max()).astype(np.float32)

        #resize image to [256,256,256]
        original_img_size = np.array(image.shape)
        # Adjust the size of the block to ensure that it can be partitioned

        z_pad = 256-image.shape[0]
        y_pad = 256-image.shape[1]
        x_pad = 256-image.shape[2]
        '''
        z_pad = 128 - image.shape[0]
        y_pad = 128 - image.shape[1]
        x_pad = 128 - image.shape[2]
        '''
        image1 = np.pad(image, ((0, z_pad), (0, y_pad), (0, x_pad)), 'reflect')
        image1 = transform_images(image1)
        image1 = image1.permute(1, 2, 0)

        image1 = image1.unsqueeze(dim=0).unsqueeze(dim=0)

        return image1,original_img_size

def soma_segment(raw_image,image_path):
    save_path = image_path + '_somamark.tif'
    #seg_block_size = [64, 128, 128]
    seg_block_size = [64, 64, 64]
    Segment_model = S_UNet(in_dim=1, out_dim=2, num_filters=16)

    Segment_MODEL_PATH = code_path+'/weight/SFSUNet12_200.pt'
    Segment_model.load_state_dict(torch.load(Segment_MODEL_PATH,map_location='cpu')['model'])
    Segment_model.eval().to(device)

    image, raw_image_size = adjust_img(raw_image)
    image = image.to(device)
    batch, c, w, h, l = image.shape
    image_down = torch.nn.functional.interpolate(image, size=[w//2, h//2,l//2])

    num = np.array([w, h, l]) // 2//seg_block_size

    #buf_0 = np.zeros([batch, w//2, h//2, l//2], dtype=np.uint8)
    buf_0 = torch.zeros([batch, w // 2, h // 2, l // 2])
    for i in range(num[2]):
        for j in range(num[1]):
            for k in range(num[0]):
                image_block = image_down[:, :, seg_block_size[0] * k:seg_block_size[0] * (k + 1),
                              seg_block_size[1] * j:seg_block_size[1] * (j + 1),
                              seg_block_size[2] * i:seg_block_size[2] * (i + 1)]
                #print(image_block.shape)
                if image_block.max() == 0 or image_block.max() == image_block.min():
                    continue
                output = Segment_model(image_block)  # (batch*2*64*128*128)
                output_argmax = torch.argmax(output, dim=1)  # (batch*64*128*128)
                #print(output.shape, output_argmax.shape)
                ppl = output_argmax.float()
                ppl = ppl.detach().cpu()#.numpy()

                buf_0[:, seg_block_size[0] * k: seg_block_size[0] * (k + 1),
                seg_block_size[1] * j: seg_block_size[1] * (j + 1),
                seg_block_size[2] * i: seg_block_size[2] * (i + 1)] = ppl
                del output
                del output_argmax

    buf_0 = torch.nn.functional.interpolate(buf_0.unsqueeze(0),size =[w,h,l])
    buf_0 = buf_0.numpy()
    somamark = buf_0[0,0,0:raw_image_size[0],0:raw_image_size[1],0:raw_image_size[2]] #[0,1]
    #print(somamark.shape)
    if somamark.max()>0:
        structuring_element = morphology.ball(radius=7)
        somamark = binary_dilation(somamark,structuring_element)
    #print(somamark.shape, raw_image_size)
    #if somamark.max() != 0:
    #    save_image(somamark,save_path)
    return somamark

def find_soma_with_mark(image,image_path):
    # Mark the input 3D image
    centarl_point_x = image.shape[0]//2
    centarl_point_y = image.shape[1]//2
    centarl_point_z = image.shape[2]//2
    label, num = measure.label(image, connectivity=1, return_num=True)  # 1代表４连通，２代表８连通
    if num < 1:
        print("no soma in this block ")
        return np.zeros_like(image)
    # obtain soma location
    region = measure.regionprops(label)
    num_list = [i for i in range(1, num + 1)]
    for i in num_list:
        if region[i-1].area>7000: #17545/17302: 3000
            print(region[i-1].area)
            if i in label[centarl_point_x-20:centarl_point_x+20,centarl_point_y-20:centarl_point_y+20,centarl_point_z-20:centarl_point_z+20]:
                label[label == i] = 1
            else:
                label[label == i] = -1
        else:
            label[label == i] = 0
    save_image(label,savepath=image_path+"_somamark.tif")
    return label

