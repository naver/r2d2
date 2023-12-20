import os
import time
import csv
import cv2
import torch
import numpy as np


SEED = 123
np.random.seed(SEED)

'''
-------------------------------------------
|                                         | 
|    (x1, y1)                             |
|      ------------------------           |
|      |                      |           |
|      |                      |           | 
|      |         ROI          |           |  
|      |                      |           |   
|      |                      |           |   
|      |                      |           |       
|      ------------------------           |   
|                           (x2, y2)      |    
|                                         |             
|                                         |             
|                                         |             
-------------------------------------------
'''

def compute_svd(data_dir, N, tau, device):
    '''
    Input: 
        data_dir: directory of the image
        N: size of the ROI
        tau: relaxation of singular values
        device: CPU or GPU
    '''
    ## Load Image
    img= cv2.imread(data_dir, 0)
    img=img.astype(float)
    print(img.shape)
    height, width= img.shape
    ## Create a one matrix for I2 representation
    SF= torch.ones(height, width)

    img_t= torch.from_numpy(img)
    img_t.to(device)
    ROI_number= 0
    for i in range (1, height):
        for j in range(1, width):

            ROI= img_t[i:i+N, j:j+N]
            # print(ROI.shape)
            u, s, v_t = torch.linalg.svd(ROI, full_matrices= False) ## SVD: ROI-> [mxn]; u-> [mxm]; s-> [mxn]; v_t-> [nxn] 
            ## relaxation of singular values
            s+= tau

            ## Normalize the largest singular values
            index_max= torch.argmax(s)
            s_sum= torch.sum(s)
            SF[i,j]= s[index_max]/s_sum

            ROI_number+= 1

    svd_img= SF.cpu().detach().numpy()
    ## Scale to 0 -255
    svd_img_scaled = ((svd_img - svd_img.min()) * (1/(svd_img.max() - svd_img.min()) * 255)).astype('uint8')

    return svd_img_scaled


def main():
    start = time.time()
    torch.manual_seed(0)
    ## To do: Use pytorch SVD on GPU
    ## SVDFace Hyperparameters
    # N= 10
    # tau= 80

    ## SVD Face paper settings
    N= 3
    # tau= 5
    # tau= 10
    tau= 20
    
    data_dir= './data/oxbuild_images-v1/'

    for filename in os.listdir(data_dir):
    # for filename in ls_data[0]:
        filename= data_dir.split('/')[-1]
        imagename= filename.split('.')[0]
        print(imagename)
        dir_filename= data_dir+'{}' .format(filename)
        print(dir_filename)

        ## Compute SVD Transform, CUDA for pytorch
        use_cuda= torch.cuda.is_available()
        device= torch.device("cuda:0" if use_cuda else "cpu")
        svd_img_scaled= compute_svd(dir_filename, N, tau, device)

        # Writing the SVDFace image
        filename1 = './results/oxbuild_images-v1_svd_n3t20/{}_SVDtau{}.jpg' .format(imagename, tau)
        cv2.imwrite(filename1, svd_img_scaled)
        print('Image saved: {}' .format(filename1))

    end = time.time()
    print('Time elapsed:\t',end - start)

## Uncomment to resume from a checkpoint (csv file)
    ## SVD Face paper settings
    N= 3
    tau= 20
    
    data_dir= './data/oxbuild_images-v1/'
    todo_dir= './results/to_do.csv'

    ## Read the csv file
    ls_data= []
    with open(todo_dir, newline='') as f:
        reader = csv.reader(f)
        ls_data = list(reader)
    print(len(ls_data[0]))

    for filename in ls_data[0]:
        imagename= filename.split('.')[0]
        print(imagename)
        dir_filename= data_dir+'{}' .format(filename)
        print(dir_filename)

        ## Compute SVD Transform, CUDA for pytorch
        use_cuda= torch.cuda.is_available()
        device= torch.device("cuda:0" if use_cuda else "cpu")

        svd_img_scaled= compute_svd(dir_filename, N, tau, device)

        # Writing the SVDFace image
        filename1 = './results/oxbuild_images-v1_svd_n3t20/{}_SVDtau{}.jpg' .format(imagename, tau)
        cv2.imwrite(filename1, svd_img_scaled)
        print('Image saved: {}' .format(filename1))

    end = time.time()
    print('Time elapsed:\t',end - start)


if __name__ == '__main__':
    main()