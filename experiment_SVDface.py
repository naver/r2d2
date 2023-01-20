import time
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
    data_dir = '/home/pfvaldez/Development/r2d2/data/aachen/images_upright/db/2.jpg'
    N= 10
    # tau= 5
    tau= 80

    ## CUDA for pytorch
    use_cuda= torch.cuda.is_available()
    device= torch.device("cuda:0" if use_cuda else "cpu")

    svd_img_scaled= compute_svd(data_dir, N, tau, device)

    # Filename
    filename1 = './results/SVDface_tau{}.jpg' .format(tau)
    # Using cv2.imwrite() method Saving the image
    cv2.imwrite(filename1, svd_img_scaled)

    # cv2.imshow('SVDface_tau{}' .format(tau),svd_img_scaled)
    # ## Add a waitkey
    # cv2.waitKey(0)
    # ## Destroy/close all windows
    # cv2.destroyAllWindows()

    end = time.time()
    print('Time elapsed:\t',end - start)

if __name__ == '__main__':
    main()