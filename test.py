import os
import cv2
import torch
import numpy as np
from kernal import togray,laplacian,gridImg,gaussion,conv2d

#torch laplacian
def torch_lap_img(img):
    input = togray(img)
    data = torch.from_numpy(input).unsqueeze(0).unsqueeze(0)
    weight = laplacian(0)
    out = torch.abs(conv2d(data,weight))
    r = out.squeeze(0).squeeze(0).numpy()
    return r

#opencv laplacian
def LaplacianTransform(img):
    dst_img = cv2.Laplacian(img, cv2.CV_32F)
    laplacian_edge = cv2.convertScaleAbs(dst_img)
    return laplacian_edge
    
#manual laplacian
def LaplacianOperator(img,op_type=0):
    ops = [[[0,1,0],[1,-4,1],[0,1,0]],[[1,1,1],[1,-8,1],[1,1,1]]]
    operator = np.array(ops[op_type])
    h,w = img.shape
    result = img.copy()#np.zeros((h,w))
    for r in range(1,h-1):
        for c in range(1,w-1):
            result[r,c] = abs(np.sum(img[r-1:r+2,c-1:c+2] * operator))
    return np.uint8(result)
    
def get_lap_img(img_path):       
    img = cv2.imread(img_path,0)
    print("img:",img)
    torch_lap = lap_img(img)
    cv_lap = LaplacianTransform(img)
    manual_lap = LaplacianOperator(img)
    
    print("lap1:",lap1)
    print("lap2:",lap2)
    print("lap3:",lap3)
    
if __name__ == "__main__":
    img_path = "image/101.jpg"
    get_lap_img(img_path)
