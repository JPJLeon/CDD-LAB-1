# Para obtener imagen en collab. Reemplazar URL por imagen a gusto.
import urllib.request
from PIL import Image
import os
url = "https://devblogs.nvidia.com/wp-content/uploads/2012/10/CUDA_Cube_1K.jpg"
urllib.request.urlretrieve(url, "img.jpg")
img = Image.open(r'img.jpg')
img = img.resize((2400, 2400), Image.ANTIALIAS)
img.save(r'img.png')
os.remove("img.jpg")

import matplotlib.pyplot as mpimg
import numpy as np

def RGBtoTXT(name):
    img = mpimg.imread(name+'.png')
    M,N,_ = img.shape
    RGB = np.array([img[:,:,i].reshape(M*N) for i in range(3)])
    np.savetxt(name+'.txt', RGB, fmt='%.8f', delimiter=' ', header='%d %d'%(M,N), comments='')

def TXTtoRGB(name):
    RGB = np.loadtxt(name+'.txt', delimiter=' ', skiprows = 1)
    with open(name+'.txt') as imgfile:
        M,N = map(int,imgfile.readline().strip().split())
    img = np.ones((M,N,4))
    for i in range(3):
        img[:,:,i] = RGB[i].reshape((M,N)) 
    mpimg.imsave(name+'_fromfile.png', img)