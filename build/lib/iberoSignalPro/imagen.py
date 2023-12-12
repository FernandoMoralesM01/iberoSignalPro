import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2

#create_colormap(Nlevel = 256)
#crea un mapa de color en escala de grises.

def create_colormap(Nlevel=256):
  mycolormap = np.linspace(0,1,Nlevel)[:,np.newaxis]
  mycolormap = mycolormap@np.ones((1,3))
  mycolormap = ListedColormap(mycolormap)
  return mycolormap

def normalize_image(Image):
  Image = (Image - Image.min())/ (Image.max() - Image.min())
  return Image

def KernelCirc(N):
  return [[np.round(np.sqrt(np.abs(j-int(N/2))**2+np.abs(i-int(N/2))**2),3) for i in range (0,N)] for j in range (0,N)]

def explain_histogram (lena = np.zeros((100,100)), mycolormap=None):
  plt.rcParams['figure.figsize'] = (10, 4)
  fig = plt.figure()

  xbins = np.linspace (0, 1, 256)
  # Grafica del histograma
  ax1 = fig.add_subplot(1,3,1)
  sbn.histplot(lena.ravel(), bins = xbins, ax = ax1, )
  ax1.grid()
  ax1.set_xlabel('nivel gris')
  xx, bins = np.histogram(lena.ravel(), bins = xbins )

  # grafica de la cdf
  ax2 = fig.add_subplot(1,3,2)
  ax2.plot(bins[1:], np.cumsum(xx), linewidth = 3)
  ax2.grid()
  ax2.set_xlabel('nivel gris')

  # imagen
  ax3 = fig.add_subplot(1,3,3)
  ax3.imshow(lena, cmap = mycolormap)
  plt.show()

# show_two_normalized(Ip,lena, mycolormap = 'gray'):
# muestra dos imagenes en el mismo plot, ambas con un mapa de color elegido

def show_two_normalized(Ip,lena, mycolormap = 'gray'):
  plt.subplot(1, 2, 1)
  plt.imshow(Ip, cmap=mycolormap)
  plt.subplot(1, 2, 2)
  plt.imshow(lena, cmap=mycolormap)
  plt.show()

def equalize_image(image, minVal, maxVal):
  sImage = np.where(image > minVal, image, minVal)
  sImage = np.where(image < maxVal, sImage, maxVal)
  return sImage

def thresh_mask(Image, minVal = 0, maxVal = 1):
  img_thresh1 = (Image>minVal).astype(float)
  img_thresh2 = (Image<maxVal).astype(float)
  return img_thresh1 + img_thresh2

def im2cvtype(Image):
  return Image.astype(np.uint8)

# FILTROS

def highboost (N = 8, Ftype = 'dir'):
  if(Ftype == 'dir'):
    arr = np.array([[-1, -1, -1], [-1, N, -1], [-1, -1, -1]])
  if(Ftype == 'lap'):
    arr = np.array([[0, -1, 0], [-1, N, -1], [0, -1, 0]])
  return arr

def butter (x,  ws = 2, N = 1, type_b = 'lp'):
  if(type_b == 'lp'):
    butter_fun = (1 / np.sqrt(1 + (x / ws) ** N ))
  if(type_b == 'hp'):
    butter_fun = (1 / np.sqrt(1 + (ws / x) ** N ))
  return butter_fun

def imfilt (Image, filt):
    
    g = np.fft.fft2(Image)
    g = np.fft.fftshift(g)
    g1 =   (g * filt)
    return abs(np.fft.ifft2(g1))

def Coords_Kernel(N = 5):
  xx = np.arange(-N,N+1)
  xx,yy = np.meshgrid(xx,xx)

  Coords = np.sqrt(xx**2 + yy**2)
  return Coords

def conv2d(Image, func):
    s = func.shape + tuple(np.subtract(Image.shape, func.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(Image, shape = s, strides = Image.strides * 2)
    res = np.einsum('ij,ijkl->kl', func, subM)
    res = cv2.resize(res, (Image.shape), interpolation=cv2.INTER_LINEAR)
    return normalize_image(res)

def mexhat (NKernel = 5, a = 1):
  x = Coords_Kernel(NKernel)
  A = 2/(np.sqrt(3*a)*(np.pi**0.25))
  mexhat_fun = A * (1 - (x/a)**2) * np.exp(-0.5*(x/a)**2)
  return mexhat_fun

def Gauss (NKernel = 5, mu= 0, sigma = 1):
  x = Coords_Kernel(NKernel)
  Gauss_fun = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp( -1/2 * ((x - mu)/sigma) ** 2)
  return Gauss_fun