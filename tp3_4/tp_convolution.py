import numpy as np
import cv2 as cv
import math


"""
    Compute a mean filter of size 2k+1.

    Pixel values outside of the image domain are supposed to have a zero value.
"""

def meanFilter(image, k):
    res = np.zeros(image.shape, dtype=float)
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            top_limit = max(0, i-k)
            bottom_limit = min(rows-1, i+k)
            left_limit = max(0, j-k)
            right_limit = min(cols-1, j+k)

            neighborhood = image[top_limit:bottom_limit+1, left_limit:right_limit+1]
            res[i,j] = np.mean(neighborhood)

    return res.astype(np.uint8)


"""
    Compute the convolution of a float image by kernel.
    Result has the same size as image.
    
    Pixel values outside of the image domain are supposed to have a zero value.
"""
def convolution(image, kernel):
    res = np.array(image, dtype=float)
    krows, kcols = kernel.shape
    rows, cols = image.shape
    offset_r = krows//2
    offset_c = kcols//2

    for i in range(rows):
        for j in range(cols):
            input = 0

            for m in range(krows):
                for n in range(kcols):
                    x = i + m-offset_r
                    y = j + n-offset_c

                    if 0<=x<rows and 0<=y<cols:
                        input += image[x,y] * kernel[m,n]
            
            res[i,j] = input
            
    return res

"""
    Compute the sum of absolute partial derivative according to Sobel's method
"""
def edgeSobel(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]])

    df_dx = convolution(image, kernel_x)
    df_dy = convolution(image, kernel_y)

    res = np.abs(df_dx) + np.abs(df_dy)
    return np.clip(res, 0, 255).astype(np.uint8)


"""
    Performs a bilateral filter with the given spatial smoothing kernel 
    and a intensity smoothing of scale sigma_r.
"""

def gaussian(x, sigma2):
    return 1.0 / (2 * math.pi * sigma2) * np.exp(-x * x / (2 * sigma2))

def bilateralFilter(image, kernel, sigma_r):
    res = np.array(image, dtype=float)
    rows, cols = image.shape
    k_h, k_w = kernel.shape
    
    pad_h = k_h//2
    pad_w = k_w//2
    
    for y in range(rows):
        for x in range(cols):
            
            pixel_val = image[y+pad_h, x+pad_w] 
            sum_val = 0.0
            sum_weight = 0.0
            


    return res.astype(np.uint8)




"""
    Compute a median filter of the input float image.
    The filter window is a square of (2*size+1)*(2*size+1) pixels.

    Values outside image domain are ignored.

    The median of a list l of n>2 elements is defined as:
     - l[n/2] if n is odd 
     - (l[n/2-1]+l[n/2])/2 is n is even 
"""
def median(image, size):
    res = np.array(image, dtype=float)
    rows, cols = image.shape
    k = size//2

    for i in range(rows):
        for j in range(cols):
            top_limit = max(0, i-k)
            bottom_limit = min(rows-1, i+k)
            left_limit = max(0, j-k)
            right_limit = min(cols-1, j+k)

            neighborhood = []

            for x in range(top_limit, bottom_limit+1):
                for y in range(left_limit, right_limit+1):
                    neighborhood.append(image[x,y])
            
            neighborhood.sort()
            n = len(neighborhood)

            if n%2 ==0 :
                res[i,j] = (float(neighborhood[n//2 - 1]) + float(neighborhood[n//2])) /2.0
            else:
                res[i,j] = neighborhood[n//2]

    return res.astype(np.uint8)



############################### MAIN


############ IMAGES
camera = cv.imread("camera.png", cv.IMREAD_GRAYSCALE)
cat = cv.imread("cat.jpg", cv.IMREAD_GRAYSCALE)
camera_bruit = cv.imread("camera_bruit_poivre_et_sel.png", cv.IMREAD_GRAYSCALE)


############ KERNELS
rehausseur = np.array([[0,-1,0],[-1,5, -1], [0,-1,0]])
moyenneur = (1/9) * np.array([[1,1,1],[1,1,1], [1,1,1]])
gaussian_kernel = (1/16) * np.array([[1,2,1],[2,4, 2], [1,2,1]])

############ TESTS

## MEAN
img_mean = meanFilter(cat, 3)

## CONVOLUTION
img_convolution_moyenneur = convolution(cat, moyenneur).astype(np.uint8)
img_convolution_rehausseur = convolution(cat, rehausseur).astype(np.uint8)

## SOBEL
img_sobel = edgeSobel(cat)


"""
## BILATERAL
img_bilateral = bilateralFilter(cat, gaussian_kernel, 30.0)
"""

## MEDIAN
img_median = median(camera_bruit, 3)



############ WRITE RESULTS
cv.imwrite("res/mean.png", img_mean)
cv.imwrite("res/convolution_moyenneur.png", img_convolution_moyenneur)
cv.imwrite("res/convolution_rehausseur.png", img_convolution_rehausseur)
cv.imwrite("res/sobel.png", img_sobel)
cv.imwrite("res/median.png", img_median)