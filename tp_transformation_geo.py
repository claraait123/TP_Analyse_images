import numpy as np
import cv2 as cv
import math


"""    
    Transpose the input image,
    ie. performs a planar symmetry according to the
    first diagonal (upper left to lower right corner).
"""

def transpose(image):
    rows, cols = image.shape
    res = np.zeros((cols, rows), dtype=float)

    for i in range(rows):
        for j in range(cols):
            res[j,i] = image[i,j]

    return res.astype(np.uint8)




"""
    Compute the value of a nearest neighbour interpolation
    in image Mat at position (x,y)
"""
def interpolate_nearest(image, x, y):
    rows, cols = image.shape

    x = int(round(x))
    y = int(round(y))

    if 0<=int(y)<rows and 0<=int(x)<cols:
        v = float(image[int(y), int(x)])
    else:
        v=0.0

    return v



"""
    Multiply the image resolution by a given factor using the given interpolation method.
    If the input size is (h,w) the output size shall be ((h-1)*factor, (w-1)*factor)
"""
def expand(image, factor, interpolationFunction):
    assert(factor>0)
    rows, cols = image.shape
    new_rows = (rows-1)*factor
    new_cols = (cols-1)*factor
    res = np.zeros((new_rows, new_cols),dtype=float)

    for i in range(new_rows):
        for j in range(new_cols):
            y = float(i)/factor
            x = float(j)/factor

            res[i,j] = interpolationFunction(image, x, y)

    return res.astype(np.uint8)


"""
    Compute the value of a bilinear interpolation in image Mat at position (x,y)
"""
def interpolate_bilinear(image, x, y):
    rows, cols = image.shape

    x = int(round(x))
    y = int(round(y))
    x1 = int(math.floor(x))
    y1 = int(math.floor(y))
    x2 = x1 + 1
    y2 = y1 + 1

    if x1<0 or x2>=cols or y1<0 or y2>=rows:
        if 0<=y< rows and 0<=x<cols:
            return float(image[y,x])
        return 0.0
    
    dx = x-x1
    dy = y-y1

    v = (1-dx) * (1-dy) * float(image[y1,x1])
    v += dx * (1-dy) * float(image[y1,x2])
    v += (1-dx) * dy * float(image[y2,x1])
    v += dx * dy * float(image[y2,x2])

    return v




"""
    Performs a rotation of the input image with the given angle (clockwise) and the given interpolation method.
    The center of rotation is the center of the image.

    Ouput size depends of the input image size and the rotation angle.

    Output pixels that map outside the input image are set to 0.
"""
def rotate(image, angle, interpolationFunction):
    rows, cols = image.shape

    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    new_cols = int(abs(cols * cos_a) + abs(rows * sin_a))
    new_rows = int(abs(cols * sin_a) + abs(rows * cos_a))
    res = np.zeros((new_rows, new_cols), dtype=np.float32)

    cx = cols/2
    cy = rows/2
    ncx = new_cols/2
    ncy = new_rows/2

    for i in range(new_rows):
        for j in range(new_cols):
            x = j-ncx
            y = i-ncy

            rotation_x = x * cos_a + y * sin_a + cx
            rotation_y = -x * sin_a + y * cos_a + cy

            res[i,j] = interpolationFunction(image, rotation_x, rotation_y)

    return res.astype(np.uint8)


############################### MAIN


############ IMAGES
cat = cv.imread("cat.jpg", cv.IMREAD_GRAYSCALE)
 

############ TESTS

## TRANSPOSE    
img_transpose = transpose(cat)


## EXPAND
img_expand_nearest = expand(cat, 3, interpolate_nearest)
img_expand_bilinear = expand(cat, 3, interpolate_bilinear)


## ROTATE
img_rotate_nearest = rotate(cat, 45, interpolate_nearest)
img_rotate_bilinear = rotate(cat, 45, interpolate_bilinear)



############ WRITE RESULTS
cv.imwrite("res/transpose.png", img_transpose)
cv.imwrite("res/expand_nearest.png", img_expand_nearest)
cv.imwrite("res/expand_bilinear.png", img_expand_bilinear)
cv.imwrite("res/rotate_nearest.png", img_rotate_nearest)
cv.imwrite("res/rotate_bilineare.png", img_rotate_bilinear)
