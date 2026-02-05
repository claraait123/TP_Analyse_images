import numpy as np
import cv2 as cv

"""
    Compute the dilation of the input float image by the given structuring element.
     Pixel outside the image are supposed to have value 0
"""
def dilate(image, structuringElement, center=None):
    res = np.zeros(image.shape, dtype=float)
    rows, cols = image.shape
    se_rows, se_cols = structuringElement.shape

    if center is None:
        ax = se_rows // 2
        ay = se_cols // 2
    else:
        ax, ay = center

    for x in range(rows):
        for y in range(cols):
            val = 0
            for i in range(se_rows):
                for j in range(se_cols):
                    nx = x + (i - ax)
                    ny = y + (j - ay)
                    if 0 <= nx < rows and 0 <= ny < cols:
                        val = val or (image[nx, ny] != 0)
            res[x, y] = val
            
    return res


"""
    Compute the erosion of the input float image by the given structuring element.
    Pixel outside the image are supposed to have value 1.
"""
def erode(image, structuringElement, center=None):
    res = np.zeros(image.shape, dtype=float)
    rows, cols = image.shape
    se_rows, se_cols = structuringElement.shape

    if center is None:
        ax = se_rows // 2
        ay = se_cols // 2
    else:
        ax, ay = center

    for x in range(rows):
        for y in range(cols):
            val = 1
            for i in range(se_rows):
                for j in range(se_cols):
                    if structuringElement[i, j] != 0: 
                        nx = x + (i - ax)
                        ny = y + (j - ay)

                        if 0 <= nx < rows and 0 <= ny < cols:
                            neigh = 1 if image[nx, ny] != 0 else 0
                        else:
                            neigh = 0

                        val = val and neigh

            res[x, y] = val

    return res

"""
    Compute the closing of the input float image by the given structuring element.
"""
def close(image, structuringElement):
    return erode(dilate(image, structuringElement), structuringElement)

"""
    Compute the opening of the input float image by the given structuring element.
"""
def open(image, structuringElement):
    return dilate(erode(image, structuringElement), structuringElement)


"""
    Compute the morphological gradient of the input float image by the given structuring element.
"""
def morphologicalGradient(image, structuringElement):
    return dilate(image, structuringElement) - erode(image, structuringElement)



############################### MAIN


############ IMAGE
binary = cv.imread("binary.png", cv.IMREAD_GRAYSCALE)


############ TESTS

element = np.ones((3,3))

## DILATE
img_dilate_centered = dilate(binary, element)
img_dilate = dilate(binary, element, center=(0,0))


## ERODE
img_erode_centered = erode(binary, element)
img_erode = erode(binary, element, center=(0,0))

## OPEN
img_open = open(binary, element)
## CLOSE
img_close = close(binary, element)

## MORPHOLOGICAL GRADIENT
img_gradient = morphologicalGradient(binary, element)

############ WRITE RESULTS
cv.imwrite("res/dilate_centered.png", img_dilate_centered*255)
cv.imwrite("res/dilate.png", img_dilate*255)

cv.imwrite("res/erode_centered.png", img_erode_centered*255)
cv.imwrite("res/erode.png", img_erode*255)  

cv.imwrite("res/open.png", img_open*255)
cv.imwrite("res/close.png", img_close*255)
cv.imwrite("res/morphological_gradient.png", img_gradient*255)