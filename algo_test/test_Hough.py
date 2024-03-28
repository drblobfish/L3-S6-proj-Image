import numpy as np
import matplotlib.pyplot as plt
from test_traitement import *
import cv2

np.set_printoptions(linewidth=400)
def gaussian_kernel(size, sigma):
    center = size//2
    x, y = np.meshgrid(np.arange(-center, center+1), np.arange(-center, center+1))
    kernel = np.exp(-( (x**2 + y**2) / (2*sigma**2)))
    return kernel / np.sum(kernel)

def gaussian_filter(image, size, sigma):
    height = image.shape[0]
    width = image.shape[1]
    center = size//2
    kernel = gaussian_kernel(size, sigma)
    filtered_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            sum = 0
            for k in range(-center, center + 1):
                for l in range(-center, center + 1):
                    if i-k>=0 and j-l>=0 and i-k<height and j-l<width:
                        sum += image[i - k,j - l] * kernel[k + center,l + center]
            filtered_image[i,j] = sum
    return filtered_image

def sobel_filter(image):
    kernelx = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    kernely = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    height = image.shape[0]
    width = image.shape[1]
    gradientx = np.zeros_like(image,dtype=np.float32)
    gradienty = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            sumx = 0
            sumy = 0
            for k in range(-1, 1+1):
                for l in range(-1, 1+1):
                    if i-k>=0 and j-l>=0 and i-k<height and j-l<width:
                        sumx += np.int32(image[i - k,j - l] * kernelx[k + 1,l + 1])
                        sumy += np.int32(image[i - k, j - l] * kernely[k + 1, l + 1])
            gradientx[i,j] = sumx
            gradienty[i, j] = sumy
    gradient = np.hypot(gradientx, gradienty)
    theta = np.arctan2(gradientx, gradienty)
    return (gradient*255/gradient.max()), theta

def non_maximum_suppression(image, theta):
    height = image.shape[0]
    width = image.shape[1]
    filtered_image = np.zeros_like(image, dtype=np.int32)
    angle = theta*180/np.pi
    angle[angle<0] += 180

    for i in range(1,height-1):
        for j in range(1,width-1):
            q = 255
            r = 255

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = image[i, j - 1]
                q = image[i, j + 1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                r = image[i - 1, j + 1]
                q = image[i + 1, j - 1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                r = image[i - 1, j]
                q = image[i + 1, j]
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                r = image[i + 1, j + 1]
                q = image[i - 1, j - 1]

            if (image[i, j] >= q) and (image[i, j] >= r):
                filtered_image[i, j] = image[i, j]
            else:
                filtered_image[i, j] = 0
    return filtered_image


def canny(image, thresh1, thresh2):
    #Preprocessing : gray+blur
    canny_image = gaussian_filter(color_to_gray(image), 5, 1)
    pass






if __name__ == "__main__":
    image = normalize_image_data_type(scale_image(plt.imread("../Images/0.jpg"), 1000))
    image = color_to_gray(image,cmap='gray', vmin=0, vmax=255)
    plt.subplot(3,3,1)
    plt.imshow(image)

    sobel_img = sobel_filter(image)
    plt.subplot(3,3,2)
    plt.imshow(sobel_img[0],cmap='gray', vmin=0, vmax=255)

    non_max_img = non_maximum_suppression(sobel_img[0], sobel_img[1])
    plt.subplot(3,3,3)
    plt.imshow(non_max_img,cmap='gray', vmin=0, vmax=255)



    plt.show()
