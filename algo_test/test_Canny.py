import numpy as np
import matplotlib.pyplot as plt
from test_traitement import *
import cv2

#source : https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
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
    #Réduction de l'épaisseur des contours
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

def double_threshold(image, low_thresh, high_thresh):
    #Double seuillage : <low = suppression, low< <high = pixel faible, >high = pixel fort

    height = image.shape[0]
    width = image.shape[1]
    filtered_image = np.zeros_like(image, dtype=np.int32)

    #Obtention des coordonnées des pixels forts et faibles
    strong_i, strong_j = np.where(image>=high_thresh)
    weak_i, weak_j = np.where((image>=low_thresh) & (image<high_thresh))

    #Valeurs des pixels forts et faibles
    strong = np.int32(255)
    weak = np.int32(25)
    #Affectation des valeurs aux coordonnées
    filtered_image[strong_i, strong_j] = strong
    filtered_image[weak_i, weak_j] = weak
    return filtered_image, strong, weak

def hysteresis(image, strong, weak):
    #On transforme les pixels faibles en pixels fort s'ils sont connectés à un pixel fort
    height = image.shape[0]
    width = image.shape[1]
    for i in range(1, height-1):
        for j in range(1,width-1):
            if image[i,j]==weak:
                if (image[i-1,j-1]==strong or image[i-1,j]==strong or image[i-1,j+1]==strong or
                    image[i, j-1]==strong or image[i,j+1]==strong or
                    image[i+1, j-1]==strong or image[i+1,j]==strong or image[i+1,j+1]==strong):
                    image[i,j] = strong
                else:
                    image[i,j] = 0
    return image





def canny(image, low_thresh, high_thresh):
    gaussian_img = gaussian_filter(color_to_gray(image), 5, 1)
    sobel_img, theta = sobel_filter(gaussian_img)
    non_max_img = non_maximum_suppression(sobel_img, theta)
    thresh_img, strong, weak = double_threshold(non_max_img, low_thresh, high_thresh)
    hysteresis_img = hysteresis(thresh_img, strong, weak)

    return hysteresis_img






if __name__ == "__main__":
    image = normalize_image_data_type(scale_image(plt.imread("../Images/0.jpg"), 1000))
    image = color_to_gray(image)
    plt.subplot(2,3,1)
    plt.imshow(image,cmap='gray', vmin=0, vmax=255)
    plt.title("image gris")

    gaussian_img = gaussian_filter(image, 5, 1)
    plt.subplot(2,3,2)
    plt.imshow(gaussian_img,cmap='gray', vmin=0, vmax=255)
    plt.title("Après filtre gaussien")

    sobel_img = sobel_filter(image)
    plt.subplot(2,3,3)
    plt.imshow(sobel_img[0],cmap='gray', vmin=0, vmax=255)
    plt.title("Après filtre de Sobel")

    non_max_img = non_maximum_suppression(sobel_img[0], sobel_img[1])
    plt.subplot(2,3,4)
    plt.imshow(non_max_img,cmap='gray', vmin=0, vmax=255)
    plt.title("Après non-max ")

    double_thresh_img = double_threshold(non_max_img, 50, 100)
    plt.subplot(2,3,5)
    plt.imshow(double_thresh_img[0],cmap='gray', vmin=0, vmax=255)
    plt.title("Après double seuillage")

    hysteresis_img = hysteresis(double_thresh_img[0], double_thresh_img[1], double_thresh_img[2])
    plt.subplot(2,3,6)
    plt.imshow(hysteresis_img,cmap='gray', vmin=0, vmax=255)
    plt.title("Après hysteresis")

    plt.show()
