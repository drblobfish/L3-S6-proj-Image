import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def color_to_gray(image):
    height = image.shape[0]
    width = image.shape[1]
    gray_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            gray_image[i][j] = int((image[i][j][0] + image[i][j][1] * 2 + image[i][j][2]) / 4)
    return gray_image

def histogram(image):
    height = image.shape[0]
    width = image.shape[1]
    hist = np.zeros(256)
    for i in range(height):
        for j in range(width):
            hist[int(image[i][j])] += 1
    return hist

def otsu(image, limit):
    height = image.shape[0]
    width = image.shape[1]
    otsu_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if (image[i][j] > limit):
                otsu_image[i][j] = 0
            else:
                otsu_image[i][j] = 255
    return otsu_image

image = color_to_gray(mpimg.imread("./Images/67.JPG"))
mpimg.imsave("test.png", image, cmap="gray", vmin=0, vmax=255)
mpimg.imsave("test_otsu.png", otsu(image, 60), cmap="gray", vmin=0, vmax=255)
plt.plot(histogram(image), '.')
plt.show()