import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd
import os

def scale_image(image, target_size):
    height = image.shape[0]
    width = image.shape[1]
    if height <= target_size and width <= target_size:
        print("The image is already smaler than the target size.\nScaling will not proceed.")
        return image
    if height > width:
        ratio = target_size / height
    else:
        ratio = target_size / width
    new_height = int(height * ratio)
    new_width = int(width * ratio)
    if len(image.shape) > 2:
        new_image = np.zeros((new_height, new_width, image.shape[2]))
    else:
        new_image = np.zeros((new_height, new_width))
    for i in range(new_height):
        for j in range(new_width):
            new_image[i][j] = image[int(i / ratio)][int(j / ratio)]
    print("Image Scaling : Done")
    return new_image

def normalize_image_data_type(image):
    height = image.shape[0]
    width = image.shape[1]
    if len(image.shape) > 2 and image.shape[2] >= 3:
        new_image = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
        if np.max(image) <= 1:
            for i in range(height):
                for j in range(width):
                    new_image[i][j][0] = int(image[i][j][0] * 255)
                    new_image[i][j][1] = int(image[i][j][1] * 255)
                    new_image[i][j][2] = int(image[i][j][2] * 255)
        else:
            for i in range(height):
                for j in range(width):
                    new_image[i][j][0] = int(image[i][j][0])
                    new_image[i][j][1] = int(image[i][j][1])
                    new_image[i][j][2] = int(image[i][j][2])
    else:
        new_image = np.zeros((height, width), dtype=np.uint8)
        if np.max(image) <= 1:
            for i in range(height):
                for j in range(width):
                    new_image[i][j] = int(image[i][j] * 255)
        else:
            for i in range(height):
                for j in range(width):
                    new_image[i][j] = int(image[i][j])
    print("Image Data Type Normalization : Done")
    return new_image

def color_to_gray(image):
    if len(image.shape) < 3 or image.shape[2] < 3:
        print("Image is already in gray scale.\nThe conversion of color space will not proceed.")
        return image
    height = image.shape[0]
    width = image.shape[1]
    gray_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            red_value = image[i][j][0] * 299
            green_value = image[i][j][1] * 587
            blue_value = image[i][j][2] * 114
            gray_value = int((red_value + green_value + blue_value) / 1000)
            gray_image[i][j] = gray_value
    print("Convertion of image color space to gray scale : Done")
    return gray_image

def generate_histogram(image):
    height = image.shape[0]
    width = image.shape[1]
    hist = np.zeros(256, dtype=np.uint32)
    for i in range(height):
        for j in range(width):
            hist[image[i][j]] += 1
    return hist

def generate_cumulative_histogram(hist):
    cumulative_hist = np.zeros(256, dtype=np.uint32)
    cumulative_hist[0] = hist[0]
    for i in range(1, cumulative_hist.shape[0]):
        cumulative_hist[i] = cumulative_hist[i - 1] + hist[i]
    return cumulative_hist

def generate_normalized_cumulative_histogram(cumulative_hist):
    normalized_cumulative_hist = np.zeros(256, dtype=np.float32)
    for i in range(normalized_cumulative_hist.shape[0]):
        normalized_cumulative_hist[i] = cumulative_hist[i] / cumulative_hist[-1]
    return normalized_cumulative_hist

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

def equalize_image(image):
    height = image.shape[0]
    width = image.shape[1]
    equalized_image = np.zeros((height, width), dtype=np.uint8)
    hist = generate_histogram(image)
    cumulative_hist = generate_cumulative_histogram(hist)
    normalized_cumulative_hist = generate_normalized_cumulative_histogram(cumulative_hist)
    for i in range(height):
        for j in range(width):
            equalized_image[i][j] = (normalized_cumulative_hist[image[i][j]] * 255).astype(np.uint8)
    print("Image Equalization : Done")
    return equalized_image
    
def k_means(image):
    height = image.shape[0]
    width = image.shape[1]
    center_image = np.zeros((height, width), dtype=np.uint8)
    k_means_image = np.zeros((height, width), dtype=np.bool_)
    center_1 = rd.randint(0, 255)
    center_2 = rd.randint(0, 255)
    while center_2 == center_1:
        center_2 = rd.randint(0, 255)
    center_1_iteration_precedante = None
    center_2_iteration_precedante = None
    while center_1 != center_1_iteration_precedante and center_2 != center_2_iteration_precedante:
        number_center_1 = 0
        number_center_2 = 0
        sum_center_1 = 0
        sum_center_2 = 0
        for i in range(height):
            for j in range(width):
                if (abs(image[i][j] - center_1) < abs(image[i][j] - center_2)):
                    center_image[i][j] = 1
                    sum_center_1 += image[i][j]
                    number_center_1 += 1
                else:
                    center_image[i][j] = 2
                    sum_center_2 += image[i][j]
                    number_center_2 += 1
        if number_center_1 == 0:
            moyenne_center_1 = rd.randint(0,255)
        else:
            moyenne_center_1 = sum_center_1 / number_center_1
        if number_center_2 == 0:
            moyenne_center_2 = rd.randint(0, 255)
        else:
            moyenne_center_2 = sum_center_2 / number_center_2
        center_1_iteration_precedante = center_1
        center_2_iteration_precedante = center_2
        center_1 = int(moyenne_center_1)
        center_2 = int(moyenne_center_2)
    for i in range(height):
        for j in range(width):
            if center_image[i][j] == 1:
                k_means_image[i][j] = 0
            elif center_image[i][j] == 2:
                k_means_image[i][j] = 1
    print("Application of Kmeans : Done")
    return k_means_image

def median_filter(image, filter_size):
    height = image.shape[0]
    width = image.shape[1]
    filtered_image = np.zeros((height, width), dtype=np.bool_)
    for i in range(height):
        for j in range(width):
            pixel_values = np.zeros((filter_size*filter_size))
            counter = 0
            for k in range(i - int(filter_size / 2), i + int(filter_size / 2) + 1):
                for l in range(j - int(filter_size / 2), j + int(filter_size / 2) + 1):
                    if k >= height:
                        k = k - height
                    if l >= width:
                        l = l - width
                    pixel_values[counter] = image[k][l]
                    counter += 1
            filtered_image[i][j] = np.median(pixel_values)
    print("Application of Median Filter : Done")
    return filtered_image

if __name__ == "__main__":
    for i in range(20, 47):
        print("============================================")
        print("===============Processing #" + str(i) + "===============")
        print("============================================")
        image = normalize_image_data_type(scale_image(mpimg.imread("../Images/" + str(i) + ".jpg"), 1000))
        gray_image = color_to_gray(image)
        kmeans_gray_image = k_means(gray_image)
        filtered_image = median_filter(kmeans_gray_image, 9)
        mpimg.imsave("../Test_Resultat/" + str(i) + "_1_gray_image.jpg", gray_image, cmap="gray", vmin=0, vmax=255)
        mpimg.imsave("../Test_Resultat/" + str(i) + "_2_kmeans_gray_image.jpg", kmeans_gray_image, cmap="gray", vmin=0, vmax=1)
        mpimg.imsave("../Test_Resultat/" + str(i) + "_3_filtered_image.jpg", filtered_image, cmap="gray", vmin=0, vmax=1)
