import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd

def scale_image(image, target_size):
    height = image.shape[0]
    width = image.shape[1]
    if height > width:
        ratio = target_size / height
    else:
        ratio = target_size / width
    new_height = int(height * ratio)
    new_width = int(width * ratio)
    new_image = np.zeros((new_height, new_width, image.shape[2]))
    for i in range(new_height):
        for j in range(new_width):
            i_pos = int(i / ratio)
            j_pos = int(j / ratio)
            new_image[i][j] = image[i_pos][j_pos]
    print("Image Scaling : Done")
    return new_image

def convert_image_data_type(image):
    height = image.shape[0]
    width = image.shape[1]
    if len(image.shape) > 2 and image.shape[2] >= 3:
        new_image = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
        if isinstance(image[0][0][0], float):
            for i in range(height):
                for j in range(width):
                    new_image[i][j][0] = image[i][j][0] * 255
                    new_image[i][j][1] = image[i][j][1] * 255
                    new_image[i][j][2] = image[i][j][2] * 255
        elif isinstance(image[0][0][0], int):
            for i in range(height):
                for j in range(width):
                    new_image[i][j][0] = image[i][j][0]
                    new_image[i][j][1] = image[i][j][1]
                    new_image[i][j][2] = image[i][j][2]
    else:
        new_image = np.zeros((height, width), dtype=np.uint8)
        if isinstance(image[0][0], np.float32):
            for i in range(height):
                for j in range(width):
                    new_image[i][j] = image[i][j] * 255
        elif isinstance(image[0][0], int):
            for i in range(height):
                for j in range(width):
                    new_image[i][j] = image[i][j]
    print("Image Data Type Conversion : Done")
    return new_image

def color_to_gray(image):
    height = image.shape[0]
    width = image.shape[1]
    gray_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            gray_image[i][j] = int((image[i][j][0] + image[i][j][1] * 2 + image[i][j][2]) / 4)
    print("Image To Gray : Done")
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
    center_image = np.zeros((height, width))
    k_means_image = np.zeros((height, width))
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
                k_means_image[i][j] = 255
    return k_means_image

gray_image = color_to_gray(convert_image_data_type(scale_image(mpimg.imread("../Images/7.jpg"), 1000)))
equalized_image = equalize_image(gray_image)
plt.plot(generate_histogram(equalized_image))
plt.show()
plt.plot(generate_cumulative_histogram(generate_histogram(equalized_image)))
plt.show()
mpimg.imsave("../Test_Equalized.jpg", equalized_image, cmap="gray", vmin=0, vmax=255)

"""
for i in range(10):
    image = redimensionner_image(mpimg.imread("../Images/"+ str(i) +".jpg"), 1000)
    image_gray = color_to_gray(image)
    image_equalized = equal(image_gray)
    image_kmeans = k_means(image_gray)
    plt.plot(histogramme_cumule(image_gray))
    plt.show()
    mpimg.imsave("../Resultat_Kmeans/"+ str(i) +"_gray.png", image_gray, cmap="gray", vmin=0, vmax=255)
    mpimg.imsave("../Resultat_Kmeans/"+ str(i) +"_equalized.png", image_equalized, cmap="gray", vmin=0, vmax=255)
    mpimg.imsave("../Resultat_Kmeans/"+ str(i) +"_kmeans.png", image_kmeans, cmap="gray", vmin=0, vmax=255)
"""