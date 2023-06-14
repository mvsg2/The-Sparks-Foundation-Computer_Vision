import cv2
# import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import os
import time

dir_path = "E:/Study and Work/The Sparks Foundation (TSF)/Task-2/images/"

def BGR_to_RGB(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image

def RGB_to_BGR(image):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return bgr_image

def BGR_to_GRAY(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def RGB_to_GRAY(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(path):
    image = cv2.imread(path)
    rgb_image = BGR_to_RGB(image=image)
    return rgb_image

def modify_image(image):    
    # modified_image = cv2.resize(image, resize_params, interpolation = cv2.INTER_AREA)
    modified_image = image.reshape(image.shape[0]*image.shape[1],3)
    return modified_image

def get_colors(image, no_of_colors:int, show_chart:bool=False):
    
    modified_image = modify_image(image=image)
    kmeans_classifer = KMeans(n_clusters=no_of_colors)
    labels = kmeans_classifer.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = kmeans_classifer.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (10, 10))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        plt.show()

    return rgb_colors

images = []
print(os.listdir(dir_path))
for file in os.listdir(dir_path):
    if not file.endswith(".txt") or not file.startswith('.'):
        images.append(get_image(dir_path + file))

print(f"There are {len(images)} images in the working directory.")

for image, file in zip(images, os.listdir(dir_path)):
    img = plt.imread(dir_path+file)
    plt.imshow(img); plt.show()
    time.sleep(1)
    print(get_colors(image=image, no_of_colors=5, show_chart=True))
    time.sleep(2)