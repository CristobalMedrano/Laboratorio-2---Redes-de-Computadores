#-*- coding: utf-8 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 19/06/2020
#LABORATORY 2: CONVOLUCIÓN 2D

# IMPORTS
import imageio
import os.path
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
# GLOBAL VARIABLES
# CLASSES
class Image:

    def __init__(self, filename, matrix):
        self.__filename = filename
        self.__matrix = matrix
        self.__convolution = []
        self.__height = 0
        self.__width = 0
        self.__n_channels = 0

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def set_filename(self, filename):
        self.__filename = filename

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def set_matrix(self, matrix):
        self.__matrix = matrix

    @property
    def height(self):
        return self.__matrix.shape[0]

    @height.setter
    def set_heigth(self, height):
        self.__height = height

    @property
    def width(self):
        return self.__matrix.shape[1]

    @width.setter
    def set_width(self, width):
        self.__width = width

    @property
    def n_channels(self):
        if len(self.__matrix.shape) > 2:
            return self.__matrix.shape[2]
        else:
            return 1

    @n_channels.setter
    def set_n_channels(self, n_channels):
        self.__n_channels = n_channels


class Kernel:

    def __init__(self, matrix):
        self.__matrix = matrix
        self.__height = 0
        self.__width = 0

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def set_matrix(self, matrix):
        self.__matrix = matrix

    @property
    def height(self):
        return self.__matrix.shape[0]

    @height.setter
    def set_heigth(self, height):
        self.__height = height

    @property
    def width(self):
        return self.__matrix.shape[1]

    @width.setter
    def set_width(self, width):
        self.__width = width

# FUNCTIONS


def is_valid_image_file(filename):
    """ Check if it's a valid image file.
    
    Returns True if it's valid, False if it's not.
    
    Parameters:
    ----------
    filename : string
        Input image filename. 

    Returns:
    -------
    Status: boolean
        True if it's valid, False if it's not.
   """
    if os.path.exists(filename):
        return True
    else:
        return False


def read_image(filename):
    return plt.imread(filename)

def save_image(filename, matrix):
    if (matrix.shape[2] == 1):
        plt.imsave(f"salida_{filename}", matrix[:,:,0], cmap="gray")
    else:
        plt.imsave(f"salida_{filename}", matrix)


def create_3d_matrix(number_rows, number_cols, number_channels):
    if number_rows > 0 and number_cols > 0:
        return np.zeros((number_rows, number_cols, number_channels))


def insert_value_matrix(matrix, value, pos_x, pos_y):
    if matrix.size > 0 and pos_x >= 0 and pos_y >= 0:
        matrix[pos_x][pos_y] = value
    return matrix

def convert_rgb_to_gray_scale(rgb_matrix):
    return np.dot(rgb_matrix[..., :3], [0.2989, 0.5870, 0.1140])

def get_matrix_height(matrix):
    return matrix.shape[0]

def get_matrix_width(matrix):
    return matrix.shape[1]

def get_border_size(kernel):
    return int((kernel.height - 1) / 2)

def add_magic_border_to_image(image, kernel):
    if image.n_channels > 1:    
        temp_image_matrix = create_3d_matrix(image.height + kernel.height - 1,
                                            image.width + kernel.width - 1,
                                            image.n_channels)

        height = get_matrix_height(temp_image_matrix)
        width = get_matrix_width(temp_image_matrix)
        v_border_size = get_border_size(kernel)
        h_border_size = get_border_size(kernel)

        for channel in range(0, image.n_channels):
            for i in range(v_border_size, height - v_border_size):
                for j in range(h_border_size, width - h_border_size):
                    temp_image_matrix[i][j][channel] = image.matrix[i - v_border_size][j - h_border_size][channel]
    else:
        temp_image_matrix = create_3d_matrix(image.height + kernel.height - 1,
                                             image.width + kernel.width - 1,
                                             image.n_channels)

        height = get_matrix_height(temp_image_matrix)
        width = get_matrix_width(temp_image_matrix)
        v_border_size = get_border_size(kernel)
        h_border_size = get_border_size(kernel)

        for channel in range(0, image.n_channels):
            for i in range(v_border_size, height - v_border_size):
                for j in range(h_border_size, width - h_border_size):
                    temp_image_matrix[i][j] = image.matrix[i -
                                                                    v_border_size][j - h_border_size]
    
    return temp_image_matrix

def convolution(image, kernel):
    temp_image_matrix = add_magic_border_to_image(image, kernel)
    height = get_matrix_height(temp_image_matrix)
    width = get_matrix_width(temp_image_matrix)
    border_size = get_border_size(kernel)

    #guarda jiji
    
    if image.n_channels > 1:
        convolution_image = create_3d_matrix(
            image.height, image.width, image.n_channels)
        for channel in range(0, image.n_channels):
            for i in range(border_size, height - border_size):
                for j in range(border_size, width - border_size):
                    pixel = 0
                    move_down = - border_size
                    for k in range(0, kernel.height):
                        move_right = - border_size
                        for l in range(0, kernel.width):
                            pixel = pixel + temp_image_matrix[i + move_down, j + move_right,channel] * kernel.matrix[k,l]
                            move_right = move_right + 1
                        move_down = move_down + 1
                    convolution_image[i - border_size][j - border_size][channel] = pixel
                    #guarda jiji
    else:
        convolution_image = create_3d_matrix(
            image.height, image.width, image.n_channels)

        for i in range(border_size, height - border_size):
            for j in range(border_size, width - border_size):
                pixel = 0
                move_down = - border_size
                for k in range(0, kernel.height):
                    move_right = - border_size
                    for l in range(0, kernel.width):
                        pixel = pixel + \
                            temp_image_matrix[i + move_down, j +
                                                move_right] * kernel.matrix[k, l]
                        move_right = move_right + 1
                    move_down = move_down + 1
                convolution_image[i - border_size][j -
                                                    border_size] = pixel
                #guarda jiji
    return convolution_image
    #border_size = get_border_size(kernel)
    #print(temp_image_matrix[
    #    border_size: - border_size,
    #    border_size: - border_size,
    #    0])
    #print(image.matrix[:,:,0])

#229 404
#226 401
            

    

def is_number_odd(number):
    if (number % 2) != 0:
        return True
    else:
        return False

def is_valid_kernel(kernel):
    #Check if a height is odd and the kernel is square
    if is_number_odd(kernel.height) and (kernel.height == kernel.width):
        return True
    else:
        return False

def plot_image(image):
    plt.title(image.filename)
    if (image.n_channels == 1):
        plt.imshow(image.matrix, cmap="gray")
    else:
        plt.imshow(image.matrix)
    plt.xlabel("Ancho (px)")
    plt.ylabel("Alto (px)")
    plt.show()


def normalize_matrix(matrix):
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    return (matrix - min_value) / (max_value - min_value)

# MAIN
def main():
    """ Main function of program """
    print("Laboratorio 2 - Convolución 2D")
    jpg = "lena.jpg"
    png = "images.png"
    bmp = "lena512.bmp"
    dog = "dog.jpg"

    filename = bmp

    laplacian_filter = Kernel(np.matrix([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]]))


    gaussian_filter = Kernel(np.matrix([[1, 4, 6, 4, 1],
                                        [4, 16, 24, 16, 4],
                                        [6, 24, 36, 24, 6],
                                        [4, 16, 24, 16, 4],
                                        [1, 4, 6, 4, 1]])*1/256)

    edge_detection_filter = Kernel(np.matrix([[1, 2, 0, -2, -1],
                                            [1, 2, 0, -2, -1],
                                            [1, 2, 0, -2, -1],
                                            [1, 2, 0, -2, -1],
                                            [1, 2, 0, -2, -1]]))
    #image_convolution = convolution(image, kernel)
    if is_valid_image_file(filename) and is_valid_kernel(gaussian_filter):
        original_image = read_image(filename)
        filtered_image = apply_filter(image, gaussian_filter)
        normalized_matrix = normalize_matrix(filtered_image)
        #plot_image(new_image)
        #print(original_image.matrix)
        #gray_scale_to_rgb(convert_rgb_to_gray_scale(original_image.matrix))
        #save_image("2"+filename, gray_scale_to_rgb(convert_rgb_to_gray_scale(original_image.matrix)).astype(np.uint8))
        save_image(filename, normalized_matrix)


# MAIN
main()

# REFERENCES
#https://pypi.org/project/imageio/
#https://qastack.mx/stats/70801/how-to-normalize-data-to-0-1-range
