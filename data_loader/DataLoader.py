import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, data, batch_size, shuffle = False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def shuffle_data(self):
        if self.shuffle:
            np.random.shuffle(self.data)

    def new_batch(self):
        pointer = 0
        while True:
            if pointer == 0:
                self.shuffle_data()
            batch_data = self.data[pointer:pointer + self.batch_size]
            pointer = (pointer + self.batch_size) % len(self.data)
            if len(batch_data) == self.batch_size:
                yield batch_data

    def __iter__(self):
        return self.new_batch()


if __name__ == '__main__':
    def read_images_to_array(image_folder):
        images = []
        image_files = os.listdir(image_folder)
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
        return np.array(images)

    image_folder = "train"
    image_array = read_images_to_array(image_folder)

    a = DataLoader(image_array, 32, True)

    d = iter(a)

    for k, batch in enumerate(d):
        if k == 3:
            break
        plt.imshow(batch[0])
        plt.axis('off')  # Hide axes
        plt.show()
