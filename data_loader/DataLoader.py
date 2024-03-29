from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image

class DataLoader:
    def __init__(self, data, batch_size, shuffle = False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def shuffle_data(self):
        if self.shuffle:
            idx = torch.randperm(len(self.data))
            self.data = self.data[idx]

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
            if image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
                image_path = os.path.join(image_folder, image_file)
                image = read_image(image_path)
                if image is not None:
                    images.append(image)
        return torch.stack(images)


    image_folder = "train"
    image_array = read_images_to_array(image_folder)

    a = DataLoader(image_array, 32, True)

    d = iter(a)

    for k, batch in enumerate(d):
        if k == 3:
            break
        plt.imshow(batch[0].permute(1, 2, 0))
        plt.axis('off')  # Hide axes
        plt.show()
        print(batch.shape)
