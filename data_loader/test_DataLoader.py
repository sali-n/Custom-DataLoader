import unittest
import os
import cv2
import torch
from DataLoader import DataLoader
from torchvision.io import read_image

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.image_folder = "train"
        self.batch_size = 64

        # Loading Images.
        image_files = os.listdir(self.image_folder)
        self.images = []
        for image_file in image_files:
            if image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
                image_path = os.path.join(self.image_folder, image_file)
                image = read_image(image_path)
                if image is not None:
                    self.images.append(image)
        self.image_data = torch.stack(self.images)

    def test_batch_size(self):
        data_loader = DataLoader(self.image_data, self.batch_size, True)
        for k, batch in enumerate(data_loader):
            if k == 10:
                break
            self.assertEqual(len(batch), self.batch_size, f"Batch size does not match expected size, index = {k}")

    def test_unique_batches(self):
        data_loader = DataLoader(self.image_data, self.batch_size, True)
        data_iterator = iter(data_loader)
        prev_batch = next(data_iterator)
        for _ in range(5):
            batch = next(data_iterator)
            self.assertFalse(torch.equal(batch, prev_batch), "Duplicate batch found")
            prev_batch = batch

if __name__ == '__main__':
    unittest.main()
