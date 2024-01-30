import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from utils.ElasticDeform import ElasticDeform2D_rand
import os


# %%
def shift_image(image, label, transition_x, transition_y):
    image = ndimage.shift(image, (transition_y, transition_x), order=0)
    label = ndimage.shift(label, (transition_y, transition_x), order=0)
    return image, label


# %%
def random_shift(image, label, shift, rng):
    if shift[0] == 0 and shift[1] == 0:
        return image, label
    transition_x = 0
    transition_y = 0
    if shift[0] != 0:
        transition_x = rng.randint(-shift[0], shift[0])
    if shift[1] != 0:
        transition_y = rng.randint(-shift[1], shift[1])
    image, label = shift_image(image, label, transition_x, transition_y)
    return image, label


def random_rotate(image, label, rotate, rng):
    if rotate == 0:
        return image, label
    angle = rng.randint(-rotate, rotate)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def rand_elasticdeform(image, label, elastic, sigma, rng):
    if isinstance(elastic, float) or isinstance(elastic, int):
        if elastic > 0:
            image, label = ElasticDeform2D_rand(image, label, (elastic, elastic), sigma, 0.5)
    elif isinstance(elastic, list) or isinstance(elastic, tuple):
        for ela in elastic:
            if ela > 0:
                image, label = ElasticDeform2D_rand(image, label, (ela, ela), sigma, 0.5)
    return image, label


def RandomGenerator(image, label, shift, rotate, elastic, sigma, rng):
    image, label = rand_elasticdeform(image, label, elastic, sigma, rng)
    image, label = random_rotate(image, label, rotate, rng)
    image, label = random_shift(image, label, shift, rng)
    return image, label


class aug_disk_512(torch.utils.data.Dataset):
    def __init__(self, path, num_classes, shift, rotate, elastic, seed):
        self.num_classes = num_classes
        if isinstance(shift, int) or isinstance(shift, float):
            shift = (shift, shift)
        self.shift = shift
        self.rotate = rotate
        self.elastic = elastic
        self.sigma = 0.25
        self.rng = np.random.RandomState(seed)
        self.deterministic_shift = False
        self.folder_path = path
        self.file_list = self._get_file_list()
        # with open(path,'r') as f:
        #    self.line = f.readlines()

    def __len__(self):
        return len(self.file_list)

    def set_deterministic_shift(self, flag=True):
        self.deterministic_shift = flag

    def set_sigma(self, sigma):
        self.sigma = sigma

    def _get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
        return file_list

    def __getitem__(self, idx):
        file = self._get_file_list()[idx]
        # file = self.line[idx].rstrip()
        data = torch.load(file)
        img = data['img']
        # img=data['img_corrected']
        # img = data['img_equalized']
        label = data['label'][0]
        # 0: background
        # 1~6:  bone top to bottom
        # 7~11: disk top to bottom
        if self.num_classes == 2:
            label[label > 0] = 1  # bone and disk
        elif self.num_classes == 3:
            label[(label >= 1) & (label <= 6)] = 1  # bone
            label[label >= 7] = 2  # disk
        elif self.num_classes == 12:
            pass
        else:
            raise ValueError("num_classes is not suported")
        if self.deterministic_shift == False:
            img, label = RandomGenerator(img, label, self.shift, self.rotate, self.elastic, self.sigma, self.rng)
        else:
            img, label = RandomGenerator(img, label, (0, 0), self.rotate, self.elastic, self.sigma, self.rng)
            img, label = shift_image(img, label, self.shift[0], self.shift[1])

        img = img.reshape(1, img.shape[0], img.shape[1])  # CxHxW
        if torch.is_tensor(img) == False:
            img = torch.tensor(img, dtype=torch.float32)
        if torch.is_tensor(label) == False:
            label = torch.tensor(label, dtype=torch.float32)
            #
        all_mask = torch.zeros((self.num_classes, label.shape[0], label.shape[1]), dtype=torch.float32)
        for i in range(self.num_classes):
            all_mask[i] = (label == i)

        return img, label, all_mask


# %%
if __name__ == "__main__":
    train_dataset = aug_disk_512(r'../dataset/Train', 12, [0, 0], 0, [9, 17, 33], 1)
    train_dataset.set_sigma(0.5)
    img, label, all_mask = train_dataset[0]
    img = img.numpy().reshape(512, 512)
    label = label.numpy().reshape(512, 512)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(label, cmap='gray')
    plt.show()
