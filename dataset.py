import os
import json

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class ULSDataset(Dataset):
    def __init__(self, data_dir, location_file, dataset_size=5, random_seed=2) -> None:
        self.data_dir = data_dir

        with open(location_file) as f:
            dataset_folders = f.readlines()

        dataset_folders = [os.path.join(self.data_dir, folder.strip(), 'labels') for folder in dataset_folders]

        self.label_files = []
        for dataset_folder in dataset_folders:
            self.label_files = self.label_files + [os.path.join(dataset_folder, image_file) for image_file in os.listdir(dataset_folder)]
        
        if dataset_size is not None:
            np.random.seed(seed=random_seed)
            self.label_files = np.random.choice(self.label_files, size=dataset_size)
        np.random.seed(seed=None)


    def __getitem__(self, index):
        label_file = self.label_files[index]
        image_file = label_file.replace('labels', 'images')
        annotation_file = label_file.replace('labels', 'annotations').replace('.nii.gz', '.json')

        #image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()
        with open(annotation_file) as f:
            annotation_data = json.load(f)

        image_data = nib.load(image_file).get_fdata()
        
        major_axis = np.array(annotation_data['major'])

        return image_data, label_data, major_axis

    def __len__(self):
        return len(self.label_files)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from annotation_tools import get_major_axis
    from tqdm import tqdm

    dataset = ULSDataset(data_dir='/UserData/', location_file='eval_data.txt', dataset_size=20, random_seed=11)
    for idx, (image_3d, mask_3d, major_axis) in enumerate(tqdm(dataset)):
        # print(major_axis)
        slice_idx = major_axis[0][2]
        mask_slice = mask_3d[:,:,slice_idx]
        image_slice = image_3d[:,:,slice_idx]

        plt.subplot(1, 2, 1)
        plt.imshow(image_slice, cmap='gray')

        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(image_slice, cmap='gray')
        plt.imshow(mask_slice, alpha=0.2)
        plt.plot(major_axis[:,0], major_axis[:,1], lw=1, c='red', alpha=0.4)

        plt.axis('off')
        plt.savefig(f'examples/example_{idx}.png')
        plt.close()