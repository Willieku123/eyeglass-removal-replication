import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm



class inference_dataset(object):
    def __init__(self, dataroot):
        self.Image_all = []     # stores path to input image
        self.name = []          # stores image name


        print(f"\nParsing images in {dataroot}")
        for _, _, fnames in sorted(os.walk(dataroot)):
            for fname in tqdm(fnames):
                self.Image_all.append(os.path.join(dataroot, fname))
                self.name.append(fname[:-4])


    def __getitem__(self, index):
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        data_Image_all = transform_image(Image.open(self.Image_all[index]).copy())
        data_name = self.name[index]

        return data_Image_all, data_name


    def __len__(self):
        return len(self.Image_all)