import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch



class removal_dataset(object):
    def __init__(self, dataroot):
        syn_dir = dataroot
        self.Image_all = []     # stores original images
        self.noShadow = []      # stores faces with glasses but without shadows
        self.noGlass = []       # stores faces without glasses and shadows
        self.GlassMask = []     # stores glass masks
        self.ShadowMask = []    # stores shadow masks

        print(f"\nParsing Synetic images in {syn_dir}")
        for _, _, fnames in sorted(os.walk(syn_dir)):
            for fname in tqdm(fnames):
                if fname.split("-")[-1] == "all.png" :
                    self.Image_all.append(os.path.join(syn_dir, fname))
                elif fname.split("-")[-1] == "glass.png" :
                    self.noShadow.append(os.path.join(syn_dir, fname))
                elif fname.split("-")[-1] == "face.png" :
                    self.noGlass.append(os.path.join(syn_dir, fname))
                elif fname.split("-")[-1] == "seg.png" :
                    self.GlassMask.append(os.path.join(syn_dir, fname))
                elif fname.split("-")[-1] == "shseg.png" :
                    self.ShadowMask.append(os.path.join(syn_dir, fname))
        
        self.Image_all.sort()
        self.noShadow.sort()
        self.noGlass.sort()
        self.GlassMask.sort()
        self.ShadowMask.sort()


    def __getitem__(self, index):
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transform_GM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
            # Authors "randomly expand or corrode the glass mask to simulate the bad prediction in real application"
            transforms.RandomResizedCrop(size = (256,256),scale=(0.95, 1.05), ratio=(0.95, 1.05))
        ])
        transform_SM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256])
        ])

        data_Image_all = transform_image(Image.open(self.Image_all[index]).copy())
        data_noShadow = transform_image(Image.open(self.noShadow[index]).copy())
        data_noGlass = transform_image(Image.open(self.noGlass[index]).copy())
        data_GlassMask = transform_GM(Image.open(self.GlassMask[index]).copy())[0:1, :, :]      # in case its a 3-channel grayscale image (?)
        data_ShadowMask = transform_SM(Image.open(self.ShadowMask[index]).copy())[0:1, :, :]

        return data_Image_all, data_noShadow, data_noGlass, data_GlassMask, data_ShadowMask


    def __len__(self):
        return len(self.Image_all)
