import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch



class DA_Dataset(object):
    def __init__(self, syn_dir, real_dir):
        self.Image_all = []
        self.Mask_shadow = []
        self.Mask_glass = []
        self.isReal = []

        print(f"\nParsing Synetic images in {syn_dir}")
        for _, _, fnames in sorted(os.walk(syn_dir)):
            for fname in tqdm(fnames):
                if fname.split("-")[-1] == "all.png" :
                    self.Image_all.append(os.path.join(syn_dir, fname))
                    self.isReal.append(0)
                elif fname.split("-")[-1] == "shseg.png" :
                    self.Mask_shadow.append(os.path.join(syn_dir, fname))
                elif fname.split("-")[-1] == "seg.png" :
                    self.Mask_glass.append(os.path.join(syn_dir, fname))
        
        self.Image_all.sort()
        self.Mask_glass.sort()
        self.Mask_shadow.sort()

        print(f"\nParsing real images in {real_dir}")       
        for root, _, fnames in sorted(os.walk(real_dir)):
            for fname in tqdm(fnames):
                self.Image_all.append(os.path.join(real_dir, fname))
                self.Mask_shadow.append("Leave it blank")
                self.Mask_glass.append("Leave it blank")
                self.isReal.append(1)


    def __getitem__(self, index):
        transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256])
        ])
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        data_Image_all = transform_image(Image.open(self.Image_all[index]).copy())
        data_isReal = self.isReal[index]
        if self.Mask_shadow[index] == "Leave it blank":
            data_Mask_shadow = torch.zeros(1, 256, 256)
        else:
            data_Mask_shadow = transform_mask(Image.open(self.Mask_shadow[index]).copy())
        
        if self.Mask_glass[index] == "Leave it blank":
            data_Mask_glass = torch.zeros(1, 256, 256)
        else:
            data_Mask_glass = transform_mask(Image.open(self.Mask_glass[index]).copy())

        return data_Image_all, data_Mask_shadow, data_Mask_glass, data_isReal


    def __len__(self):
        return len(self.Image_all)
