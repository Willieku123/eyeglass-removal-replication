import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm



class mask_inference_dataset(object):
    def __init__(self, syn_dir):
        self.Image_all = []     # stores original images
        self.noShadow = []      # stores faces with glasses but without shadows
        self.noGlass = []       # stores faces without glasses and shadows
        self.name = []          # stores image name


        print(f"\nParsing Synetic images in {syn_dir}")
        for _, _, fnames in sorted(os.walk(syn_dir)):
            for fname in tqdm(fnames):
                if fname.split("-")[-1] == "all.png" :
                    self.Image_all.append(os.path.join(syn_dir, fname))
                    self.name.append(fname[:-7])
                elif fname.split("-")[-1] == "glass.png" :
                    self.noShadow.append(os.path.join(syn_dir, fname))
                elif fname.split("-")[-1] == "face.png" :
                    self.noGlass.append(os.path.join(syn_dir, fname))
        
        self.Image_all.sort()
        self.noShadow.sort()
        self.noGlass.sort()


    def __getitem__(self, index):
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
        ])

        data_Image_all = transform_image(Image.open(self.Image_all[index]).copy())
        data_noShadow = transform_image(Image.open(self.noShadow[index]).copy())
        data_noGlass = transform_image(Image.open(self.noGlass[index]).copy())
        data_name = self.name[index]

        return data_Image_all, data_noShadow, data_noGlass, data_name


    def __len__(self):
        return len(self.Image_all)