import argparse
from tqdm import tqdm
from torchvision import transforms
from data.mask_inference_dataset import mask_inference_dataset
from torch.utils.data import DataLoader
import os
import torch
from model import networks
from torchvision.utils import save_image



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--synetic_root", type=str, default="./datasets/ALIGN_RESULT_v2", help="synetic dataset dir")
    parser.add_argument("--pth_root", type=str, default="./checkpoints/DA_train_1", help="Use trained weights in this directory.")
    parser.add_argument("--save_root", type=str, default="./inference_output/ALIGN_RESULT_v2_our_mask", help="save to this dir")

    args = parser.parse_args()
    print("Call with args:")
    print(args)


    ### gpu thingy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing {device} device')

    ### dataset setup
    train_dataset = mask_inference_dataset(args.synetic_root)
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    ### model setup
    net_DA = networks.DomainAdapter()                                   # DA network
    net_GM = networks.ResnetGeneratorMask(input_nc=64, output_nc=2)     # Glass Mask prediction network
    net_SM = networks.ResnetGeneratorMask(input_nc=65, output_nc=2)     # Shadow Mask prediction network

    net_DA.load_state_dict(torch.load(os.path.join(args.pth_root, "latest_net_G_DA.pth")))
    net_GM.load_state_dict(torch.load(os.path.join(args.pth_root, "latest_net_G_GLASS_MASK.pth")))
    net_SM.load_state_dict(torch.load(os.path.join(args.pth_root, "latest_net_G_SHADOW_MASK.pth")))

    net_DA.to(device).eval()
    net_GM.to(device).eval()
    net_SM.to(device).eval()

    
    syn_dir = os.path.join(args.dataset_root, "ALIGN_RESULT_v2")
    images = []

    transform_image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transform_toPIL = transforms.ToPILImage()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)


    for real_image_all, real_DS, real_DG, real_name in tqdm(train_dataloader):

        ### generate masks
        real_image_all = transform_image(real_image_all).to(device)
        fake_Features = net_DA(real_image_all)
        net_GM_out = net_GM(fake_Features[0])
        fake_GM = torch.clamp((net_GM_out.argmax(1).unsqueeze(1).float()), min = 0, max = 1)
        net_SM_out = net_SM(torch.cat([fake_Features[1], fake_GM], dim=1))
        fake_SM = torch.clamp((torch.nn.Softmax2d()(net_SM_out)[:, 1:2, :, :]) * 1.25, min = 0, max = 1)

            
        ### store images
        save_image(real_image_all.squeeze(0), os.path.join(args.save_root, real_name[0] + "all.png"), normalize=True, range=(-1, 1))
        save_image(transform_image(real_DG).squeeze(0), os.path.join(args.save_root, real_name[0] + "face.png"), normalize=True, range=(-1, 1))
        save_image(transform_image(real_DS).squeeze(0), os.path.join(args.save_root, real_name[0] + "glass.png"), normalize=True, range=(-1, 1))
        save_image(fake_SM.squeeze(0), os.path.join(args.save_root, real_name[0] + "shseg.png"), normalize=True, range=(0, 1))
        save_image(fake_GM.squeeze(0), os.path.join(args.save_root, real_name[0] + "seg.png"), normalize=True, range=(0, 1))
            
