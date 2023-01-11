import argparse
from tqdm import tqdm
from data.inference_dataset import inference_dataset
from torch.utils.data import DataLoader
import os
import torch
from model import networks
from torchvision.utils import save_image



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="./inference_input", help="Infer images in this directory.")
    parser.add_argument("--pth_root", type=str, default="./checkpoints/fully_trained", help="Use trained weights in this directory.")
    parser.add_argument("--output_root", type=str, default="./inference_output", help="Save images to this directory.")
    parser.add_argument('--verbose', action='store_true', help='Also save glass/shadow masks and de-shadowed images.')
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    args = parser.parse_args()
    args.verbose = True
    print("Call with args:")
    print(args)

    ### gpu thingy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing {device} device')

    ### dataset setup
    infer_dataset = inference_dataset(args.input_root)
    infer_dataloader = DataLoader(infer_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    ### model setup
    net_DA = networks.DomainAdapter()                                   # DA network
    net_GM = networks.ResnetGeneratorMask(input_nc=64, output_nc=2)     # Glass Mask prediction network
    net_SM = networks.ResnetGeneratorMask(input_nc=65, output_nc=2)     # Shadow Mask prediction network
    net_DS = networks.ResnetGenerator(input_nc=5, output_nc=3)          # De-Shadow network
    net_DG = networks.ResnetGenerator(input_nc=4, output_nc=3)          # De-Glass network

    net_DA.load_state_dict(torch.load(os.path.join(args.pth_root, "latest_net_G_DA.pth")))
    net_GM.load_state_dict(torch.load(os.path.join(args.pth_root, "latest_net_G_GLASS_MASK.pth")))
    net_SM.load_state_dict(torch.load(os.path.join(args.pth_root, "latest_net_G_SHADOW_MASK.pth")))
    net_DS.load_state_dict(torch.load(os.path.join(args.pth_root, "latest_net_deshadow.pth")))
    net_DG.load_state_dict(torch.load(os.path.join(args.pth_root, "latest_net_deglass.pth")))

    net_DA.to(device).eval()
    net_GM.to(device).eval()
    net_SM.to(device).eval()
    net_DS.to(device).eval()
    net_DG.to(device).eval()

    ### create dir if needed
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)


    for real_image_all, real_name in tqdm(infer_dataloader):
        batch_size = real_image_all.size(dim=0)

        ### generate masks
        real_image_all = real_image_all.to(device)
        fake_Features = net_DA(real_image_all)   
        net_GM_out = net_GM(fake_Features[0])
        fake_GM = torch.clamp((net_GM_out.argmax(1).unsqueeze(1).float()), min = 0, max = 1)
        net_SM_out = net_SM(torch.cat([fake_Features[1], fake_GM], dim=1))
        fake_SM = torch.clamp((torch.nn.Softmax2d()(net_SM_out)[:, 1:2, :, :]) * 1.25, min = 0, max = 1)

        ### generate de-shadowed and de-glassed images
        fake_DS_out = net_DS(torch.cat([real_image_all, fake_SM, fake_GM], dim = 1))
        fake_DS = fake_DS_out * (1 - fake_GM)
        fake_DG = net_DG(torch.cat([fake_DS, fake_GM], dim = 1))

            
        ### store images
        for i in range(batch_size):
            save_image(fake_DG[i], os.path.join(args.output_root, real_name[i] + ".png"), normalize=True, range=(-1, 1))
            if args.verbose:
                save_image(fake_DS[i], os.path.join(args.output_root, real_name[i] + "-DS.png"), normalize=True, range=(-1, 1))
                save_image(fake_SM[i], os.path.join(args.output_root, real_name[i] + "-SM.png"), normalize=True, range=(0, 1))
                save_image(fake_GM[i], os.path.join(args.output_root, real_name[i] + "-GM.png"), normalize=True, range=(0, 1))