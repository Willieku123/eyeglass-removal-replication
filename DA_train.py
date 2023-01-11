import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from data.DA_dataset import DA_Dataset
from model import networks
import torch.nn as nn
from torchvision.utils import save_image
import argparse
import json



def set_all_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_path", type=str, default="./datasets/vgg_normalised.pth", help="path to vgg_normalised.pth")
    parser.add_argument("--synetic_root", type=str, default="./datasets/ALIGN_RESULT_v2", help="synetic dataset dir")
    parser.add_argument("--celebA_root", type=str, default="./datasets/celebA/img_align_celeba_glass_256x256", help="real dataset dir")
    parser.add_argument("--pth_root", type=str, default="./checkpoints", help="checkpoint dir")
    parser.add_argument("--experiment", type=str, default="DA_train_1", help="name for this project / experiment")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs", type=int, default=30, help="epoch count")
    parser.add_argument("--random_seed", type=int, default=123, help="fixed seed for reproducing result")
    parser.add_argument("--log_iter", type=int, default=10, help="for every this iteration save losses to log")
    parser.add_argument("--display_iter", type=int, default=10, help="for every this iteration save images")
    parser.add_argument("--ckpt_iter", type=int, default=10, help="for every this iteration save model weights")
    parser.add_argument('--continue_train', action='store_true', help='continue training from previous progress')

    args = parser.parse_args()
    print("Call with args:")
    print(args)


    ### Hyperparameters
    lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    lambda_L_G_DA = 0.1
    lambda_L_D_DA = 0.1
    lambda_L_GLASS_MASK = 1
    lambda_L_SHADOW_MASK = 1

    ### create directories if needed
    if not os.path.exists(args.pth_root):
        os.makedirs(args.pth_root)

    if not os.path.exists(os.path.join(args.pth_root, args.experiment)):
        os.makedirs(os.path.join(args.pth_root, args.experiment))

    if not os.path.exists(os.path.join(args.pth_root, args.experiment, "images")):
        os.makedirs(os.path.join(args.pth_root, args.experiment, "images"))
    
    if  os.path.exists(os.path.join(args.pth_root, args.experiment, "loss_log.txt")) and not args.continue_train:
        os.remove(os.path.join(args.pth_root, args.experiment, "loss_log.txt"))

    ### gpu thingy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing {device} device')

    ### fix seeds for reproductability
    set_all_seed(args.random_seed)
    
    ### dataset setup
    train_dataset = DA_Dataset(args.synetic_root, args.celebA_root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True)

    ### model setup
    net_DA = networks.DomainAdapter(vgg_path=args.vgg_root)             # DA network
    net_GM = networks.ResnetGeneratorMask(input_nc=64, output_nc=2)     # Glass Mask prediction network
    net_SM = networks.ResnetGeneratorMask(input_nc=65, output_nc=2)     # Shadow Mask prediction network
    net_D = networks.Discriminator(input_nc=128)                        # Discriminator

    ### weights initialization
    if args.continue_train and os.path.exists(os.path.join(args.pth_root, args.experiment, "latest_info.json")):
        #read info in json
        with open(os.path.join(args.pth_root, args.experiment, "latest_info.json"))as f:
            ckpt_info = json.load(f)
            args.batch_size = ckpt_info["batch_size"]
            args.epochs = ckpt_info["epochs"]
            args.random_seed = ckpt_info["random_seed"]
            starting_epoch = ckpt_info["latest_epoch"]
            starting_iter = ckpt_info["latest_iter"]
        # load existing training progress
        net_DA.load_state_dict(torch.load(os.path.join(args.pth_root, args.experiment, "latest_net_G_DA.pth")))
        net_GM.load_state_dict(torch.load(os.path.join(args.pth_root, args.experiment, "latest_net_G_GLASS_MASK.pth")))
        net_SM.load_state_dict(torch.load(os.path.join(args.pth_root, args.experiment, "latest_net_G_SHADOW_MASK.pth")))
        net_D.load_state_dict(torch.load(os.path.join(args.pth_root, args.experiment, "latest_net_D.pth")))
    else:
        #we have to initialize the weights
        net_DA.apply(networks.init_weights)
        net_GM.apply(networks.init_weights)
        net_SM.apply(networks.init_weights)
        net_D.apply(networks.init_weights)
        starting_epoch = 0
        starting_iter = 0

    net_DA.to(device).train()
    net_GM.to(device).train()
    net_SM.to(device).train()
    net_D.to(device).train()

    print("\nModels created\n")


    ### optimizers and loss criterion
    criterionGAN = nn.MSELoss()
    criterionMASK = nn.BCELoss()
    optimizer_DA = torch.optim.Adam(net_DA.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_SM = torch.optim.Adam(net_SM.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_GM = torch.optim.Adam(net_GM.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr, betas=(beta1, beta2))


    ### buffer for loss log
    loss_log_buffer = []


    ### training loop
    for epoch in range(starting_epoch, args.epochs):
        iter_count = 0

        for real_image_all, real_SM, real_GM, real_isReal in train_dataloader:
            iter_count += 1
            if epoch == starting_epoch and iter_count <= starting_iter:
                continue

            batch_size = real_isReal.size(dim = 0)
            real_isReal = real_isReal.to(torch.float32)

            real_image_all = real_image_all.to(device)
            real_SM = real_SM.to(device)
            real_GM = real_GM.to(device)
            real_isReal = real_isReal.to(device)

            enable_filter_256 = real_isReal.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256).to(device)
            enable_filter_30 = real_isReal.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 30).to(device)
            ones_30 = torch.ones(batch_size, 1, 30, 30).to(device)
            enable_filter_256.detach()
            enable_filter_30.detach()


            ### forward
            # Image_all -> [DA network] -> features (sfm & gfm)
            fake_Features = net_DA(real_image_all)

            # sfm -> [Shadow Mask network] -> glass mask        
            net_GM_out = net_GM(fake_Features[0])  # G(A)
            fake_GM = (torch.exp(net_GM_out[:,1,:,:]) / (torch.exp(net_GM_out[:,0,:,:]) + torch.exp(net_GM_out[:,1,:,:]))).unsqueeze(1).float()

            # sfm + glass mask -> [Shadow Mask network] -> shadow mask
            net_SM_out = net_SM(torch.cat([fake_Features[1], fake_GM], dim=1))
            fake_SM = (torch.nn.Softmax2d()(net_SM_out)[:, 1:2, :, :]) * 1.25


            ### loss calculation for D
            # enable backprop for D
            for param in net_D.parameters():
                param.requires_grad = True
                
            fake_Features_cat = torch.cat(fake_Features, dim = 1)

            # concatenate gfm (64 ch) and sfm (64 ch) into 128 channels. dim 0 is batch and so dim 1 is channel.
            fake_isReal = net_D(fake_Features_cat.detach())

            loss_D = criterionGAN(fake_isReal, enable_filter_30) *  lambda_L_D_DA
            
            
            ### back propagate and update weights for D
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            
            ### loss calculation for G
            for param in net_D.parameters():  # D requires no gradients when optimizing G
                param.requires_grad = False

            fake_GM_forloss = torch.where(enable_filter_256 > 0.9, torch.tensor(0, dtype=fake_GM.dtype).to(device), fake_GM)
            fake_SM_forloss = torch.where(enable_filter_256 > 0.9, torch.tensor(0, dtype=fake_SM.dtype).to(device), fake_SM)
            real_GM_forloss = torch.where(enable_filter_256 > 0.9, torch.tensor(0, dtype=real_GM.dtype).to(device), real_GM)
            real_SM_forloss = torch.where(enable_filter_256 > 0.9, torch.tensor(0, dtype=real_SM.dtype).to(device), real_SM)

            # DA network should fake the discriminator
            num_fake = batch_size - sum(real_isReal)
            fake_Features_cat = torch.cat(fake_Features, dim = 1) # concatenate gfm and sfm, dim 0 is batch, dim 1 is channel.
            fake_isReal = net_D(fake_Features_cat)
            fake_isReal_forloss = torch.where(enable_filter_30 > 0.9, torch.tensor(1, dtype=fake_isReal.dtype).to(device), fake_isReal)

            loss_DA = criterionGAN(fake_isReal_forloss, ones_30) *  lambda_L_G_DA

            loss_vDA = loss_DA * batch_size / num_fake if num_fake != 0 else 0    #for visualization

            # Glass Mask prediction and shadow mask prediction
            fake_GM_clamp = torch.clamp(fake_GM_forloss, min=1e-14, max=1-1e-14)
            fake_SM_clamp = torch.clamp(fake_SM_forloss, min=1e-14, max=1-1e-14)
            real_GM_clamp = torch.clamp(real_GM_forloss, min=1e-14, max=1-1e-14)
            real_SM_clamp = torch.clamp(real_SM_forloss, min=1e-14, max=1-1e-14)
            real_GM_clamp.detach()
            real_SM_clamp.detach()

            loss_GM = criterionMASK(fake_GM_clamp, real_GM_clamp) * lambda_L_GLASS_MASK
            loss_SM = criterionMASK(fake_SM_clamp, real_SM_clamp) * lambda_L_SHADOW_MASK

            loss_vGM = loss_GM * batch_size / num_fake if num_fake != 0 else 0    #for visualization
            loss_vSM = loss_SM * batch_size / num_fake if num_fake != 0 else 0    #for visualization
            
            loss_G = loss_DA + loss_GM + loss_SM


            ### back propagate and update weights for G
            optimizer_DA.zero_grad()
            optimizer_GM.zero_grad()
            optimizer_SM.zero_grad()
            loss_G.backward()
            optimizer_DA.step()
            optimizer_GM.step()
            optimizer_SM.step()

            
            ### save and display images
            if iter_count % args.display_iter == 0: 
                save_image(real_image_all, os.path.join(args.pth_root, args.experiment, "images", "image_all.jpg"), normalize=True)
                save_image(fake_GM, os.path.join(args.pth_root, args.experiment, "images", "GM_fake.jpg"), normalize=True)
                save_image(real_GM, os.path.join(args.pth_root, args.experiment, "images", "GM_real.jpg"), normalize=True)
                save_image(fake_SM, os.path.join(args.pth_root, args.experiment, "images", "SM_fake.jpg"), normalize=True)
                save_image(real_SM, os.path.join(args.pth_root, args.experiment, "images", "SM_real.jpg"), normalize=True)
                save_image(enable_filter_30, os.path.join(args.pth_root, args.experiment, "images", "isReal_real.jpg"), normalize=True)
                save_image(fake_isReal, os.path.join(args.pth_root, args.experiment, "images", "isReal_fake.jpg"), normalize=True)

            ### save loss to log
            if iter_count % args.log_iter == 0:
                loss_str = f"DA: {loss_vDA:.4f}   GM: {loss_vGM:.4f}   SM: {loss_vSM:.4f}   D: {loss_D:.4f}   epo: [{epoch+1}/{args.epochs}]   iter: [{iter_count}/{len(train_dataloader)}]"
                print(loss_str)
                loss_log_buffer.append(loss_str + "\n")

            ### save model weights and info
            if iter_count % args.ckpt_iter == 0:
                # save info to json
                ckpt_info = {
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "random_seed": args.random_seed,
                    "latest_epoch": epoch,
                    "latest_iter":iter_count
                }
                json_object = json.dumps(ckpt_info, indent=4)
                with open(os.path.join(args.pth_root, args.experiment, "latest_info.json"), "w") as outfile:
                    outfile.write(json_object)

                # save model weight
                torch.save(net_DA.state_dict(), os.path.join(args.pth_root, args.experiment, "latest_net_G_DA.pth"))
                torch.save(net_GM.state_dict(), os.path.join(args.pth_root, args.experiment, "latest_net_G_GLASS_MASK.pth"))
                torch.save(net_SM.state_dict(), os.path.join(args.pth_root, args.experiment, "latest_net_G_SHADOW_MASK.pth"))
                torch.save(net_D.state_dict(), os.path.join(args.pth_root, args.experiment, "latest_net_D.pth"))

                with open(os.path.join(args.pth_root, args.experiment, "loss_log.txt"), 'a') as f:
                    f.writelines(loss_log_buffer)
                    loss_log_buffer.clear()

                print("Latest model has been saved")
