import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from data.removal_dataset import removal_dataset
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
    parser.add_argument("--synetic_root", type=str, default="./inference_output/ALIGN_RESULT_v2_our_mask", help="dataset dir")
    parser.add_argument("--pth_root", type=str, default="./checkpoints", help="checkpoint dir")
    parser.add_argument("--experiment", type=str, default="removal_train_1", help="name for this project / experiment")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=80, help="batch size")
    parser.add_argument("--random_seed", type=int, default=123, help="fixed seed for reproducing result")
    parser.add_argument("--log_iter", type=int, default=10, help="for every this iteration save losses to log")
    parser.add_argument("--display_iter", type=int, default=50, help="for every this iteration save images")
    parser.add_argument("--ckpt_iter", type=int, default=50, help="for every this iteration save model weights")
    parser.add_argument('--continue_train', action='store_true', help='continue training from previous progress')

    args = parser.parse_args()
    print("Call with args:")
    print(args)

    ### Hyperparameters
    lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    lambda_L_DS = 1
    lambda_L_DG = 1

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
    train_dataset = removal_dataset(args.synetic_root)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True)

    ### model setup
    net_DS = networks.ResnetGenerator(input_nc=5, output_nc=3)          # De-Shadow network
    net_DG = networks.ResnetGenerator(input_nc=4, output_nc=3)          # De-Glass network


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
        net_DS.load_state_dict(torch.load(os.path.join(args.pth_root, args.experiment, "latest_net_deshadow.pth")))
        net_DG.load_state_dict(torch.load(os.path.join(args.pth_root, args.experiment, "latest_net_deglass.pth")))
    else:
        #we have to initialize the weights
        net_DS.apply(networks.init_weights)
        net_DG.apply(networks.init_weights)
        starting_epoch = 0
        starting_iter = 0

    net_DS.to(device).train()
    net_DG.to(device).train()

    print("\nModels created\n")

    ### optimizers and loss criterion
    criterion = nn.L1Loss()
    optimizer_DS = torch.optim.Adam(net_DS.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_DG = torch.optim.Adam(net_DG.parameters(), lr=lr, betas=(beta1, beta2))

    ### buffer for loss log
    loss_log_buffer = []


    ### training loop
    for epoch in range(starting_epoch, args.epochs):
        iter_count = 0

        for real_image_all, real_DS, real_DG, fake_GM, fake_SM in train_dataloader:
            iter_count += 1
            if epoch == starting_epoch and iter_count <= starting_iter:
                continue


            batch_size = real_image_all.size(dim = 0)

            real_image_all = real_image_all.to(device)
            real_DS = real_DS.to(device)
            real_DG = real_DG.to(device)
            fake_GM = fake_GM.to(device)
            fake_SM = fake_SM.to(device)


            ### forward
            fake_DS_out = net_DS(torch.cat([real_image_all, fake_SM, fake_GM], dim = 1))
            fake_DS = fake_DS_out * (1 - fake_GM)
            fake_DG = net_DG(torch.cat([fake_DS, fake_GM], dim = 1))

            
            ### loss calculation
            loss_DS = criterion(fake_DS_out, real_DS) * lambda_L_DS
            loss_DG = criterion(fake_DG, real_DG) * lambda_L_DG
            loss_G = loss_DS + loss_DG



            ### back propagate and update weights
            optimizer_DS.zero_grad()
            optimizer_DG.zero_grad()
            loss_G.backward()
            optimizer_DS.step()
            optimizer_DG.step()

            
            ### save and display images
            if iter_count % args.display_iter == 0:
                save_image(real_image_all, os.path.join(args.pth_root, args.experiment, "images", "image_all.jpg"), normalize=True, range=(-1, 1))
                save_image(fake_SM, os.path.join(args.pth_root, args.experiment, "images", "SM_fake.jpg"), normalize=True, range=(0, 1))
                save_image(fake_GM, os.path.join(args.pth_root, args.experiment, "images", "GM_fake.jpg"), normalize=True, range=(0, 1))
                save_image(fake_DS, os.path.join(args.pth_root, args.experiment, "images", "DS_fake.jpg"), normalize=True, range=(-1, 1))
                save_image(real_DS, os.path.join(args.pth_root, args.experiment, "images", "DS_real.jpg"), normalize=True, range=(-1, 1))
                save_image(fake_DG, os.path.join(args.pth_root, args.experiment, "images", "DG_fake.jpg"), normalize=True, range=(-1, 1))
                save_image(real_DG, os.path.join(args.pth_root, args.experiment, "images", "DG_real.jpg"), normalize=True, range=(-1, 1))

            ### save loss to log
            if iter_count % args.log_iter == 0:     
                loss_str = f"DeShadow: {loss_DS:.4f}   DeGlass: {loss_DG:.4f}   epo: [{epoch+1}/{args.epochs}]   iter: [{iter_count}/{len(train_dataloader)}]"
                print(loss_str)
                loss_log_buffer.append(loss_str + "\n")

            ### save model weights and info
            if iter_count % args.ckpt_iter == 0:
                ### save info to json
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

                ### save model weight
                torch.save(net_DS.state_dict(), os.path.join(args.pth_root, args.experiment, "latest_net_deshadow.pth"))
                torch.save(net_DG.state_dict(), os.path.join(args.pth_root, args.experiment, "latest_net_deglass.pth"))

                with open(os.path.join(args.pth_root, args.experiment, "loss_log.txt"), 'a') as f:
                    f.writelines(loss_log_buffer)
                    loss_log_buffer.clear()

                print("Latest model has been saved")