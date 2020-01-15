#  //// Data Representation, Reduction and Analysis Project         //////
#  //// Image Super Resolution: Using ESPCN                         //////
#  // Members: Andres Sánchez, Deniz Alp Atun, Ismail Göktuğ Serin  //////
#  ///////////////////////////////////////////////////////////////////////

import os.path
import argparse
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from scripts.nnet_model import ESPCN
from math import log10

import scripts.file_process as fp
from scripts.dataset_maker import set_maker
from scripts.trainer import training


def dir_in_dir(path):
    print("Does the zip include images on root or inside folder?")
    check = input("Type 1 for yes, 0 for no: ")
    if check:
        ext_path = input("Type the path until images are seen:\n" )
        return path, ext_path
    else:
        return path, None


def train_dir_input(root_in):
    while True:
        temp = input("Type in the relative path of training data set: ")
        test = os.path.join(root_in, temp)
        print("Path: " + str(test))
        if os.path.isfile(test):
            train_path = test
            break
        else:
            print("Path doesn't end with a file!")
    extr_path, folder_path = dir_in_dir(train_path)
    return extr_path, folder_path


def valid_dir_input(root_in):
    while True:
        temp = input("Type in the relative path of validation data set: ")
        test = os.path.join(root_in, temp)
        print("Path: " + str(test))
        if os.path.isfile(test):
            valid_path = test
            break
        else:
            print("Path doesn't end with a file!")
    extr_path, folder_path = dir_in_dir(valid_path)
    return extr_path, folder_path


def main():

    print("\n ██████ Training and Validation Data Preparation ██████")
    print("Important! Place everything in the same folder, since this code uses relative paths!")
    print("Path input examples: dataset.zip or folder\\dataset.zip")
    print("-------------------------------------------------------\n")
    root_dir = os.path.realpath('.')
    train_dirs = train_dir_input(root_dir)
    valid_dirs = valid_dir_input(root_dir)

    train_dir = fp.prep_files(root_dir, train_dirs[0], train_dirs[1], "train")
    valid_dir = fp.prep_files(root_dir, valid_dirs[0], valid_dirs[1], "valid")

    print("\n ██████ Loading Data into Dataset ██████")

    train_set = set_maker(train_dir, 244, args.upscale)
    valid_set = set_maker(valid_dir, 244, args.upscale)
    training_data = DataLoader(dataset=train_set, batch_size=args.trainBatchSize, shuffle=True,
                               num_workers=args.nWorkers)
    validation_data = DataLoader(dataset=valid_set, batch_size=args.validBatchSize, shuffle=True,
                               num_workers=args.nWorkers)

    #training(args.cuda, args.seed, args.upscale, args.lr,args.nEpochs,
    #         training_data, validation_data)

    if args.cuda and not torch.cuda.is_available():
        print("No CUDA supporting GPU found, using CPU")
        cuda_in = False
    else:
        cuda_in = True

    device = torch.device("cuda" if cuda_in else "cpu")

    torch.manual_seed(args.seed)
    model = ESPCN(args.upscale, 1).to(device)  # 1 is for number of channels
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    f1 = open("PSNR_value_list.log", 'w')
    for epoch in range(args.nEpochs):
        epoch_loss = 0
        iteration = 0
        f = open("epoch_%i.log" % epoch, 'w')
        for data in training_data:
            input, label = data

            input = input.to(device)
            label = label.to(device)

            loss = criterion(model(input), label)

            optimiser.zero_grad()

            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()
            f.write("Iteration [%i/%i]: Loss: %0.4f \n" % (iteration+1, len(training_data), loss.item()))
            iteration += 1
        f.write("-----------------------------------\n")
        f.write("Average Loss: %0.4f \n" % (epoch_loss / len(training_data)))
        f.close()

        avg_psnr = 0
        for data in validation_data:
            input, label = data

            input = input.to(device)
            label = label.to(device)

            with torch.no_grad():
                prediction = model(input)
                mse = criterion(prediction, label)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
        f1.write("Average PSNR of Epoch [%i]: %0.4f dB \n" % (epoch, (avg_psnr / len(training_data))))

        model_name = "epoch_%i_model.pth" % epoch+1
        torch.save(model, model_name)
        print("Epoch (%i/%i) is done! See root dir for logs and models" % (epoch+1, args.nEpochs))
    f1.close()

    # |-------------------------------------------------------------------------------------------------| #
    # Until this part, it is very similar to PyTorch-SuperResolution Example on Github
    # Link: https://github.com/pytorch/examples/tree/master/super_resolution
    # Changes are the way to acquire data files, their relative directory and how they are
    # passed to the code. Unlike other solutions, data is dynamically passed (by asking for input)
    # Rest of the code shows similarity since deviating from the example either resulted in
    # poor code readability, making an unnecessary wall of text or using additional libraries
    # such as h5py, numpy and etc.
    # |-------------------------------------------------------------------------------------------------| #
    # Next part is modelling a Network with upscale as input, training and validating by using the model,
    # PyTorch ReLU Network (torch.nn), PyTorch Optimiser (torch.optim) and so on...
    # |-------------------------------------------------------------------------------------------------| #



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DRRA Project")
    parser.add_argument('--upscale', type=int, required=True, help="upscale factor")
    parser.add_argument('--trainBatchSize', type=int, required=True, help="training batch size")
    parser.add_argument('--validBatchSize', type=int, required=True, help="validation batch size")
    parser.add_argument('--nEpochs', type=int, required=True, help="number of epochs")
    parser.add_argument('--nWorkers', type=int, default=8, help="number of workers")
    parser.add_argument('--lr', type=float, required=True, help="learning rate")
    parser.add_argument('--cuda', action='store_true', help="enable cuda?")
    parser.add_argument('--seed', type=int, default=42, help="random seed to use. Default=42")
    args = parser.parse_args()

    print("Input args:" + str(args))

    main()




