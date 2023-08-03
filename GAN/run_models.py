import os
import numpy as np
import random 
import pandas as pd
from statistics import mean
import torch
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import gc
from torch.autograd import Variable
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import math
from sklearn.model_selection import train_test_split
from utils import *
from models import *

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--cuda", type=str, default='cuda:0', required=False, help='cuda device to use')
parser.add_argument("-d", "--data", type=str, required=True, help='path to the ECGs, must be of shape (N,500,12) and normalized between 0,1')
parser.add_argument("-r", "--r_mask", type=str, required=True, help='path to the r masks npy array, should be a binary array if size (N,500,12) where R-wave regions have a value of 1 and others 0')
parser.add_argument("-n", "--nr_mask", type=str, required=True, help='path to the non-r masks npy array, should be a binary array if size (N,500,12) where non-R-wave regions have a value of 1 and others 0')
parser.add_argument("-s", "--sex_labels", type=str, required=True, help='path to the biological sex vectors')

parser.add_argument("-lr", "--learning_rate", type=float, required=False, default=1e-7, help='learning rate')
parser.add_argument("-b",  "--bath_size",      type=int,   required=False, default=64,    help='batch size')
parser.add_argument("-b1", "--beta_1", type=float , required=False, default=0.5, help='beta 1 of the adam optimiser')
parser.add_argument("-b2", "--beta_2", type=float , required=False, default=0.990, help='beta 2 of the adam optimiser')
parser.add_argument("-l", "--loss", type=str , required=False, default='MSE_R', help='reconstruction loss to use for the generator', choices=['MSE_R', 'MSE'])
parser.add_argument("-m", "--mulitplicator", type=int , required=False, default=100, help='multiplicator for the non R-wave mask')

parser.add_argument("-M", "--mode", type=str , required=False, default='PrivECG', choices=['PrivECG', 'GenGan'])

parser.add_argument("-e", "--save_every", type=int , required=False, default=3, help='save state')
parser.add_argument("-k", "--number_epochs", type=int , required=False, default=500, help='num_epochs')
parser.add_argument("-u", "--utility_budget", type=float , required=False, default=0.0001, help='num_epochs')

parser.add_argument("-o", "--out_path", type=str , required=True, help='output path')
parser.add_argument("-i", "--experiment_id", type=str, default='exp_1', required=False, help='experiment id')

args=parser.parse_args()

#load parameters paremeters
device = args.cuda
train_X = np.load(args.data)
R_mask = np.load(args.r_mask)
non_R_mask = np.load(args.nr_mask)

batch_size_ = args.bath_size
learning_rate_discriminator=args.learning_rate
opt_alpha=args.beta_1
opt_beta=args.beta_2
loss = args.loss
smooth_signal = False
multiplicator=args.mulitplicator

LongTensor = torch.cuda.LongTensor

if args.mode == 'GenGan':
    generator = GeneratorUNetGenGan().to(device)

else:
    generator = GeneratorUNet().to(device)

disciminator = Discriminator().to(device)

y_ = np.load(args.sex_labels)
y_  = np.array([0 if i == 'MALE' else 1 for i in y_])

d_train = DatasetTrain(train_X,y_,R_mask,non_R_mask)

distortion_loss = nn.MSELoss()
adversarial_loss = nn.BCEWithLogitsLoss()
adversarial_loss_rf = nn.BCEWithLogitsLoss()

train_loader = DataLoader(dataset=d_train, batch_size=int(batch_size_), shuffle=True)

optimiser_gen = torch.optim.Adam(generator.parameters(), lr=args.learning_rate ,betas = (args.beta_1, args.beta_2)) 
optimiser_discr = torch.optim.Adam(disciminator.parameters(), lr=args.learning_rate, betas = (args.beta_1, args.beta_2)) 
num_epochs = args.number_epochs

list_fake_acc = list()
list_true_acc = list()
list_generator_loss = list()
list_distortion_loss = list()
list_gender_loss = list()
list_genderless_loss = list()
list_true_data_loss = list()
list_dicriminator_loss = list()

out_path = os.path.join(args.out_path,args.experiment_id)
os.makedirs(out_path, exist_ok=True)

for epochs in range(0,num_epochs):
    print('Epoch: {}'.format(epochs))
    G_distortion_loss_accum = 0
    G_adversary_loss_accum = 0
    D_real_loss_accum = 0
    D_fake_loss_accum = 0

    generator.train()
    disciminator.train()
    
    acc_true_acc = list()
    acc_fake_acc = list()
    acc_generator_loss = list()
    acc_distortion_loss = list()
    acc_gender_loss = list()
    acc_genderless_loss = list()
    acc_true_data_loss = list()
    acc_dicriminator_loss = list()

    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor

    for i, (x, gender, r_masked, non_r_masked) in enumerate(tqdm(train_loader)):
        gender = gender.type(torch.float32)
        gender =  gender.to(device=device)
        x = x.to(device=device, dtype=torch.float32)

        non_r_masked =  non_r_masked.to(device,dtype=torch.float32)
        r_masked =  r_masked.to(device,dtype=torch.float32)

        optimiser_gen.zero_grad()

        #generate the random gender vector
        gen_secret = Variable(LongTensor(np.random.choice([1.0], x.shape[0]))).to(device)
        gen_secret = gen_secret * np.random.normal(0.5, math.sqrt(0.01))

        gen_secret = gen_secret.type(torch.float32)
        gen_secret =  gen_secret.to(device=device)

        # ----------- Generator --------------

        #get outputs of the networks
        gen_results = generator(x)

        pred_secret = disciminator(gen_results)

        # 'MSE','MSE_R'
        if args.loss == 'MSE':
            generator_distortion_loss = distortion_loss(gen_results, x).to(device)

        elif args.loss == 'MSE_R':
            generator_distortion_loss = distortion_loss(gen_results * r_masked, x * r_masked) + distortion_loss(gen_results * non_r_masked, x * non_r_masked)* multiplicator

        
        G_distortion_loss_accum += generator_distortion_loss.item()

        if args.mode == 'GenGan':
            generator_adversary_loss = adversarial_loss(pred_secret, gender).to(device)

        else: 
            generator_adversary_loss = adversarial_loss(pred_secret, gen_secret).to(device)

        G_adversary_loss_accum += generator_adversary_loss.item()

        generator_loss = generator_distortion_loss + generator_adversary_loss * args.utility_budget

        acc_distortion_loss.append(generator_distortion_loss.item())
        acc_gender_loss.append(generator_adversary_loss.item())
        acc_generator_loss.append(generator_loss.item())

        generator_loss.backward()
        optimiser_gen.step()

        # ----------- Discriminator --------------

        optimiser_discr.zero_grad()

        real_pred_secret = disciminator(x)
        fake_pred_secret = pred_secret.detach()

        acc_true_acc.append(np.mean(torch.sigmoid(real_pred_secret).reshape(-1).clone().detach().cpu().numpy().round() == gender.detach().cpu().numpy()))
        acc_fake_acc.append(np.mean(torch.sigmoid(pred_secret).reshape(-1).clone().detach().cpu().numpy().round() == gender.detach().cpu().numpy()))

        D_real_loss = adversarial_loss_rf(real_pred_secret, gender).to(device)
        D_genderless_loss = adversarial_loss_rf(fake_pred_secret, gen_secret).to(device)

        D_real_loss_accum += D_real_loss.item()
        D_fake_loss_accum += D_genderless_loss.item()

        discriminator_loss = D_real_loss + D_genderless_loss 

        acc_genderless_loss.append(D_genderless_loss.item())
        acc_true_data_loss.append(D_real_loss.item())
        acc_dicriminator_loss.append(discriminator_loss.item())

        discriminator_loss.backward()
        optimiser_discr.step()

    list_fake_acc.append(mean(acc_fake_acc))
    list_true_acc.append(mean(acc_true_acc))
    list_generator_loss.append(mean(acc_generator_loss))
    list_distortion_loss.append(mean(acc_distortion_loss))
    list_gender_loss.append(mean(acc_gender_loss))
    list_genderless_loss.append(mean(acc_genderless_loss))
    list_true_data_loss.append(mean(acc_true_data_loss))
    list_dicriminator_loss.append(mean(acc_dicriminator_loss))

    print("==============================")
    print("epoch {}".format(epochs))

    print("list_fake_acc {}".format(mean(acc_fake_acc)))
    print("list_true_acc {}".format(mean(acc_true_acc)))
    print("diff acc {}".format(mean(acc_true_acc) - mean(acc_fake_acc)))

    print("list_generator_loss {}".format(mean(acc_generator_loss)))
    print("list_distortion_loss {}".format(mean(acc_distortion_loss)))
    print("list_gender_loss {}".format(mean(acc_gender_loss)))
    print("list_genderless_loss {}".format(mean(acc_genderless_loss)))
    print("list_true_data_loss {}".format(mean(acc_true_data_loss)))
    print("list_dicriminator_loss {}".format(mean(acc_dicriminator_loss)))
    print("==============================")

    if epochs%args.save_every == 0:
        torch.save(disciminator.state_dict(), os.path.join(out_path,'disciminator_{}_final.pth'.format(epochs)))
        torch.save(generator.state_dict(), os.path.join(out_path,'generator_{}_final.pth'.format(epochs)))
       