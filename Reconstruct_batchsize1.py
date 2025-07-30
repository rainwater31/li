# import lib
import argparse
import copy
import pickle
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
import PIL.Image as Image
import random
import math
from matplotlib import pyplot as plt  # plt提供画图工具
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
# from skimage.measure import compare_ssim, compare_psnr, compare_mse

from torchvision import datasets, transforms, models


from sklearn.preprocessing import Normalizer, StandardScaler

import torchvision.transforms.functional as F
import torch.nn.functional as F1
import cv2
import ssl
from dataset import CelebA_dataset, getDataset
from models import LeNetZhu, ResNet18, ResNet34, DnCNN, Gradinversion_lenet, Gradinversion_res18, getModel
# from mia import single_image_membership_inference


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

# 新增：异常分数计算相关模块
class Autoencoder(nn.Module):
    """用于计算异常分数的自编码器"""
    def __init__(self, in_channels=3):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def anomaly_score(ae_model, x, device):
    """计算输入图像的异常分数（基于重构误差）"""
    ae_model.eval()
    with torch.no_grad():
        x_recon = ae_model(x)
        # 计算MSE重构误差作为异常分数
        recon_error = torch.mean(torch.square(x - x_recon))
    return recon_error

# 图片格式转换
def image_to_tensor(image, shape_img, device):
    transform1 = transforms.Compose([
        transforms.CenterCrop((shape_img)),  # 只能对PIL图片进行裁剪
        transforms.ToTensor(),
    ])
    dummy_data = transform1(image)
    dummy_data = torch.unsqueeze(dummy_data, 0)
    dummy_data = dummy_data.to(device).requires_grad_(True)
    return dummy_data


def compress(gradients, compress_rate):
    mask_tuple = []
    c = np.asarray(gradients[0])
    c = abs(c.ravel())
    mask_tuple.append(np.ones(gradients[0].shape))
    for x in gradients[1:]:
        a = np.asarray(x)  # 转化为array
        a = abs(a.ravel())
        c = np.append(c, a)
        mask_tuple.append(np.ones(x.shape))
    sort_c = np.sort(c)
    top = len(sort_c)
    standard = sort_c[int(-top * compress_rate)]
    print('compress shield : ', standard)
    newgra = copy.deepcopy(gradients)
    for i in range(len(newgra)):
        p = np.asarray(newgra[i])
        m = mask_tuple[i]
        m[abs(p) < standard] = 0
        p[abs(p) < standard] = 0
        mask_tuple[i] = torch.tensor(m)
        newgra[i] = torch.tensor(p)
    return newgra, mask_tuple


use_cuda = torch.cuda.is_available()
if use_cuda:
    device = 'cuda:0'
else:
    device = 'cpu'
print(device)


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def total_variation(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

def grad2InitImg(attacked_model, gi_model, original_img, attacked_model_type="res18"):
    attacked_model.to(device)
    gi_model.to(device)
    original_img = original_img.view(-1, 3, 32, 32)
    pred = attacked_model(original_img.view(-1, 3, 32, 32))
    loss = torch.nn.CrossEntropyLoss()(pred, torch.LongTensor([0]).to(device))
    grad = torch.autograd.grad(loss, attacked_model.parameters(), retain_graph=True)
    # print(grad)
    if attacked_model_type == "res18":
        grad_input = grad[-2][:10].reshape(10, 32768, 1, 1)
    if attacked_model_type == "lenet":
        grad_input = grad[-2][:20].reshape(20, 768, 1, 1)
        # grad_input += torch.normal(mean=0.,std=0.1,size=grad_input.size()).to(device)
    # else:
    #    print("Undefined attacked_model_type")
    #    return
    recons = gi_model(grad_input)
    return recons


def HCGLA_generator(original_img, attacked_model_type="lenet"):
    if attacked_model_type == "lenet":
        lenet = LeNetZhu(num_classes=20, num_channels=3).to(device)
        gi_lenet = Gradinversion_lenet().to(device)
        gi_lenet.load_state_dict(torch.load("./models/generators/gi_lenet_epoch500_2000img(formal_class_20).pkl", map_location=torch.device(device)))
        recons = grad2InitImg(lenet, gi_lenet, original_img, attacked_model_type="lenet")
        return recons
    elif attacked_model_type == "res18":
        resnet18 = ResNet18(num_classes=10).to(device)
        gi_res18 = Gradinversion_res18().to(device)
        gi_res18.load_state_dict(torch.load("./models/generators/gi_res18_epoch1500_1500img(formal_class_10).pkl", map_location=torch.device(device)))
        recons = grad2InitImg(resnet18, gi_res18, original_img, attacked_model_type="res18")
        return recons
    else:
        print("unknown model type")
        exit()


def recovery(opt, id):
    # 新增：初始化自编码器用于计算异常分数
    ae_model = Autoencoder().to(device)
    try:
        # 尝试加载预训练自编码器权重
        ae_model.load_state_dict(torch.load(opt.ae_path, map_location=torch.device(device)))
        print("成功加载预训练自编码器模型")
    except:
        print("使用随机初始化的自编码器计算异常分数")
    
    if opt.filter and opt.filter_method == "HCGLA-Filter":
        if opt.model_type == "LeNet":
            if opt.dataset == "MNIST" or opt.dataset == "FMNIST":
                opt.filter_path = "./models/DenoisingModel/LeNet_MNIST_Filter.pth"
                denoise_layer = 48
            else:
                opt.filter_path = "./models/DenoisingModel/LeNetFilter.pth"
                denoise_layer = 17
        else:
            if opt.dataset == "MNIST" or opt.dataset == "FMNIST":
                opt.filter_path = "./models/DenoisingModel/ResNet_MNIST_Filter.pth"
                denoise_layer = 17
            else:
                opt.filter_path = "./models/DenoisingModel/net_799.pth" # net_799.pth
                denoise_layer = 48
    transform = transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        transforms.ToTensor()]) #
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])
    # shape_img = (32, 32)
    num_classes = 10
    channel = 3
    ''' load data '''
    dst, num_classes, channel = getDataset(opt.dataset, opt.model_type)
    net = getModel(num_classes, channel, opt.model_type).to(device)
    if opt.is_exist:
        net.load_state_dict(torch.load(opt.model_path, map_location='cpu'))

    criterion = nn.CrossEntropyLoss().to(device)
    # num_exp = 1
    ''' train DLG and iDLG 循环变量是图片的次序 '''

    imidx_list = []
    # 索引保存下来 因为后面需要根据这个索引去获取到真实的图像

    '''
    # 找相同类别的图像
    print("find same label images")

    for ii in range(len(dst)):
        if ii != idx and dst[ii][1] == dst[idx][1]:
            same_labels_images.append(ii)
            print("find it!!")
            break

    # Iterate through the dataset and process each image-label pair individually

    gt_data_1 = dst[id][0].to(device)
    if opt.dataset == "MNIST" or opt.dataset == "FMNIST":
        gt_data_1 = torch.cat([gt_data_1, gt_data_1, gt_data_1], dim=0)
    gt_data_1 = gt_data_1.view(1, gt_data_1.shape[0], gt_data_1.shape[1], gt_data_1.shape[2])
    gt_label_1 = torch.tensor([dst[id][1]]).to(torch.int64).to(device)

    gt_input_1 = gt_data_1
    out_1 = net(gt_input_1)

    if opt.model_type == "ResNet18":
        if opt.dataset == "CelebA":
            gt_label_input_1 = gt_label_1 * 499 // 10176
        elif opt.dataset == "lfw":
            gt_label_input_1 = gt_label_1 * 499 // 5748
        elif opt.dataset == "ImgNet":
            gt_label_input_1 = gt_label_1 * 499 // 999
        else:
            gt_label_input_1 = gt_label_1
    else:
        gt_label_input_1 = gt_label_1

    y_1 = criterion(out_1, gt_label_input_1)
    dy_dx_1 = torch.autograd.grad(y_1, net.parameters())
    original_dy_dx_1 = list((_.detach().clone().cpu() for _ in dy_dx_1))

    # Process the second image
    gt_data_2 = dst[id + 1][0].to(device)
    if opt.dataset == "MNIST" or opt.dataset == "FMNIST":
        gt_data_2 = torch.cat([gt_data_2, gt_data_2, gt_data_2], dim=0)
    gt_data_2 = gt_data_2.view(1, gt_data_2.shape[0], gt_data_2.shape[1], gt_data_2.shape[2])
    gt_label_2 = torch.tensor([dst[id + 1][1]]).to(torch.int64).to(device)

    gt_input_2 = gt_data_2
    out_2 = net(gt_input_2)

    if opt.model_type == "ResNet18":
        if opt.dataset == "CelebA":
            gt_label_input_2 = gt_label_2 * 499 // 10176
        elif opt.dataset == "lfw":
            gt_label_input_2 = gt_label_2 * 499 // 5748
        elif opt.dataset == "ImgNet":
            gt_label_input_2 = gt_label_2 * 499 // 999
        else:
            gt_label_input_2 = gt_label_2
    else:
        gt_label_input_2 = gt_label_2

    y_2 = criterion(out_2, gt_label_input_2)
    dy_dx_2 = torch.autograd.grad(y_2, net.parameters())
    original_dy_dx_2 = list((_.detach().clone().cpu() for _ in dy_dx_2))

    # Sum the gradients
    summed_gradients = [grad1 + grad2 for grad1, grad2 in zip(original_dy_dx_1, original_dy_dx_2)]

    # Do whatever is needed with summed_gradients here

    # 第二次训练，使用其中一张图片
    gt_data = gt_data_1
    gt_label = gt_label_1
    '''
    
    # 加载单张图像数据
    gt_data = dst[id][0].to(device)
    if opt.dataset == "MNIST" or opt.dataset == "FMNIST":
        gt_data = torch.cat([gt_data, gt_data, gt_data], dim=0)
    gt_data = gt_data.view(1, gt_data.shape[0], gt_data.shape[1], gt_data.shape[2])
    gt_label = torch.tensor([dst[id][1]]).to(torch.int64).to(device)

    gt_input = gt_data
    out = net(gt_input)

    if opt.model_type == "ResNet18" and opt.dataset == "CelebA":
        gt_label_input = int(gt_label) * 499 // 10176
    elif opt.model_type == "ResNet18" and opt.dataset == "lfw":
        gt_label_input = int(gt_label) * 499 // 5748
    elif opt.model_type == "ResNet18" and opt.dataset == "ImgNet":
        gt_label_input = int(gt_label) * 499 // 999
    else:
        gt_label_input = gt_label

    y = criterion(out, gt_label_input)
    dy_dx_3 = torch.autograd.grad(y, net.parameters())
    original_dy_dx_3 = list((_.detach().clone().cpu() for _ in dy_dx_3))

    # 计算两次梯度的差值
    original_dy_dx = original_dy_dx_3  # 简化为单张图像的梯度

    # 将梯度差值压缩并添加噪声
    dy_dx, mask_tuple = compress(original_dy_dx, opt.compress_rate)
    if opt.noise_level == 0:
        original_dy_dx = [i.to(device) for i in dy_dx]
    else:
        original_dy_dx = [i.to(device) + torch.normal(mean=0., std=opt.noise_level, size=i.size()).to(device) for i in
                          dy_dx]

    save_filename = '{}{}_{}_{}_{}_{}_1'.format(opt.save_path, opt.model_type, opt.method, opt.dataset, opt.compress_rate,
                                           opt.init_method)
    print('%s - %s - Filter:%s' % (opt.method, opt.init_method, str(opt.filter)))
    print("当前正在攻击图像序号:%d,标签是:%d" % (id, dst[id][1]))
    if not os.path.exists(save_filename):
        os.mkdir(save_filename)
    if opt.init_method == "Random":
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        # 产生 输出层大小的label,这的label也是一个向量，而不是具体的某个类
        print("生成随机噪点初始化")
    elif opt.init_method == "same_label":
        print("find same label images")
        same_labels_images = []
        for ii in range(len(dst)):
            # if ii != idx and int(dst[ii][1]) == int(dst[idx][1]):
            if dst[ii][1] == dst[id][1]:
                same_labels_images.append(ii)
                break
        idx1 = same_labels_images[0]
        dummy_data = dst[idx1][0].float().to(device)
        dummy_data = dummy_data.view(1, *dummy_data.size())
        dummy_data = dummy_data.requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("相同标签初始化完成")
    elif opt.init_method == 'RGB':
        if opt.dataset == 'MNIST':
            dummy_image = Image.new('1', (32, 32), 'black')
        else:
            dummy_image = Image.new('RGB', (32, 32), 'red')
        dummy_data = image_to_tensor(dummy_image, (32, 32), device)
        # 产生 输出层大小的label,这的label也是一个向量，而不是具体的某个类
        init_data = copy.deepcopy(dummy_data)
        print("RGB初始化完成")
    elif opt.init_method == "ori_img":
        tmp_img = tp(dst[id][0])
        tmp_img = np.array(tmp_img)
        tmp_img.flags.writeable = True
        # 随机改变图像像素点
        c, w, h = tmp_img.shape
        total_pixels = c * w * h
        # @frac: 随机加噪的比例
        frac = 0.2
        num_random = int(frac * total_pixels)
        # 进行随机加噪
        for idx_1 in range(num_random):
            first = random.randint(0, c - 1)
            second = random.randint(0, w - 1)
            thrid = random.randint(0, h - 1)
            tmp_img[first][second][thrid] = 0

        # dummy_data = torch.from_numpy(tmp_img).float().to(device)
        tmp_tmp = Image.fromarray(tmp_img)
        dummy_data = tt(tmp_tmp).float().to(device)
        dummy_data = dummy_data.view(1, *dummy_data.size())
        dummy_data = dummy_data.requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("原始图片初始化完成")
    elif opt.init_method == "img":
        # 特定图片初始化
        img = Image.open(opt.img_path)
        image_initial = transform(img)
        image_initial = image_initial.to(device)
        dummy_data = image_initial.view(1, image_initial.shape[0], image_initial.shape[1],
                                        image_initial.shape[2])  # .to(device).detach()
        dummy_data = dummy_data.requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("一张图片初始化完成")
    elif opt.init_method == "generator":
        if opt.model_type == "LeNet":
            dummy_data = HCGLA_generator(gt_data[0], attacked_model_type="lenet")
        elif opt.model_type == "ResNet18":
            dummy_data = HCGLA_generator(gt_data[0], attacked_model_type="res18")
        else:
            print("Undefined attacked_model_type")
            exit()
        dummy_data = tt(tp(dummy_data)).to(device)
        dummy_data = dummy_data.view(1, *dummy_data.size()).requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("生成器初始化完成")
    else:
        print("Someting is wrong with the initial method, try it again!")
        return 0

    # 定义优化器
    if opt.method == 'DLG':
        dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=opt.lr)
    elif opt.method == 'geiping':
        label_pred = torch.argmin(torch.sum(original_dy_dx[-1], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(
            False)
        optimizer = torch.optim.Adam([dummy_data], lr=opt.lr)
        print("预测的标签是:", label_pred)
    elif opt.method == 'GI':
        # 假设 original_dy_dx 是原始梯度序列，dummy_data 是待优化的输入数据
        optimizer = torch.optim.Adam([dummy_data], lr=opt.lr)
    else:
        label_pred= gt_label
        optimizer = torch.optim.Adam([dummy_data], lr=opt.lr)
        print("预测的标签是:", label_pred)

    history = []
    history_iters = []
    losses = []
    mses = []
    psnrs = []
    ssims = []
    anomaly_scores = []  # 新增：保存异常分数
    train_iters = []
    print('lr =', opt.lr)


    for iters in range(opt.Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            
            # 计算总变差损失
            def total_variation_loss(dummy_data):
                vertical_diff = dummy_data[:, :, 1:, :] - dummy_data[:, :, :-1, :]
                horizontal_diff = dummy_data[:, :, :, 1:] - dummy_data[:, :, :, :-1]
                return torch.sum(torch.abs(vertical_diff)) + torch.sum(torch.abs(horizontal_diff))

            tv_loss = total_variation_loss(dummy_data)
            
            # 计算基础损失
            if opt.method == 'DLG':
                dummy_loss = - torch.mean(
                    torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
            else:
                dummy_loss = criterion(pred, gt_label)
            
            # 新增：计算异常分数损失
            current_anomaly_score = anomaly_score(ae_model, dummy_data, device)
            
            # 新增：组合损失函数（加入异常分数损失）
            total_loss = dummy_loss + 0.001 * tv_loss + opt.anomaly_weight * current_anomaly_score
            
            # 计算梯度
            dummy_dy_dx = torch.autograd.grad(total_loss, net.parameters(), create_graph=True)
            dummy_dy_dx = list(dummy_dy_dx)

            if opt.method == 'HCGLA':
                i = 0
                for tmp_1, tmp_2 in zip(dummy_dy_dx, mask_tuple):
                    dummy_dy_dx[i] = torch.mul(tmp_1, tmp_2.to(device))
                    i += 1
            grad_diff = 0
            if opt.method == 'geiping':
                ex = original_dy_dx[0]
                weights = torch.arange(len(original_dy_dx), 0, -1, dtype=ex.dtype, device=ex.device)/ len(original_dy_dx)
                for ii in range(len(original_dy_dx)):
                    grad_diff += 1 - torch.cosine_similarity(dummy_dy_dx[ii].flatten(), original_dy_dx[ii].flatten(), 0, 1e-10) * weights[ii]
                    grad_diff += ((dummy_dy_dx[ii] - original_dy_dx[ii]) **2).sum()
                grad_diff += total_variation(dummy_data)
            elif opt.method == 'HCGLA':
                if opt.model_type == "LeNet":
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx_3):
                        gx = gx.to(device)
                        gy = gy.to(device)
                        grad_diff += ((gx - gy)** 2).sum()
                else:
                    gx = dummy_dy_dx[-2]
                    gy = original_dy_dx[-2]
                    grad_diff += ((gx - gy) **2).sum()
            else:
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy)** 2).sum()
            
            # 新增：将异常分数损失加入梯度差异
            grad_diff += opt.anomaly_weight * current_anomaly_score
            
            grad_diff.backward()
            return grad_diff

        # 计算评估指标
        dummy_data_np = dummy_data[0].detach().cpu().numpy().transpose(1,2,0)
        gt_data_np = gt_data[0].detach().cpu().numpy().transpose(1,2,0)
        dummy_data_np = dummy_data_np.astype(np.float64)
        gt_data_np = gt_data_np.astype(np.float64)
        mses.append(mean_squared_error(dummy_data_np, gt_data_np))
        psnrs.append(10 * np.log10(255 * 255 / mses[-1]))
        data_range = 255
        ssims.append(structural_similarity(dummy_data_np, gt_data_np, multichannel=True,data_range=data_range))
        
        # 新增：计算并保存异常分数
        current_anomaly = anomaly_score(ae_model, dummy_data, device).item()
        anomaly_scores.append(current_anomaly)
        
        # 执行优化
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        
        # 打印信息（包含异常分数）
        if not opt.isSave:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters,
                  'loss = %.8f, mse = %.8f, psnr:%.8f, ssim:%.8f, anomaly:%.8f' % 
                  (current_loss, mses[-1], psnrs[-1], ssims[-1], current_anomaly))

        # 保存中间结果
        if opt.isSave and ( iters in [1,5,15,20,25, 30, 35, 40, 45,50,110,120,130,140,150,160,170,180,190,200] or iters % 100 == 0) and iters > 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters, 'loss = %.8f, mse = %.8f, psnr:%.8f, ssim:%.8f, anomaly:%.8f' % 
                  (current_loss, mses[-1], psnrs[-1], ssims[-1], current_anomaly))
            history.append(tp(dummy_data[0].cpu()))
            history_iters.append(iters)

            fig = plt.figure(figsize=(12, 8))
            if opt.filter :
                plt.suptitle('%s %s %s %s %s on %s %05dth picture' %(opt.model_type, opt.method, opt.init_method, opt.compress_rate,
                                                                     opt.filter_method, opt.dataset, id))
            else:
                plt.suptitle('%s %s %s %s on %s %05dth picture' % (
                opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.dataset, id))
            plt.subplot(4, 11, 1)
            plt.imshow(tp(gt_data[0].cpu()))
            plt.axis('off')
            plt.subplot(4, 11, 2)
            plt.imshow(tp(init_data[0]))
            plt.title('Initial', fontdict={"family": "Times New Roman", "size": 16})
            plt.axis('off')
            for i in range(min(len(history), 42)):
                if i > 0 and history_iters[i] == history_iters[i - 1]:
                    ax1 = fig.add_subplot(4, 11, i + 3)
                    plt.subplot(4, 11, i + 3)
                    plt.imshow(history[i])
                    plt.title('DeNoise', fontweight='heavy', color='red', fontdict={"family": "Times New Roman", "size": 16})
                    ax1.spines['top'].set_linewidth(4.0)
                    ax1.spines['right'].set_linewidth(4.0)
                    ax1.spines['bottom'].set_linewidth(4.0)
                    ax1.spines['left'].set_linewidth(4.0)
                    ax1.spines['top'].set_color('red')
                    ax1.spines['right'].set_color('red')
                    ax1.spines['bottom'].set_color('red')
                    ax1.spines['left'].set_color('red')
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                else:
                    plt.subplot(4, 11, i + 3)
                    plt.imshow(history[i])
                    plt.title('iter=%d' % (history_iters[i]), fontdict={"family": "Times New Roman", "size": 16})
                    plt.axis('off')
            plt.tight_layout()
            if i!= 5:
                plt.savefig('%s/%s_%s_%s_%s_%s_on_%s_%05d.jpg' % (
                    save_filename, opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.filter_method, opt.dataset, id))
            else:
                plt.savefig('%s/%s_%s_%s_%s_on_%s_%05d.png' % (
                    save_filename, opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.dataset, id))
            plt.close()
        if opt.isSave and (iters == 90):
            fig = plt.figure(figsize=(12, 8))
            plt.suptitle('第90轮图像', fontsize=16)
            plt.subplot(1, 2, 1)
            plt.imshow(tp(gt_data[0].cpu()))
            plt.axis('off')
            plt.title("Ground Truth")
            plt.tight_layout()
            save_path = r'F:\下载\机器遗忘初版\机器遗忘\90'
            plt.savefig(
                f'{save_path}/iteration_90_{opt.model_type}_{opt.method}_{opt.init_method}_{opt.compress_rate}_{opt.dataset}_{id}.png')
            plt.close()

        if opt.filter and iters % opt.filter_frequency == 0 and iters > 0:
            dummy_img = tp(dummy_data[0].cpu())
            dummy_name = '%s/%s_%s_%s_%s_%s_on_%s_%05d_end.jpg' % (
                save_filename, opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.filter_method,
                opt.dataset, id)
            dummy_img.save(dummy_name)

            # HLA去噪
            if opt.filter_method == "HCGLA-Filter":
                filternet = DnCNN(channels=3, num_of_layers=denoise_layer)
                filterModel = nn.DataParallel(filternet, device_ids=[0]).to(device)
                filterModel.load_state_dict(torch.load(opt.filter_path, map_location=torch.device(device)))
                filterModel = filterModel.module
                filterModel.eval()
                img = Image.open(dummy_name)
                img = tt(img)
                img = img.view(1, *img.size()).to(device)
                with torch.no_grad():
                    Out = torch.clamp(img - filterModel(img), 0., 255.)
                img = tt(tp(Out[0].cpu())).to(device)
            else:
                print("No such filter method")
                exit()
            dummy_data = img.view(1, img.shape[0], img.shape[1],
                                  img.shape[2]).to(device)
            dummy_data = dummy_data.requires_grad_(True)
            if opt.method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=opt.lr)
            if opt.method == 'geiping':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=opt.lr)
            else:
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=opt.lr)
            if opt.isShow:
                history.append(tp(dummy_data[0].cpu()))
                history_iters.append(iters)
        if current_loss < 0.0000000000000000006:  # converge
            break

    print('imidx_list:', imidx_list)

    loss = losses
    label = label_pred.item()
    print('PSNR:', psnrs[-1])
    print('loss {}:'.format(opt.method), loss[-1])
    print('mse_{}:'.format(opt.method), mses[-1])
    print('最终异常分数:', anomaly_scores[-1])  # 新增：打印最终异常分数
    print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_{}:'.format(opt.method), label)
    dummy_img = tp(transform(tp(dummy_data[0].cpu())))

    if opt.filter:
        dummy_name = '%s/%s_%s_%s_%s_%s_on_%s_%05d_end.jpg' % (
            save_filename, opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.filter_method,
            opt.dataset, id)
    else:
        dummy_name = '%s/%s_%s_%s_%s_on_%s_%05d_end.jpg' % (
            save_filename, opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.dataset, id)
    dummy_img.save(dummy_name)
    if mses[-1] < 0.05:
        print('success')
    else:
        print("fail")
    return loss, mses, psnrs, ssims, anomaly_scores  # 新增：返回异常分数


def getOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compress_rate', default=0.01, type=float, help='The rate we compress our gradients')

    # about model
    parser.add_argument('--model_type', default='LeNet', help='you can choose from LeNet, ResNet18, ResNet34')
    parser.add_argument('--model_path', default='', help=" model path ")
    parser.add_argument('--is_exist', default=False, help='Is the attacked model exist ? or we need to random init it')

    # about data
    parser.add_argument('--dataset', default='pubface', help='you can choose from CelebA, lfw, pubface, google, cifar100, ImgNet')

    # init method
    parser.add_argument('--init_method', default='generator',
                        help="you can choose 'Random', 'same_label', 'ori_img', 'RGB', 'img', 'generator'")

    # save path
    parser.add_argument('--save_path', default='./recover_result/image/', help='the path to save recover images')
    parser.add_argument('--isSave', default=True, help='')

    # recovery method
    parser.add_argument('--method', default='DLG', help='you can choose from DLG, iDLG, geiping, GI,HCGLA')

    # hyper parameter
    parser.add_argument('--img_path', default='./data/img.jpg', help='')
    parser.add_argument('--lr', default=0.1, type=float, help='')
    parser.add_argument('--Iteration', default=100, type=int, help='')
    parser.add_argument('--aux_path', default='./data/Auxiliary', help='')
    parser.add_argument('--batchsize', default=1)

    # about filter
    parser.add_argument('--filter', default=True, type=bool, help='')
    parser.add_argument('--filter_method', default='HCGLA-Filter',
                        help="you can choose from 'HCGLA-Filter','mean','Guassian','median','bilater' ")
    parser.add_argument('--filter_path', default='./models/filterModel/DnCNN/net_799.pth', help='')
    parser.add_argument('--filter_frequency', default=5, type=int)
    parser.add_argument('--isShow', default=True, help="")

    # add noise
    parser.add_argument('--noise_level', default=0, type=float, help="")
    
    # 新增：异常分数损失相关参数
    parser.add_argument('--anomaly_weight', default=0.1, type=float, help='异常分数损失的权重')
    parser.add_argument('--ae_path', default='./models/autoencoder.pth', help='自编码器模型路径')

    opt = parser.parse_args()
    return opt


def main():
    opt = getOpt()
    sImages = [1000, 2000]
    all_mses = []
    all_psnrs = []
    all_ssims = []
    all_anomaly_scores = []  # 新增：保存所有图像的异常分数
    for i in sImages:
        losses, mses, PSNR, ssim, anomaly = recovery(opt, i)  # 新增：接收异常分数
        all_mses.append(mses)
        all_psnrs.append(PSNR)
        all_ssims.append(ssim)
        all_anomaly_scores.append(anomaly)  # 新增：保存异常分数
    if opt.filter:
        filename = "./recover_result/data/" + opt.model_type + "_" + opt.method + "_1" + "_" + opt.dataset + "_" + str(opt.filter_method) + "_" + str(opt.compress_rate) + "_" \
                   + opt.init_method + ".npz"
    else:
        filename = "./recover_result/data/" + opt.model_type + "_" + opt.method + "_1" + "_" + opt.dataset + "_" + str(opt.compress_rate) + "_" + opt.init_method + ".npz"
    # 新增：保存异常分数
    np.savez(filename, all_losses=all_losses, all_mses=all_mses, all_psnrs=all_psnrs, 
             all_ssims=all_ssims, all_anomaly_scores=all_anomaly_scores)


if __name__ == '__main__':
    main()
