# Copyright (c) [2012]-[2021] MJZ@.

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from timm import utils

from models import *
from timm.models import *
from timm.models import create_model  # MLP-like; resnet, efficient等

# from models_codes import *
# from models_codes.factory import create_model, parse_model_name, safe_model_name    # transformer框架

import numpy as np
import matplotlib.pyplot as plt

import datetime
from load_data_DNDL_TonDN import get_TongueDiseasePosition, get_TongueDiseaseSymptom

from torch.optim import lr_scheduler
import time
import copy
from sklearn.metrics import roc_auc_score, roc_curve, auc
from eval_mAP import mAP_eval, AP_partial, AverageMeter
from multilabel_loss_function import AsymmetricLoss, FocalLoss, myBFLoss, wDice_loss, ASL_Loss, GHME_Loss, GHM_Loss, myGLoss, meanGLoss,Hill
from multilabel_loss_function import APLLoss, OTLoss, MLpoly_OT, BMLpoly


# ---------- 固定随机种子 ----------  # seed; 42; 2019, 123, 2026, 1
def setup_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # 关闭自动优化以保证可复现

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--dataset', type=str, default='multilabel_face_dataset', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
parser.add_argument('--b', type=int, default=64, help='batch size')
parser.add_argument('--img-size', type=int, default=224, metavar='N', help='Image patch size (default: None => model default)')
args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
best_acc = 0     # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# data_mean, data_std = [0.40973431 0.43273546 0.54664054], [0.22713209 0.22647621 0.24762501]  # tongue_dataset1
# data_mean, data_std = [0.4914228  0.4994282  0.73754448], [0.17772971 0.1571762  0.17106681]  # tongue_dataset2

print('==> Preparing data..')
# face
# transform_train = transforms.Compose([
#     transforms.Resize(args.img_size),
#     transforms.RandomCrop(args.img_size, padding=(args.img_size//8)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.43966475, 0.47025456, 0.4956167), (0.26568854, 0.27167103, 0.26902942)),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize(args.img_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.43966475, 0.47025456, 0.4956167), (0.26568854, 0.27167103, 0.26902942)),
# ])

# tongue, 5992imges数据集
transform_train = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.RandomCrop(args.img_size, padding=(args.img_size//8)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914228,0.4994282,0.73754448), (0.17772971,0.1571762,0.17106681)),
])

transform_test = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914228,0.4994282,0.73754448), (0.17772971,0.1571762,0.17106681)),
])


if args.dataset=='Chinese-Herbs98':
    # print('Please use cifar10 or cifar100 dataset.')
    # ============== 加载新的数据集 ===============
    root = "/home/b3432/Code/experiment/zhmengjian/wavemlp_pytorch/data"
    args.num_classes = 98
    DATASET_NAME = "Chinese-Herbs98"
    # trainloader, val, testloader = data_loader(root, DATASET_NAME, batch_size=args.b)
else:   # 多标签分类数据集加载
    # ================ tongue data ================
    # train_ds, test_ds = get_TongueDiseasePosition(transform_train, transform_test) # Tongue_dataset1
    train_ds, test_ds = get_TongueDiseaseSymptom(transform_train, transform_test)   # Tongue_dataset2

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=args.b, shuffle=True,num_workers=8)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=args.b, num_workers=8)    
  

# print(f'learning rate:{args.lr}, weight decay: {args.wd}')

# === Training position and disease of mutil-label classes task ===
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # print(model)
    Sigmoid_fun = nn.Sigmoid()
    since = time.time()
    best_acc = 0.80 # 开始保存模型的Acc;face-->0.85,tongue-->0.80
    best_map = 0.50 # 开始保存模型的mAP;face-->0.60,tongue-->0.41
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:1')
    criterion = criterion.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # 每训练一个epoch，验证一下网络模型
        for phase in ['train', 'test']:
            running_loss = 0.0
            running_Acc = 0.0
            running_precision = 0.0
            running_recall = 0.0
            batch_num = 0
            running_mAP = 0.0
            running_AUC = 0.0
            result_list = []
            label_list = []
            test_APs = []
            test_F1 = []

            if phase == 'train':
                scheduler.step()    # 学习率更新方式
                model.train()       # 调用模型训练
                # result_list = []
                # label_list = []

                # 依次获取所有图像，参与模型训练或测试
                for inputs, labels in trainloader:
                    # if use_gpu:     # 判断是否使用gpu
                    #     inputs = inputs.cuda()
                    #     labels = labels.cuda()
                    # else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()   # 梯度清零
                    outputs = model(inputs).to(device) # 网络前向运行
                    # print(outputs)
                    loss = criterion(Sigmoid_fun(outputs), labels)   # 计算Loss值
                    # loss = criterion(outputs, labels)   # 计算Loss值

                    # result = nn.Sigmoid()(outputs).cpu().detach().numpy().tolist()  # 经过sigmoid得到各类别概率
                    # target = labels.cpu().detach().numpy().tolist()
                    # for k in range(len(inputs)):
                    #     result_list.append({"scores": result[k]})
                    #     label_list.append({"scores": target[k]})
                    # mAP = calculate_mAP (result=result_list, ann_path=label_list, mode=1)

                    # 模型预测结果准确率的函数
                    precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)
                    micro_Acc = calculate_acuracy_mode_zero(Sigmoid_fun(outputs), labels)
                    
                    running_Acc += micro_Acc
                    running_precision += precision
                    running_recall += recall
                    batch_num += 1
                    # running_mAP += mAP
                    
                    loss.backward()     # 反传梯度
                    optimizer.step()    # 更新权重

                    running_loss += loss.item() * inputs.size(0)    # 计算一个epoch的loss值和准确率
            else:
                with torch.no_grad():   # 取消验证阶段的梯度
                    model.eval()
                    # cal_mAPs(testloader, model)
                    run_AUC, run_mAP, run_Acc, run_F1 = cal_mAPs(testloader, model)
                    for inputs, labels in testloader:
                        # if use_gpu:     # 判断是否使用gpu
                        #     inputs = inputs.cuda()
                        #     labels = labels.cuda()
                        # else:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs).to(device) # 网络前向运行
                        # print(outputs)
                        pred = Sigmoid_fun(outputs).to(device)

                        result = nn.Sigmoid()(outputs).cpu().detach().numpy().tolist()  # 经过sigmoid得到各类别概率
                        target = labels.cpu().detach().numpy().tolist() # 真实标签
                        for k in range(len(inputs)):
                            result_list.append({"scores": result[k]})
                            label_list.append({"scores": target[k]})

                        mAP, APs = calculate_mAP(result=result_list, ann_path=label_list, mode=2)

                        # BCELoss的输入（1、网络模型的输出必须经过sigmoid；2、标签必须是float类型的tensor）
                        # loss = criterion(outputs, labels) # 计算Loss
                        loss = criterion(pred, labels)      # 计算Loss
                        # 计算each epoch的loss值和准确率
                        running_loss += loss.item() * inputs.size(0)

                        # 模型预测结果准确率的函数
                        micro_Acc = calculate_acuracy_mode_zero(Sigmoid_fun(outputs), labels)
                        precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)

                        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

                        running_Acc += micro_Acc
                        running_precision += precision
                        running_recall += recall
                        batch_num += 1
                        # ============
                        running_AUC = run_AUC   # 舌象特征AUC
                        running_mAP = mAP   # 舌象特征mAP
                        test_APs = APs      # 每个舌象特征的AP准确率
                        test_F1 = f1_score
                    
            # =========== 计算Loss和准确率的均值 ============
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f} '.format(phase, epoch_loss))
            epoch_Acc = running_Acc / batch_num
            print('{} Acc: {:.4f} '.format(phase, epoch_Acc))
            epoch_precision = running_precision / batch_num
            print('{} Precision: {:.4f} '.format(phase, epoch_precision))
            epoch_recall = running_recall / batch_num
            print('{} Recall: {:.4f} '.format(phase, epoch_recall))
            # ====================
            epoch_mAP = running_mAP # 舌象特征mAP
            epoch_APs = test_APs
            print('{} mAP: {:.4f} '.format(phase, epoch_mAP))
            print('APs:', epoch_APs)
            print('F1:', test_F1)
            epoch_AUC = running_AUC # 舌象特征AUC
            print('{} AUC: {:.4f} '.format(phase, epoch_AUC))

            if phase == 'train':
                total_train_acc.append(epoch_Acc)
                total_train_loss.append(epoch_loss)
                # total_train_mAP.append(epoch_mAP)
                print("total_trACC:",total_train_acc)
            else:
                total_test_acc.append(epoch_Acc)
                total_test_loss.append(epoch_loss)
                recall_test.append(epoch_recall)
                precision_test.append(epoch_precision)
                total_test_mAP.append(epoch_mAP)
                total_test_AUC.append(epoch_AUC)
                print("total_teACC:",total_test_acc)

        # if phase == 'test' and epoch_Acc > best_acc:
        if phase == 'test' and run_mAP > best_map:    # 改用mAP作为核心指标
            best_acc = run_Acc
            best_map = run_mAP    # 增加20240922
            best_AUC = run_AUC    # 增加20240922
            best_model_wts = copy.deepcopy(model.state_dict())
            print('Saving model...')
            state = {
                'net': net.state_dict(),
                'acc': run_Acc,
                'map': run_mAP,   # 增加20240922
                'AUC': run_AUC,   # 增加20240922
                'F1': run_F1,     # 增加20240922
                'epoch': epoch,
            }
           
            save_dir = os.path.join(args.output, f'ckpt_{args.dataset}_{args.model}_{args.loss}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(state, os.path.join(save_dir, f'ckpt_{best_acc:.2f}_{best_map:.2f}_{best_AUC*100:.2f}_{run_F1:.2f}.pkl'))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val mAP: {:4f}'.format(best_map))
    
    model.load_state_dict(best_model_wts)   # 网络导入最好的网络权重

    return model

# 计算准确率Acc，直接计算预测正确的标签数
def calculate_acuracy_mode_zero(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测该类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th   # 网络输出的结果model_pred与0.5比较，超过0.5判为true
    pred_result = pred_result.float()        # 将true、false改为1、0 ，此为预测结果
    
    compare = pred_result.eq(labels).float() # 比较预测与实际是否相同，相同为true，不同为false，改为1、0
    # print(compare.shape)
    num = torch.sum(compare)    # 判断正确的标签个数（num为tensor格式）
    # print(num)
    ma_l, ma_w = labels.shape   # 标签size为(64,12) = (batch_sieze, label)
    # print(labels.shape)
    ma_size = ma_l * ma_w       # 总数为64*12 = batch_sieze * label
    # print(ma_size)
    micro_Acc = num.item() / ma_size  # Acc准确率，即预测正确的标签数，微平均
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0.
    
    return micro_Acc
    
# 计算精准率和召回率，方式1
def calculate_acuracy_mode_one(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    eps = 1e-8
    pred_result = model_pred > accuracy_th  # 网络输出的结果model_pred与0.5比较，超过0.5判为true
    pred_result = pred_result.float()       # 将true、false改为1、0 ，此为预测结果

    # print(labels)
    # print(pred_result)
    # print(pred)

    pred_one_num = torch.sum(pred_result)   # TP+FP
    # print(pred_one_num)

    if pred_one_num == 0:
        return 0., 0.
    
    target_one_num = torch.sum(labels)      # TP+FN
    true_predict_num = torch.sum(pred_result * labels)  # TP
    # print(target_one_num)
    # print(true_predict_num)
    
    precision = true_predict_num / pred_one_num # 模型预测的结果中有多少个是正确的
    recall = true_predict_num / target_one_num  # 模型预测正确的结果中，占所有真实标签的数量
        
    # pred_thresh = (model_pred > accuracy_th)*1.0
    # tp = torch.sum((pred_thresh == labels)*(labels==1.0), dim=0)
    # fp = torch.sum((pred_thresh != labels)*(labels==0.0), dim=0)
    # fn = torch.sum((pred_thresh != labels)*(labels==1.0), dim=0)
    # recall = tp / (fn + tp + eps)
    # precision = tp / (fp + tp + eps)

    f1_score = 2 * precision * recall / (precision + recall + eps)

    return precision.item(), recall.item()#, f1_score.item()


# ========================计算mAP指标=====================
def calculate_mAP (result, ann_path, mode):
    max_mAP = 0.0
    classes = ['0','1','2','3','4','5','6','7','8','9','10','11','12']  # 舌象特征13个类：舌质、舌苔

    aps = np.zeros(len(classes), dtype=np.float64)

    pred_json= result      # 预测标签值
    ann_json = ann_path    # 实际标签

    for i, _ in enumerate(classes):     # 列举多个类别
        ap = json_map(i, pred_json, ann_json)
        aps[i] = ap
    mAP = np.mean(aps)
    # print("AP:",aps)    # 输出每个标签的准确率,自动叠加至测试样本数量

    if mode == 1:    # training stage
        print("mAP: {:4f}".format(mAP))
    elif mode == 2:  # testing stage
        if mAP > max_mAP:
            max_mAP = mAP
            # print("mAP: {:4f}".format(mAP), "max_mAP: {:4f}".format(max_mAP))
        else:
            print("mAP: {:4f}".format(mAP))

    return 100 * mAP.item(), 100 * aps

def json_map(cls_id, pred_json, ann_json):  # 对每个标签类cls_id，所有图片求准确率
    assert len(ann_json) == len(pred_json)
    num = len(ann_json)     # batch_size的图片数量
    predict = np.zeros(num, dtype=np.float64)
    target = np.zeros(num, dtype=np.float64)

    for i in range(num):    # 对每一张图片
        predict[i] = pred_json[i]["scores"][cls_id] # 预测标签值
        target[i] = ann_json[i]["scores"][cls_id]   # 实际标签

    tmp = np.argsort(-predict)  # 取预测概率从大到小排序的索引
    # print(tmp)
    target = target[tmp]        # 按预测概率从大到小排序
    # print("target:",target)

    index_id = 1
    if index_id ==0:
    # =========== method one ===========
        pre, obj = 0.0, 0.0
        esp = 1e-8
        for i in range(num):    # 设有M个正样本，则有M个label为1，取M个recall值,从0开始
            if target[i] == 1:  # 每当label为1，就是这个recall值下的precision最大的时候，取M个这个最大pre求平均即ap值
                obj += 1.0      # 0~M
                pre += obj / (i+1)  # 最大precision
        pre += pre              # Sum of the Max precision 
        # print("obj:",obj)
        # print("pre:", pre)
        AP = pre / (obj+ esp)   # 最大pre求平均

    elif index_id ==1:
    # =========== method two ===========
        pos_count = 1e-8
        total_count = 0.0
        pre_at_i = 0.0
        for i in range(num):
            label = target[i]
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                pre_at_i += pos_count / total_count
        # print("obj:", obj)
        # print("pre:", pre_at_i)
        pre_at_i /= pos_count   # 最大pre求平均 
        AP = pre_at_i

    return AP


# ==================== 计算AUC ====================
def calculate_AUC(label, preds):
    fpr = np.zeros((preds.shape[1]))
    tpr = np.zeros((preds.shape[1]))
    auc_score = np.zeros((preds.shape[1]))
    roc_auc = np.zeros((preds.shape[1]))

    for k in range(preds.shape[1]):
        scores = preds[:, k]
        target = label[:, k]

        fpr, tpr, _ = roc_curve(target, scores)
        # print(fpr.shape, tpr.shape)

        roc_auc[k] = auc(fpr, tpr)
        auc_score[k] = roc_auc_score(target, scores)
     
    return auc_score.mean(), roc_auc.mean(), roc_auc


# ============== add parameters ==================
def cal_mAPs(val_loader, model):
    Sig = torch.nn.Sigmoid()

    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    acc = []

    for i, (input, target) in enumerate(val_loader):
        target = target
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda(1))).cpu()
        num_classes = output.shape[-1]

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(0.5).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)
        mic_acc = (tp.sum().float() + tn.sum().float()) / (num_classes*count) * 100.0

    print('==================================================================')
    print(' * P_C {:.4f} R_C {:.4f} F_C {:.4f} P_O {:.4f} R_O {:.4f} F_O {:.4f} Acc {:.4f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o, mic_acc))

    # mAP_score, AP_score = mAP_eval(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    # print("mAP score:", mAP_score,"AP_score:",AP_score)
    
    _, auc, auc_ap = calculate_AUC(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("auc:",auc*100,"auc_ap:",auc_ap*100)

    AP_score1, mAP_score1, macro_AP = AP_partial(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("AP_score:",AP_score1,"mAP score:", mAP_score1,"macro_AP:",macro_AP )

    return auc, mAP_score1, mic_acc, f_o    #(AUC, mAP, Acc, F1)

if __name__ == '__main__':

    dataset_sizes = {'train': train_ds.__len__(), 'test': test_ds.__len__()}
    # use_gpu = torch.cuda.is_available()

    # [1,2,3,4]: vip_T(10layer-256-512); focalnet_tiny_T(10layer-16patch-768); mobilenetv3_small_100;efficientnet_b3(timm) 
    
    # [10,11,12]:vit_small_patch16_224(384); swin_v2_cr_tiny_ns_224(4patch-768);convnext_nano_hnf (自带)
    # [6,7,8,9]: convit_small(432);nest_tiny(384)速度慢;poolformer_s12_T(7,768);pit_s_224(576);

    # 调试(adamw优化器):focalnet_tiny_Ts (adamw:4e-3,1e-3); efficientnet_b3
    # 调试(adamw优化器4e-3-1e-5):focalnet_tiny_T; efficientnet_b3

    # model_id = [2,4,5,8] # 原始sgd(0.01-2e-4);调试(adamw优化器4e-3-1e-5),被挤断了
    # model_id = [2,4,8,6,7,3,1] # 原始sgd(0.01-2e-4);调试(adamw优化器4e-3-1e-5), 128 batch-size

    # ==== 20240505 
    # model_id = [10,2] # 调试(sgd优化器0.01-2e-5)，64 batch-size
    # model_id = [11,8] # 调试(sgd优化器0.01-2e-5)，64 batch-size, 20240511

    # model_id = [10] # HWCATMLP_Ttoken,20240512,调试(sgd优化器0.01-2e-5); 
    # model_id = [12] # HWvipMLP_Ttoken,20240514,调试(sgd优化器0.01-2e-5);
    # model_id = [13] # DFTCubicMLP_T, 调试(sgd优化器0.01-2e-5)，64 batch-size
    
    # 20260205， MLP-like模型测试
    # model_id = [12,2,6,7,3,1,14] # sgd(0.01-2e-5), 202340610, 20240613
    # model_id = [12,6,7,3,1,14,18,19,20] # sgd(0.01-2e-4), 20260205, 64-batch, 
    # model_id = [10,11,4,2,8,13,5] # [9,10] # sgd(0.01-2e-4), 20260206, 64-batch
    # model_id = [21,22]  # 测试,20260218,batch_size:64--32
    # model_id = [21,23,2,8,13,9]  # 消融测试v6版本，20260220;
    # model_id = [21,23]  # v7, 20260224
    # model_id = [26,27,18,28,29,25]  # 超图, 20260304
    # model_id = [24,34,35] # 31,30,32, 33,34， 24
    # model_id = [40,41,42,39,38]  # 202603014, 36,37,38; 39-42
    # model_id = [48,35] # 43，[44,45],47
    # model_id = [49,50]
    # 超图验证，seed, 20260913
    # model_id = [26,27,28,29,30,32,33,34,51,52,53,54,55,45] # 57
    model_id = [33,34,51,52,53,54,55,45]
                
    # losses_id = [10]  
    # losses_id = [12,13,1,2,3] # 固定算法，优化Loss
    # losses_id = [15, 16, 14, 2, 17] # 固定算法，优化Loss，[12,13,1,2,3]; 增加APL损失; [15,14,12]
    # losses_id = [17,15,16,2,0]
    for id in model_id:       # 多种模型对比
    # for loss_id in losses_id:
    # =================== create model =======================
        # id = 55  # 【51，54, 55】；34-pvihg_kmeans_ti_224; 33-vihg_kmeans_ti_224
        print("========id:=======",id)
        print('==> Building model..')   # 原始sgd(0.01-2e-4); adamw(5e-4,1e-5); adamw(2e-2,1e-3)
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')    # tongue,0.01; 
        parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
        parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')#min-lr=2e-4,调整（2023.08.28）:min-lr=1e-3
        # parser.add_argument('--lr', default=4e-3, type=float, help='learning rate') 
        # parser.add_argument('--min-lr', default=1e-5, type=float, help='minimal learning rate')#min-lr=2e-4,调整（2023.08.28）:min-lr=1e-3
       
        # 补充五折实验，20240610
        # parser.add_argument('--dataset', type=str, default='Tondataset1_fold1_sgd_20260203', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='TonD1_fold2_sgd64_20260224v7test', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='Tondataset1_fold3_sgd_20260203', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='Tondataset1_fold4_sgd_20260203', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='Tondataset1_fold5_sgd_20260203', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        parser.add_argument('--dataset', type=str, default='TonD1_f2_sgd64_seed42_0922', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='TD1_fold2_adamw64_20260318', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98

        parser.add_argument('--b', type=int, default=64, help='batch size') # 32-64-128-256（128测试）；
        parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version of specified network (if avail)')
        parser.add_argument('--num-classes', type=int, default=13, metavar='N', help='number of label classes (default: 1000)')
        if id == 1:
            parser.add_argument('--model', default='vip_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 2:
            parser.add_argument('--model', default='DFTSinusoidalMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 3:
            parser.add_argument('--model', default='ActivexTiny', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 4:
            parser.add_argument('--model', default='WaveMLP_Tcct', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 5:
            parser.add_argument('--model', default='HWCATMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 6:
            parser.add_argument('--model', default='CycleMLP_B1', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 7:
            parser.add_argument('--model', default='WaveMLP_Tor', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 8:
            parser.add_argument('--model', default='DFTchaosMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 9:
            parser.add_argument('--model', default='FFTmixerMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 10:
            parser.add_argument('--model', default='HWCATMLP_Ttoken', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 11:
            parser.add_argument('--model', default='PWIMLP_Ttoken', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 12:
            parser.add_argument('--model', default='HWvipMLP_Ttoken', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 13:
            parser.add_argument('--model', default='DFTCubicMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 14:
            parser.add_argument('--model', default='spanet_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 15:
            parser.add_argument('--model', default='nest_tiny', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 16:
            parser.add_argument('--model', default='convit_small', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 17:
            parser.add_argument('--model', default='vit_small_patch16_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 18:
            parser.add_argument('--model', default='HAT_Net_tiny', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 19: # [26,27,28,29,30,32,33,34,51,52,53,54,55,57]
            parser.add_argument('--model', default='vig_ti_224_gelu', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 20:
            parser.add_argument('--model', default='pvig_ti_224_gelu', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 21:  #补充新方法，20260208
            parser.add_argument('--model', default='PeDFWMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 22:
            parser.add_argument('--model', default='PeFDmixMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 23:
            parser.add_argument('--model', default='PeDFWDSMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 24:  # 改进的代码+图卷积
            parser.add_argument('--model', default='PHWDFWMLPgraph_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 25:
            parser.add_argument('--model', default='HAT_Net_small', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 26: #超图网络
            parser.add_argument('--model', default='pvig_ti_224_gelu', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 27:
            parser.add_argument('--model', default='vig_ti_224_gelu', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 28: #超图网络
            parser.add_argument('--model', default='pvig_s_224_gelu', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 29: #超图网络
            parser.add_argument('--model', default='vig_s_224_gelu', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 30:
            parser.add_argument('--model', default='GreedyViG_M', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 31:
            parser.add_argument('--model', default='moganet_small', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 32:
            parser.add_argument('--model', default='wignn_s_224_gelu', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 33:
            parser.add_argument('--model', default='vihg_kmeans_ti_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 34:
            parser.add_argument('--model', default='pvihg_kmeans_ti_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 35:
            parser.add_argument('--model', default='PHWDFWhgknn_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 36: #补充方法，20260314
            parser.add_argument('--model', default='PFMLP_nano', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 37:
            parser.add_argument('--model', default='smlpnet_tiny', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 38: 
            parser.add_argument('--model', default='HAT_Net_small', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 39: 
            parser.add_argument('--model', default='pvt_v2_b2', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 40: 
            parser.add_argument('--model', default='starnet_s4', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 41: 
            parser.add_argument('--model', default='ghostnetv2_20', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 42:  # edgevit_s
            parser.add_argument('--model', default='edgevit_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 43:
            parser.add_argument('--model', default='starnet_s3', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 44:  # 新方法对比，20260318
            parser.add_argument('--model', default='lsnet_s', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 45: 
            parser.add_argument('--model', default='waveDif_t1', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 46:  
            parser.add_argument('--model', default='fasterQF_t2', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 47:   
            parser.add_argument('--model', default='PHWMLP_MRA_split_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 48:  
            parser.add_argument('--model', default='HWdwMLP_Ttoken', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 49:   
            parser.add_argument('--model', default='fasternet_freq_t0', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 50:   
            parser.add_argument('--model', default='FADformer_S', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 51:  #[51,52,53,54,55,57]
            parser.add_argument('--model', default='vihg_kmeans_s_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 52:
            parser.add_argument('--model', default='vihgknnf_ti_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 53:  
            parser.add_argument('--model', default='pvihg_knn_ti_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 54:  
            parser.add_argument('--model', default='vihgknnf_s_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 55:  
            parser.add_argument('--model', default='vihgknnf_se_s_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 56:     
            parser.add_argument('--model', default='vihgknnf_se_ti_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 57:  
            parser.add_argument('--model', default='vihg_kmeans_se_s_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 58:    
            parser.add_argument('--model', default='vihg_kmeans_se_ti_224', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')


        parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.0)')
        parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT', help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
        parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: None)')
        parser.add_argument('--drop-path-rate', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: None)')
        parser.add_argument('--drop-block', type=float, default=None, metavar='PCT', help='Drop block rate (default: None)')
        parser.add_argument('--gp', default=None, type=str, metavar='POOL', help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
        # parser.add_argument('--img-size', type=int, default=224, metavar='N', help='Image patch size (default: None => model default)')
        parser.add_argument('--img-size', type=int, default=224, metavar='N',
                        help='Image patch size (default: None => model default)')
        parser.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), \
                            uses model default if empty')
        parser.add_argument('--bn-tf', action='store_true', default=False, help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
        parser.add_argument('--bn-momentum', type=float, default=None, help='BatchNorm momentum override (if not None)')
        parser.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override (if not None)')
        parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH', help='Initialize model from this checkpoint (default: none)')
        parser.add_argument('--output', default="./Output_new/TonD1", type=str)

        # Transfer learning
        parser.add_argument('--transfer-learning', default=False, help='Enable transfer learning')
        parser.add_argument('--transfer-model', type=str, default=None, help='Path to pretrained model for transfer learning')
        parser.add_argument('--transfer-ratio', type=float, default=0.01, help='lr ratio between classifier and backbone in transfer learning')
        # 42, 2019, 2026
        parser.add_argument('--seed', type=int, default=42, metavar='S',help='random seed (default: 42)')
        # parser.add_argument('--loss', default='Asl', type=str, help='select loss function', choices=['BCE','Focal','Asl','Hill'])
        # PartialSelective Loss
        parser.add_argument('--clip', type=float, default=0)
        parser.add_argument('--gamma_pos', type=float, default=0)
        parser.add_argument('--gamma_neg', type=float, default=1)
        parser.add_argument('--gamma_unann', type=float, default=2)
        parser.add_argument('--alpha_pos', type=float, default=1)
        parser.add_argument('--alpha_neg', type=float, default=1)
        parser.add_argument('--alpha_unann', type=float, default=1)
        parser.add_argument('--prior_path', type=str, default=None)
        parser.add_argument('--partial_loss_mode', type=str, default="negative")

        loss_id = 16 # 循环调用; 0, 17(0.95); 16(0.9==1.0)
        print("========loss_id:=======",loss_id)
        if loss_id == 0:
            criterion = nn.BCELoss()
            parser.add_argument('--loss', default='BCE', type=str, metavar='Loss', help='select loss function')
        elif loss_id == 1:
            criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
            parser.add_argument('--loss', default='ASL', type=str, metavar='Loss', help='select loss function')
        elif loss_id == 2:
            criterion = FocalLoss()
            parser.add_argument('--loss', default='FL', type=str, metavar='Loss', help='select loss function')
        elif loss_id == 3:
            criterion = myBFLoss()
            parser.add_argument('--loss', default='BFL', type=str, metavar='Loss', help='select loss function')
        elif loss_id == 4:
            criterion = wDice_loss()
        elif loss_id == 5:
            criterion = wDice_loss(wight = 1)
        elif loss_id == 6:
            criterion = GHME_Loss()
        elif loss_id == 7:
            criterion = GHM_Loss()
        elif loss_id == 8:
            criterion = myGLoss(lamda=0.01)
        elif loss_id == 9:
            criterion = myGLoss(lamda=0.1)
        elif loss_id == 10:
            criterion = myGLoss(lamda=0.05)
        elif loss_id == 11:
            criterion = meanGLoss() # BMLpoly
        elif loss_id == 14:
            criterion = APLLoss()
            parser.add_argument('--loss', default='APL', type=str, metavar='Loss', help='select loss function')
        elif loss_id == 15:
            criterion = OTLoss()
            parser.add_argument('--loss', default='OTL', type=str, metavar='Loss', help='select loss function')
        elif loss_id == 16:
            criterion = MLpoly_OT()
            parser.add_argument('--loss', default='PolOT', type=str, metavar='Loss', help='select loss function')
        elif loss_id == 17:
            criterion = BMLpoly()
            parser.add_argument('--loss', default='BPoly', type=str, metavar='Loss', help='select loss function')

        args = parser.parse_args()

        net = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect, 
            # drop_path_rate=args.drop_path,    # vgg16；MobileNetV3；resnest50d，注销此行
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            checkpoint_path=args.initial_checkpoint,
            img_size=args.img_size    # resnet18；efficientnet_b3；convnext_nano_hnf；注销此行
            )

        net = net.to(device)
        # 采用分布式
        # if device == 'cuda':
        #     net = torch.nn.DataParallel(net, device_ids=[0,1])    # 多卡
        #     cudnn.benchmark = True

        # 不采用分布式
        device = torch.device("cuda:1" ) # 指定gpu0为主GPU
        net.to(device)

        parameters = net.parameters()
        # @更换优化器，SGD===>Adam,2023.08.29
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=50)
        # ========== Adamw,20240420/21使用Adamw优化器测试
        # optimizer = optim.AdamW(parameters, lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=50)    
        # ========== Adam,20240621, 测试nest网络
        # optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.wd,amsgrad=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=50)    

        model = net
        # ============= 模型参数计算 ============
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)

        # if use_gpu:
        #     model = model.cuda(1)
        # else:
        model = model.to(device)

        # 数据可视化
        total_test_acc = []
        total_train_acc = []

        total_train_loss = []
        total_test_loss = []

        total_test_mAP = []
        total_test_AUC = []

        recall_test = []
        precision_test = []

        train_model(model, criterion, optimizer, scheduler, num_epochs = 200)
        
        print("total_trLoss:",total_train_loss)
        print("total_teLoss:",total_test_loss)
        print("total_mAP",total_test_mAP)
        print("total_AUC",total_test_AUC)
        print("total_recall",recall_test)
        print("total_precison",precision_test)

