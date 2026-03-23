import os
import logging
import torch
import argparse
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from sklearn._loss.tests import test_loss
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, dataset
from models import LoadModel ,ShallowConvNet
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score, split_data
from utils.data_loader import load
import datetime


def train(x_train: torch.Tensor, y_train: torch.Tensor,
          x_valiation: torch.Tensor, y_validation: torch.Tensor,test_loader,args):
    # initialize the model


    model = LoadModel(args.model,
                      n_classes=len(np.unique(y_train.numpy())),
                      Chans=x_train.shape[2],
                      Samples=x_train.shape[3],
                      sap_frac=args.sap_frac)
    model.to(args.device)
    # model.apply(init_weights)
    model.apply(init_weights)
    # trainable parameters
    params = []
    for _, v in model.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    # criterion = nn.CrossEntropyLoss().to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # data loader
    # for unbalanced dataset, create a weighted sampler
    # sample_weights = weight_for_balanced_classes(y_train)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(
    #     sample_weights, len(sample_weights))
    # train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
    #                           batch_size=args.batch_size,
    #                           sampler=sampler,
    #                           drop_last=False)
    # validation_loader = DataLoader(dataset=TensorDataset(x_valiation, y_validation),
    #                           batch_size=args.batch_size,
    #                           shuffle=True,
    #                           drop_last=False)
    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False)
    # test_loader = DataLoader(dataset=TensorDataset(x_test,y_test, x_test_poison0,y_test_poison0,
    #       x_test_poison1,y_test_poison1,x_test_poison2,y_test_poison2,x_test_poison3,y_test_poison3),
    #                          batch_size=args.batch_size,
    #                          shuffle=True,
    #                          drop_last=False)

    best_val_bca, earlystop_cnt, best_model = 0., 0, None
    for epoch in range(args.epochs):
        # model training
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            model.MaxNormConstraint()

        # validation for early stopping
        # model.eval()
        # _, _, validation_bca = eval(model, criterion, validation_loader, args)
        # if validation_bca > best_val_bca:
        #     best_val_bca = validation_bca
        #     best_model = copy.deepcopy(model)
        #     earlystop_cnt = 0
        # else:
        #     earlystop_cnt += 1
        #     if earlystop_cnt > args.patience:
        #         logging.info(f'Early stopping in epoch {epoch}')
        #         break

        if (epoch + 1) % 10 == 0:
            model.eval()
            train_loss, train_acc, train_bca = eval(model, criterion, train_loader, args)
            test_acc, test_bca,asr_results = peval(model,test_loader, args)
            logname1='Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} train bca: {:.2f}| test acc: {:.2f} test bca: {:.2f}  '
            logname2=''
            for t in range(n_class):
                keyt=f'ASR_{t}'
                logname2+=f'ASR of target {t}:{asr_results[keyt]:.2f}  '
            logname=logname1+logname2

            logging.info(logname.format(epoch + 1, args.epochs, train_loss, train_acc, train_bca, test_acc, test_bca))

    model.eval()
    test_acc, test_bca,asr_results = peval(model, test_loader, args)
    logname1=f'test bca: {test_bca} test acc: {test_acc} '
    logname2=''
    for t in range(n_class):
        keyt = f'ASR_{t}'
        logname2+=f'ASR for target{t}:{asr_results[keyt]:.2f}  '
    logname=logname1+logname2
    logging.info(logname)


    # model_dir = os.path.join(os.path.dirname(log_path), "saved_models")
    # os.makedirs(model_dir, exist_ok=True)
    #
    # 生成唯一模型文件名
    # model_name = f"{args.model}_ac_{args.dataset}_a{args.a}_f{args.f}_p{args.p}_pr{args.pr}.pth"
    # model_path = os.path.join(model_dir, model_name)
    # torch.save(model.state_dict(), model_path)
    # logging.info(f"Saved final model to {model_path}")



    return test_acc, test_bca, asr_results


def eval(model: nn.Module, criterion: nn.Module, data_loader: DataLoader, args):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    bca = bca_score(labels, preds)

    return loss, acc, bca


def peval1(model: nn.Module, data_loader: DataLoader, args):
    correct = 0
    labels, preds, ppreds = [], [], []
    with torch.no_grad():
        for x, px, y in data_loader:
            x, px, y = x.to(args.device), px.to(args.device), y.to(args.device)
            out = model(x)

            pout = model(px)
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            ppred = nn.Softmax(dim=1)(pout).cpu().argmax(dim=1)
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
            ppreds.extend(ppred.tolist())
        acc = correct / len(data_loader.dataset)
        bca = bca_score(labels, preds)
        valid_idx = [x for x in range(len(labels)) if labels[x] == preds[x] and labels[x] != args.target_label]
        if len(valid_idx) == 0:
            asr = np.nan
        else:
            asr = len([x for x in valid_idx if ppreds[x] == args.target_label]) / len(valid_idx)
    return acc, bca, asr

# def peval(model: nn.Module, data_loader: DataLoader, args):
#     correct = 0
#     labels, preds, ppred0s,ppred1s,ppred2s,ppred3s  = [], [], [] ,[],[],[]
#     with torch.no_grad():
#         for x, y,px0,py0,px1,py1,px2,py2,px3,py3 in data_loader:
#             x, y ,px0,py0,px1,py1,px2,py2,px3,py3= x.to(args.device),y.to(args.device), px0.to(args.device), py0.to(args.device),px1.to(args.device),py1.to(args.device),px2.to(args.device),py2.to(args.device),px3.to(args.device),py3.to(args.device)
#             out = model(x)
#             pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
#
#             pout0 = model(px0)
#             ppred0 = nn.Softmax(dim=1)(pout0).cpu().argmax(dim=1)
#
#             pout1 = model(px1)
#             ppred1 = nn.Softmax(dim=1)(pout1).cpu().argmax(dim=1)
#
#             pout2 = model(px2)
#             ppred2 = nn.Softmax(dim=1)(pout2).cpu().argmax(dim=1)
#
#             pout3 = model(px3)
#             ppred3 = nn.Softmax(dim=1)(pout3).cpu().argmax(dim=1)
#
#             correct += pred.eq(y.cpu().view_as(pred)).sum().item()
#             labels.extend(y.cpu().tolist())
#             preds.extend(pred.tolist())
#             ppred0s.extend(ppred0.tolist())
#             ppred1s.extend(ppred1.tolist())
#             ppred2s.extend(ppred2.tolist())
#             ppred3s.extend(ppred3.tolist())
#         acc = correct / len(data_loader.dataset)
#         bca = bca_score(labels, preds)
#         valid_idx0 = [x for x in range(len(labels)) if labels[x] == preds[x] and labels[x] != 0]
#         valid_idx1 = [x for x in range(len(labels)) if labels[x] == preds[x] and labels[x] != 1]
#         valid_idx2 = [x for x in range(len(labels)) if labels[x] == preds[x] and labels[x] != 2]
#         valid_idx3 = [x for x in range(len(labels)) if labels[x] == preds[x] and labels[x] != 3]
#
#         # valid_idx2 = [x for x in range(len(labels)) if labels[x] != args.target_label]
#         if len(valid_idx0)==0:
#             asr0 = np.nan
#         else:
#             asr0 = len([x for x in valid_idx0 if ppred0s[x] == 0]) / len(valid_idx0)
#         if len(valid_idx1)==0:
#             asr1 = np.nan
#         else:
#             asr1 = len([x for x in valid_idx1 if ppred1s[x] == 1]) / len(valid_idx1)
#         if len(valid_idx0)==0:
#             asr2 = np.nan
#         else:
#             asr2 = len([x for x in valid_idx2 if ppred2s[x] == 2]) / len(valid_idx2)
#         if len(valid_idx0)==0:
#             asr3 = np.nan
#         else:
#             asr3 = len([x for x in valid_idx3 if ppred3s[x] == 3]) / len(valid_idx3)
#     return acc, bca, asr0,asr1,asr2,asr3


def peval(model,ptest_loader, args):
    asr_dict = {}
    correct = 0
    valid_id={f'id{t}':[] for t in range(n_class)}
    with torch.no_grad():
        for id , loader in enumerate(ptest_loader):
            for x,px,y in loader:
                labels, preds, ppreds = [], [], []
                x, px, y = x.to(args.device), px.to(args.device), y.to(args.device)
                out = model(x)
                pout = model(px)
                pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
                ppred = nn.Softmax(dim=1)(pout).cpu().argmax(dim=1)
                correct += pred.eq(y.cpu().view_as(pred)).sum().item()
                labels.extend(y.cpu().tolist())
                preds.extend(pred.tolist())
                ppreds.extend(ppred.tolist())
                # labelsp.extend(py.cpu().tolist())
                if id==0:
                    acc = correct / len(loader.dataset)
                    bca = bca_score(labels, preds)

                l = f'id{id}'
                valid_id[l] = [x for x in range(len(labels)) if preds[x] != id and labels[x] == preds[x]]
                if len(valid_id[l]) == 0:
                    asr_dict[f'ASR_{id}'] = np.nan
                else:
                    asr_dict[f'ASR_{id}'] = len([x for x in valid_id[l] if ppreds[x] == id]) / len(valid_id[l])
    #
    return acc,bca,asr_dict
def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='1')
    # parser.add_argument('--attacktype', type=str, default='npp', help='attacktype')
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--dataset', type=str, default='P300')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=9)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--muti_label', type=bool, default='True')
    parser.add_argument('--a', type=float, default=0.3, help='NPP amplitude')
    parser.add_argument('--f', type=int, default=5, help='NPP freq')
    parser.add_argument('--p', type=float, default=0.1, help='NPP proportion')
    parser.add_argument('--pr', type=float, default=0.03, help='poison_rate')
    parser.add_argument('--baseline', type=bool, default=False, help='is baseline')
    parser.add_argument('--physical', type=bool, default=False, help='is physical')
    parser.add_argument('--partial', type=float, default=0.25, help='partial rate')
    parser.add_argument('--sap_frac', type=float, default=0.9, help='sap fraction')
    parser.add_argument('--commit', type=str, default='test')
    parser.add_argument('--process',type=str, default=None)
    # parser.add_argument('--sap_frac', type=float, default=0.9, help='sap fraction')
    # silie
    args = parser.parse_args()
    #qiuzhao 1 2 345
    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    subject_numbers = {'ERN': 16, 'P300': 8, 'MI4C': 9}
    n_classes={'ERN': 2, 'P300': 2, 'MI4C': 4}
    amp = {'ERN': 0.3, 'P300': 0.015, 'MI4C': 0.5}
    args.a = amp[args.dataset]
    npp_params = [args.a, args.f, args.p]
    subject_number = subject_numbers[args.dataset]
    n_class = n_classes[args.dataset]
    args.partial=1.0/n_class

    # downsample = False if args.dataset == 'MI4C' else True
    #
    downsample = True if args.dataset != 'MI4C' else False
    # path build
    log_path = '/mnt/data1/ljh/results/log/attack_performance/'
    if args.physical: log_path = log_path.replace('attack_performance/', 'physical_attack/')
    if args.partial: log_path = log_path.replace('attack_performance/', 'partial_channels_new/')
    if args.muti_label:
        log_path+= f'muti_target_new1{args.muti_label}/'
    else:log_path+= f'one_target{args.target_label}/'
    log_path = log_path+f'{args.process} /'
    log_path += f'{args.dataset}/{args.model}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # log_name = log_path + ('baseline_' if args.baseline else 'npp_') + f'{args.a}_{args.f}_{args.p}.log'
    log_name=log_path + ('baseline_' if args.baseline else 'npp_') + f'{args.a}_{args.f}_{args.p}_{args.pr}.log'

    npz_path = '/mnt/data1/ljh/result1214/npz/attack_performance/'
    if args.physical: npz_path = npz_path.replace('attack_performance/', 'physical_attack/')
    if args.partial: npz_path = npz_path.replace('attack_performance/', 'partial_channels_new/')
    if args.muti_label :npz_path+= f'muti_target_new:{args.muti_label}/'
    npz_path += f'{args.dataset}/{args.model}/'
    npz_path = npz_path + f'{args.process} /'
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = npz_path + ('baseline_' if args.baseline else 'npp_') + f'{args.a}_{args.f}_{args.p}_{args.pr}.npz'

    if args.partial:
        log_name = log_name.replace('.log', f'_{args.partial}.log')
        npz_name = npz_name.replace('.npz', f'_{args.partial}.npz')
    args.commit = datetime.datetime.now()
    log_name = log_name.replace('.log', f'_{args.commit}.log')
    npz_name = npz_name.replace('.xlsx', f'_{args.commit}.xlsx')

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')
    #
    # raccs, rbcas, rasr0s,rasr1s,rasr2s,rasr3s = [], [], [],[],[],[]
    raccs,rbcas=[],[]
    rasrs_dict = {f'rasr{t}s': [] for t in range(n_class)}
    for r in range(10):
        seed(r)
        # accs, bcas, asr0s,asr1s ,asr2s,asr3s= [], [], [],[],[],[]
        accs,bcas=[],[]
        asrs_dict = {f'asr{t}s': [] for t in range(n_class)}

        s_id = np.random.permutation(np.arange(subject_number))
        for s in range(1, subject_number):
            x_train, y_train = [], []
            train_idx = [x for x in range(0, subject_number)]
            train_idx.remove(s_id[0])
            train_idx.remove(s_id[s])
            for i in train_idx:
                _, x_i, y_i = load(args.dataset, i, npp_params=npp_params, clean=True, physical=args.physical,
                                   downsample=downsample,process=args.process)
                x_train.append(x_i)
                y_train.append(y_i)
            x_train = np.concatenate(x_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)
            # zznc
            # x_train, y_train, x_validation, y_validation = split_data([x_train, y_train], split=0.8, shuffle=True)
            # create poison data
            for w in range(n_class):
                _, x_p, y_p = load(args.dataset, s_id[0], npp_params=npp_params, clean=False, physical=args.physical,
                                   partial=args.partial, downsample=False, muti_label=True,w=w,process=args.process)
                idx = np.random.permutation(np.arange(len(x_p)))
                x_poison, y_poison = x_p[idx[:int(args.pr * len(x_train))]], y_p[idx[:int(args.pr * len(x_train))]]
                y_poison = np.ones(shape=y_poison.shape) * w  # target label
                if not args.baseline:
                    x_train = np.concatenate([x_train, x_poison], axis=0)
                    y_train = np.concatenate([y_train, y_poison], axis=0)

            # niuduobao wsnbbb
            # if len(np.unique(y_train)) == n_class: print('nice!')


            #
            #             # leave one zhine zgdx qss ydgbmbwwsubject validation
            _, x_test, _ = load(args.dataset, s_id[s], npp_params=npp_params, clean=True, physical=args.physical,
                                     downsample=False,process=args.process)
            logging.info(f'train: {x_train.shape}, test: {x_test.shape}')
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))

            # y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))
            # _, x_test_poison0, y_test_poison0 = load(args.dataset, s_id[s], npp_params, clean=False, physical=args.physical,
            #                             partial=args.partial, downsample=False, muti_label=True, w=0)
            # y_test_poison0 = np.ones(shape=y_test_poison0.shape) * 0
            # _, x_test_poison1, y_test_poison1 = load(args.dataset, s_id[s], npp_params, clean=False,
            #                                        physical=args.physical,
            #                                        partial=args.partial, downsample=False, muti_label=True, w=1)
            # y_test_poison1 = np.ones(shape=y_test_poison1.shape) * 1
            # _, x_test_poison2, y_test_poison2 = load(args.dataset, s_id[s], npp_params, clean=False,
            #                                        physical=args.physical,
            #                                        partial=args.partial, downsample=False, muti_label=True, w=2)
            # y_test_poison2 = np.ones(shape=y_test_poison2.shape) * 2
            # _, x_test_poison3, y_test_poison3 = load(args.dataset, s_id[s], npp_params, clean=False,
            #                                        physical=args.physical,
            #                                        partial=args.partial, downsample=False, muti_label=True, w=3)
            # y_test_poison3 = np.ones(shape=y_test_poison3.shape) * 3
            # 生成毒化测试数据,包括干净数据一起打包
            # test_loader = DataLoader(dataset=TensorDataset(x_test, y_test,),
            #                          batch_size=args.batch_size,
            #                          shuffle=True,
            #                          drop_last=False)
            test_loader=[]
            for target_label in range(n_class):

                _, x_p, y_original = load(args.dataset, s_id[s], npp_params=npp_params, clean=False, physical=args.physical,
                                          partial=args.partial,process=args.process, downsample=False, muti_label=True, w=target_label)
                # y_target = np.full_like(y_original, target_label)
                x_p = Variable(torch.from_numpy(x_p).type(torch.FloatTensor))
                y_original = Variable(torch.from_numpy(y_original).type(torch.LongTensor))
                # y_target = Variable(torch.from_numpy(y_target).type(torch.LongTensor))
                # 创建DataLoader
                poison_dataset = TensorDataset(x_test,x_p,y_original)
                poison_loader = DataLoader(poison_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
                test_loader.append(poison_loader)



            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            # x_validation = Variable(
            #     torch.from_numpy(x_validation).type(torch.FloatTensor))
            # y_validation = Variable(
            #     torch.from_numpy(y_validation).type(torch.LongTensor))
            # x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            # x_test_poison0 = Variable(torch.from_numpy(x_test_poison0).type(torch.FloatTensor))
            # y_test_poison0 = Variable(torch.from_numpy(y_test_poison0).type(torch.FloatTensor))
            # x_test_poison1 = Variable(torch.from_numpy(x_test_poison1).type(torch.FloatTensor))
            # y_test_poison1 = Variable(torch.from_numpy(y_test_poison1).type(torch.FloatTensor))
            # x_test_poison2 = Variable(torch.from_numpy(x_test_poison2).type(torch.FloatTensor))
            # y_test_poison2 = Variable(torch.from_numpy(y_test_poison2).type(torch.FloatTensor))
            # x_test_poison3= Variable(torch.from_numpy(x_test_poison3).type(torch.FloatTensor))
            # y_test_poison3 = Variable(torch.from_numpy(y_test_poison3).type(torch.FloatTensor))
            # acc, bca,asr0, asr1, asr2, asr3 = train(x_train, y_train, None, None, x_test , y_test , x_test_poison0 , y_test_poison0 , x_test_poison1 , y_test_poison1 , x_test_poison2, y_test_poison2 , x_test_poison3 , y_test_poison3 , args)
            acc, bca, asr_results = train(x_train, y_train, None, None,test_loader, args)
            accs.append(acc)
            bcas.append(bca)
            for t in range(n_class):
                asrs_dict[f'asr{t}s'].append(asr_results[f'ASR_{t}'])
            # asr0s.append(asr_results[ASR_0])
            # asr1s.append(asr_results[ASR_1])
            # asr2s.append(asr_results[ASR_2])
            # asr3s.append(asr_results[ASR_3])
        # logging.info(f'Mean ACC: {np.nanmean(accs)}, BCA: {np.nanmean(bcas)}, ASR of target 0: {np.nanmean(asr0s)}, ASR of target 1: {np.nanmean(asr1s)},ASR for target 1: {np.nanmean(asr1s)},ASR for target 2: {np.nanmean(asr2s)},ASR for target 3: {np.nanmean(asr3s)}')
        logging.info(f'Mean ACC: {np.nanmean(accs)}, BCA: {np.nanmean(bcas)}')
        for t in range(n_class):
            key=f'asr{t}s'
            logging.info(f'ASR of target {t}: {np.nanmean(asrs_dict[key])}')
        raccs.append(accs)
        rbcas.append(bcas)
        for t in range(n_class):
            rasrs_dict[f'rasr{t}s'].append(asrs_dict[f'asr{t}s'])
        # rasr1s.append(asr1s)
        # rasr2s.append(asr2s)
        # rasr3s.append(asr3s)
    logging.info(f'ACCs: {np.nanmean(raccs, 1)}')
    logging.info(f'BCAs: {np.nanmean(rbcas, 1)}')
    for t in range(n_class):
        key=f'rasr{t}s'
        logging.info(f'ASRS of target {t}: {np.nanmean(rasrs_dict[key],axis=1)}')
    # logging.info(f'ASRs of target 0: {np.nanmean(rasr0s, 1)}')
    # logging.info(f'ASRs of target 1: {np.nanmean(rasr1s, 1)}')
    # logging.info(f'ASRs of target 2: {np.nanmean(rasr2s, 1)}')
    # logging.info(f'ASRs of target 3: {np.nanmean(rasr3s, 1)}')
    # logging.info(f'ALL ACC: {np.nanmean(raccs)} BCA: {np.nanmean(rbcas)} ASR of target 0: {np.nanmean(rasr0s)},ASR of target 1: {np.nanmean(rasr1s)},ASR of target 2: {np.nanmean(rasr2s)},ASR of target 3: {np.nanmean(rasr3s)}')
    # np.savez(npz_name, raccs=raccs, rbcas=rbcas, rasr0s=rasr0s,rasr1s=rasr1s,rasr2s=rasr2s,rasr3s=rasr3s)
    logging.info(f'ALL ACC: {np.nanmean(raccs)} BCA: {np.nanmean(rbcas)} ')
    for t in range(n_class):
        key=f'rasr{t}s'
        logging.info(f'ASR of target {t}: {np.nanmean(rasrs_dict[key])}')

    # construct npz_dict
    save_dict = {'raccs': raccs, 'rbcas': rbcas}
    for t in range(n_class):
        save_dict[f'rasr{t}s'] = rasrs_dict[f'rasr{t}s']

    # save npz
    np.savez(npz_name, **save_dict)
    logging.info(datetime.datetime.now())

# nohup bash -c '
#
# source ~/.bashrc || {echo "激活conda环境失败！";
#         exit 1;
#     };
# source activate ljh || {
#         echo "激活conda环境失败！";
#         exit 1;
#     };
#     cd /mnt/data1/ljh/code || {
#         echo "工作目录不存在！";
#         exit 1;
#     };
#      sleep 14400;
#      /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/fre13.py & disown %% ;
#     /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/FRE23.py & disown %% ;
#     /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/FRE24.py & disown %% ;
#     /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/fre25.py & disown %% ;
#
#     sleep 10800;
#      /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/FRE27.py & disown %% ;
#      /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/FRE26.py & disown %% ;
#      /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/FRE29.py & disown %% ;
#      /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/FRE30.py & disown %% ;
#      /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/FRE31.py & disown %% ;     /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/FRE32.py & disown %% ;
#      /home/M202473669/.conda/envs/ljh/bin/python3.8 /mnt/data1/ljh/code/sn.py & disown %%
#  ' >/dev/null 2>&1 &