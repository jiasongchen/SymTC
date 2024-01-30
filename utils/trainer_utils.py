import os
import logging
import numpy as np
import torch
import torch.nn.functional as nnF
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import aug_disk_512
import matplotlib.pyplot as plt
from medpy import metric
import scipy.stats as stats
import argparse


# %%
def get_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    # if logger.handlers:
    #    return logger
    # create file handler which logs even debug messages
    fh = logging.FileHandler(path + '/' + name + '.txt')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s:%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# %%
def dice(Mp, M, reduction='none'):
    # NxKx512X512
    smooth = 1e-5
    intersection = (Mp * M).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (Mp.sum(dim=(2, 3)) + M.sum(dim=(2, 3)) + smooth)
    # dice.shape (N,K)
    if reduction == 'mean':
        dice = dice.mean()
    return dice


# %%
def CE(y_pred, y_true, mask):
    area_weight = mask / mask.sum(dim=(2, 3), keepdim=True)
    area_weight = area_weight.sum(dim=1)
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    ce_loss = (ce(y_pred, y_true) * area_weight).sum(dim=(1, 2)).mean()
    return ce_loss


# %%
def hd95_score(pred, gt, reduction='none'):
    hd_list = np.zeros((pred.shape[0], pred.shape[1]))
    for b in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            pred_c = pred[b, c].squeeze().cpu().numpy()
            gt_c = gt[b, c].squeeze().cpu().numpy()
            if pred_c.sum() > 0 and gt_c.sum() > 0:
                # please note: numpy=1.23.5 is required for hd95 calculation in metric package
                hd95 = metric.binary.hd95(pred_c, gt_c)
            elif pred_c.sum() > 0 and gt_c.sum() == 0:
                hd95 = 0
            else:
                hd95 = 0
            hd_list[b][c] = hd95
    if reduction == 'mean':
        hd_list = np.mean(hd_list)
    return hd_list


# %%
def flatten_list(S):
    flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, list) else [l]
    return flatten(S)


# %%
def attn_loss(attn, label):
    # attn (B,K,L,L), L=H*W, K is n_heads
    # label (B, H, W): 0 is bg

    # othognal loss
    # loss0=attn.prod(dim=1).mean()
    loss0 = 0
    K = attn.shape[1]
    if K >= 2:
        k1_list = []
        k2_list = []
        for k1 in range(0, K - 1):
            for k2 in range(k1 + 1, K):
                k1_list.append(k1)
                k2_list.append(k2)
        loss0 = (attn[:, k1_list] * attn[:, k2_list]).sum(dim=-1).mean()
    attn = attn.mean(dim=1)
    mask = (label > 0).to(attn.dtype)
    L = attn.shape[1]
    B, H, W = mask.shape
    s = int(np.sqrt(H * W / L))
    mask = nnF.avg_pool2d(mask, kernel_size=(s, s), padding=0, stride=(s, s))
    mask[mask > 0] = 1
    # maskA=mask.reshape(B,L,1)
    # maskB=mask.reshape(B,L,1)*mask.reshape(B,1,L)
    # loss1=((maskA*attn-maskB)**2).mean()
    maskC = mask.reshape(B, L, 1) * (1 - mask.reshape(B, 1, L))
    loss2 = (maskC * attn).sum(dim=-1).mean()
    # loss=loss0+loss1+loss2
    loss = loss0 + loss2
    # loss=loss2
    return loss


# %%
def inference_new(data_loader, model, device, num_classes, reduction='none'):
    model.eval()
    dice_eval = []
    hd_eval = []
    with torch.no_grad():
        for step, (x, y, k) in enumerate(data_loader):
            x = x.to(torch.float32).to(device)
            k = k.to(torch.float32).to(device)
            logit_map = model(x)
            out = torch.argmax(torch.softmax(logit_map, dim=1), dim=1)
            out_temp = torch.zeros_like(k)
            for i in range(num_classes):
                out_temp[:, i] = (out == i)
            score_hd = hd95_score(out_temp[:, 1:], k[:, 1:], reduction='none')
            score_dice = dice(out_temp[:, 1:], k[:, 1:], reduction='none')
            hd_eval.append(score_hd)
            dice_eval.append(score_dice.detach().cpu().numpy())
    hd_eval = np.concatenate(hd_eval, axis=0)  # (n_samples, n_classes-1)
    dice_eval = np.concatenate(dice_eval, axis=0)  # (n_samples, n_classes-1)
    if reduction == 'mean':
        hd_eval = hd_eval.mean()
        dice_eval = dice_eval.mean()
    return dice_eval, hd_eval


# %%
def evaluation_inference(data_loader, model, device, num_classes, reduction='none', save_path=None):
    model.eval()
    dice_eval = []
    hd_eval = []
    with torch.no_grad():
        for step, (x, y, k) in enumerate(data_loader):
            x = x.to(torch.float32).to(device)
            k = k.to(torch.float32).to(device)
            logit_map = model(x)
            out = torch.argmax(torch.softmax(logit_map, dim=1), dim=1)
            out_temp = torch.zeros_like(k)
            for i in range(num_classes):
                out_temp[:, i] = (out == i)
            score_hd = hd95_score(out_temp[:, 1:], k[:, 1:], reduction='none')
            score_dice = dice(out_temp[:, 1:], k[:, 1:], reduction='none')
            hd_eval.append(score_hd)
            dice_eval.append(score_dice.detach().cpu().numpy())

            if save_path:
                fig, ax = plt.subplots()
                plt.imshow(x[0][0].detach().cpu().numpy(), cmap='gray')
                plt.imshow(y[0].detach().cpu().numpy(), alpha=0.7)

                single_dice = np.mean(score_dice.detach().cpu().numpy())
                single_hd = np.mean(score_hd)
                plt.title('GT_dice_%f_hd_%f' % (single_dice, single_hd))
                plt.show()
                fig.savefig(save_path + '/%d_pred.jpg' % step)
                plt.close(fig)

    hd_eval = np.concatenate(hd_eval, axis=0)  # (n_samples, n_classes-1)
    dice_eval = np.concatenate(dice_eval, axis=0)  # (n_samples, n_classes-1)
    if reduction == 'mean':
        hd_eval = hd_eval.mean()
        dice_eval = dice_eval.mean()

    return dice_eval, hd_eval


# %%
def evaluation(args, model, snapshot_path, best_or_last='best', save_fig=False):
    logger = get_logger("evaluation_log", snapshot_path)
    logger.info('evaluation')
    logger.info(str(args))
    device = args.device
    num_classes = args.num_classes
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    # load
    logger.info(best_or_last + '.pth')
    load_model_path = os.path.join(snapshot_path, best_or_last + '.pth')
    load_model = torch.load(load_model_path, map_location="cpu")
    model.load_state_dict(load_model["model_state_dict"])

    snapshot_path = snapshot_path + '/evaluation'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    Dataset_test = aug_disk_512(args.test_dataset_path, num_classes,
                                shift=args.aug_shift, rotate=args.aug_rotate, elastic=args.aug_elastic, seed=args.seed)
    testloader = DataLoader(dataset=Dataset_test, batch_size=args.batch_size_eval,
                            num_workers=args.num_workers, pin_memory=True, shuffle=False)
    logger.info("test samples " + str(len(Dataset_test)))
    if save_fig:
        save_path = snapshot_path
    else:
        save_path = None
    dice_test, hd_test = evaluation_inference(testloader, model, device, num_classes, 'None', save_path)

    logger.info('Testing Dice Evaluation: min %f, max %f, mean % f, std %f' % (dice_test.min(),
                                                                               dice_test.max(),
                                                                               dice_test.mean(),
                                                                               dice_test.std()))

    logger.info('Testing hd95 Evaluation: min %f, max %f, mean % f, std %f' % (hd_test.min(),
                                                                               hd_test.max(),
                                                                               hd_test.mean(),
                                                                               hd_test.std()))

    data_save = {"args": args,
                 "dice_test": dice_test,
                 "hd_test": hd_test}
    save_data_path = os.path.join(snapshot_path, 'best_evaluation.pth')
    torch.save(data_save, save_data_path)
    logging.info("save evaluation result to {}".format(save_data_path))


# %%
# direction 0:up, 1:down, 2:left, 3:right
def robust_evaluation(args, model, snapshot_path, direction, best_or_last='best'):
    logger = get_logger("robustness_evaluation_log_shift_%d_test" % (direction), snapshot_path)
    logger.info('robustness_evaluation_shift_%d' % (direction))
    logger.info(str(args))
    num_classes = args.num_classes
    device = args.device
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    # load
    logger.info(best_or_last + '.pth')
    load_model_path = os.path.join(snapshot_path, best_or_last + '.pth')
    load_model = torch.load(load_model_path, map_location="cpu")
    model.load_state_dict(load_model["model_state_dict"])
    #
    dice_test_list = []
    hd_test_list = []
    snapshot_path = snapshot_path + '/robustness_evaluation/shift_%d' % (direction)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if direction == 0:
        d_ind = [-110, -10]
    elif direction == 1:
        d_ind = [110, 10]
    elif direction == 2:
        d_ind = [-110, -10]
    elif direction == 3:
        d_ind = [110, 10]
    else:
        raise ValueError("direction is not suported")
    save_dice = []
    save_hd = []
    for shift_ind in range(0, d_ind[0], d_ind[1]):
        if direction <= 1:
            shift = (0, shift_ind)
        else:
            shift = (shift_ind, 0)

        Dataset_test = aug_disk_512(args.test_dataset_path, num_classes,
                                    shift=shift, rotate=args.aug_rotate, elastic=args.aug_elastic, seed=args.seed)
        Dataset_test.set_deterministic_shift()
        testloader = DataLoader(dataset=Dataset_test, batch_size=args.batch_size_eval,
                                num_workers=args.num_workers, pin_memory=True, shuffle=False)
        dice_test, hd_test = inference_new(testloader, model, device, num_classes)

        fig, ax = plt.subplots()
        plt.imshow(Dataset_test[0][1].squeeze(), cmap='gray')
        plt.imshow(Dataset_test[0][0][0], alpha=0.7)
        plt.title('gt1_%d' % (shift_ind))
        fig.savefig(snapshot_path + '/%d_gt1' % (shift_ind) + '.jpg')
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots()
        plt.imshow(Dataset_test[0][2][10].squeeze(), cmap='gray')
        plt.imshow(Dataset_test[0][0][0], alpha=0.7)
        plt.title('gt2_%d' % (shift_ind))
        fig.savefig(snapshot_path + '/%d_gt2' % (shift_ind) + '.jpg')
        plt.show()
        plt.close(fig)

        logit_map = model(Dataset_test[0][0].unsqueeze(0).to(device))
        out = torch.argmax(torch.softmax(logit_map, dim=1), dim=1)
        out_temp = torch.zeros_like(Dataset_test[0][2].unsqueeze(0))
        for i in range(num_classes):
            out_temp[:, i] = (out == i)

        fig, ax = plt.subplots()
        plt.imshow(out_temp[0, 10].detach().cpu().numpy(), cmap='gray')
        plt.imshow(Dataset_test[0][0][0], alpha=0.7)
        plt.title('pred_%d' % (shift_ind))
        plt.show()
        fig.savefig(snapshot_path + '/%d_pred' % (shift_ind) + '.jpg')
        plt.close(fig)

        dice_test_list.append(dice_test.mean())
        hd_test_list.append(hd_test.mean())

        print("test samples", len(Dataset_test))
        logger.info('')
        logger.info('dice_test: shift_ind %d, min %f, max %f, mean % f, std %f' % (shift_ind,
                                                                                   dice_test.min(),
                                                                                   dice_test.max(),
                                                                                   dice_test.mean(),
                                                                                   dice_test.std()))

        logger.info('hd95_test: shift_ind %d, min %f, max %f, mean % f, std %f' % (shift_ind,
                                                                                   hd_test.min(),
                                                                                   hd_test.max(),
                                                                                   hd_test.mean(),
                                                                                   hd_test.std()))

        save_dice.append(np.mean(dice_test, axis=1))
        save_hd.append(np.mean(hd_test, axis=1))

    dice_test_list = np.array(dice_test_list)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 110, 10), dice_test_list)
    ax.set_ylim(0.5, 1)
    ax.set_yticks(np.linspace(0.5, 1, 11))
    ax.grid(True)
    ax.set_title('test dice_up (' + best_or_last + ')')
    plt.show()
    fig.savefig(snapshot_path + '/evaluater_dice_shift_%d_' % (direction) + best_or_last + '.jpg')
    plt.close(fig)

    hd_test_list = np.array(hd_test_list)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 110, 10), hd_test_list)

    ax.grid(True)
    ax.set_title('test hd_up (' + best_or_last + ')')
    plt.show()
    fig.savefig(snapshot_path + '/evaluater_hd95_shift_%d_' % (direction) + best_or_last + '.jpg')
    plt.close(fig)

    np.save(snapshot_path + '/dice_eval%d.npy' % (direction), save_dice)
    np.save(snapshot_path + '/hd_eval%d.npy' % (direction), save_hd)


# %%
def get_learning_rate(epoch, args, method=2):
    if method == 1:
        lr = get_learning_rate1(args.base_lr, args.decay_epoch, args.min_lr, epoch, args.max_epochs)
    elif method == 2:
        lr = get_learning_rate1(args.base_lr, args.decay_epoch, args.min_lr, epoch, args.max_epochs)
        # lr=get_learning_rate2(args.base_lr, epoch, args.max_epochs)
    return lr


# %% lr decay 1
def get_learning_rate1(lr_base, decay_epoch, lr_min, epoch, max_epochs):
    if lr_base == lr_min:
        return lr_base
    lr_decay = np.exp(np.log(lr_min / lr_base) / ((max_epochs - decay_epoch) // decay_epoch))
    lr = lr_base
    for epoch_num in range(0, epoch + 1):
        if (epoch_num + 1) % decay_epoch == 0:
            lr = lr * lr_decay
            lr = max(lr, lr_min)
    return lr


# %% lr decay2
def get_learning_rate2(lr_base, decay_epoch, lr_min, epoch, max_epochs):
    if lr_base == lr_min:
        return lr_base
    lr_decay = (lr_base - lr_min) / ((max_epochs - decay_epoch) // decay_epoch)
    lr = lr_base
    for epoch_num in range(0, epoch + 1):
        if (epoch_num + 1) % decay_epoch == 0:
            lr = lr - lr_decay
            lr = max(lr, lr_min)
    return lr


# %%
grad_record = [0]


def clip_grad_(model, v=1):
    if v <= 0:
        return
    '''
    #grad_record.clear()
    for p in model.parameters():
        if p.grad is not None:
            #print("p_max", p_max.item())
            p_norm=torch.norm(p.grad.data, p=2)            
            #grad_record.append(p_norm.item())
            if p_norm > v:                
                p.grad.data*=v/p_norm
    #'''
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=v, norm_type=2)


# %%
def trainer(args, model, snapshot_path):
    logger = get_logger("training_log", snapshot_path)
    logger.info('trainer')
    logger.info(str(args))

    num_classes = args.num_classes
    Dataset_train = aug_disk_512(args.train_dataset_path, num_classes,
                                 shift=args.aug_shift, rotate=args.aug_rotate, elastic=args.aug_elastic, seed=args.seed)
    trainloader = DataLoader(dataset=Dataset_train, batch_size=args.batch_size_train,
                             num_workers=args.num_workers, pin_memory=True, shuffle=True)
    logger.info("training samples " + str(len(Dataset_train)))
    Dataset_vali = aug_disk_512(args.vali_dataset_path, num_classes,
                                shift=0, rotate=0, elastic=0, seed=args.seed)
    valiloader = DataLoader(dataset=Dataset_vali, batch_size=args.batch_size_eval,
                            num_workers=args.num_workers, pin_memory=True, shuffle=False)
    logger.info("validation samples " + str(len(Dataset_vali)))
    Dataset_test = aug_disk_512(args.test_dataset_path, num_classes,
                                shift=0, rotate=0, elastic=0, seed=args.seed)
    testloader = DataLoader(dataset=Dataset_test, batch_size=args.batch_size_eval,
                            num_workers=args.num_workers, pin_memory=True, shuffle=False)
    logger.info("test samples " + str(len(Dataset_test)))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    loss_train_list = []
    dice_train_list = []
    dice_vali_list = []
    best_vali_epoch = 0
    dice_test_list = []
    epoch_list = []
    device = args.device
    dice_per = args.dice_percentage
    try:
        loss_attn_weight = args.loss_attn_weight
    except:
        loss_attn_weight = 0
    grad_clip_max_norm = args.grad_clip_max_norm

    epoch_start = 0
    if args.resume_training != 0:
        load_model_path = os.path.join(snapshot_path, 'last.pth')
        data = torch.load(load_model_path, map_location="cpu")
        model.load_state_dict(data["model_state_dict"])
        Dataset_train.rng.set_state(data['rng_state_train'])
        loss_train_list = data["loss_train_list"]
        dice_train_list = data["dice_train_list"]
        dice_vali_list = data["dice_vali_list"]
        dice_test_list = data["dice_test_list"]
        epoch_list = data["epoch_list"]
        max_vali_dice = max(dice_vali_list)
        epoch_start = data["epoch_num"] + 1
        logger.info("resume training, epoch_start=" + str(epoch_start))

    lr = get_learning_rate(epoch_start, args)

    if args.optimizer == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=1e-4)

    if epoch_start > 0:
        optimizer.load_state_dict(data["optimizer_state_dict"])

    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    iterator = tqdm(range(epoch_start, max_epoch), ncols=70, total=max_epoch)
    for epoch_num in iterator:
        if epoch_num == 5:
            break
        logger.info('epoch %d : lr: %f' % (epoch_num, lr))
        # Dataset_train.set_sigma(min(0.25, 0.25*2*epoch_num/(max_epoch-1)))
        epoch_loss = 0
        epoch_dice = 0
        model.train()
        for batch_idx, (image_batch, label_batch, mask_batch) in enumerate(trainloader):
            image_batch = image_batch.to(torch.float32).to(device)
            label_batch = label_batch.to(torch.int64).to(device)
            mask_batch = mask_batch.to(torch.float32).to(device)
            loss_attn = 0
            if 'SymTC' not in args.net_name:
                outputs = model(image_batch)
            else:
                outputs, attn_weight = model(image_batch, return_attn_weight=True)
                attn_weight = flatten_list(attn_weight)
                if loss_attn_weight > 0:
                    loss_attn = 0
                    for attn_w in attn_weight:
                        if torch.is_tensor(attn_w):
                            loss_attn = loss_attn + attn_loss(attn_w, label_batch)

            loss_ce = CE(outputs, label_batch, mask_batch)
            outputs = torch.softmax(outputs, dim=1)
            # only calculate the bone and disk dice
            # loss_dice = 1-dice(outputs, mask_batch, reduction='mean')
            loss_dice = 1 - dice(outputs[:, 1:], mask_batch[:, 1:], reduction='mean')
            loss = (1 - dice_per) * loss_ce + dice_per * loss_dice + loss_attn * loss_attn_weight
            optimizer.zero_grad()
            loss.backward()
            clip_grad_(model, grad_clip_max_norm)
            optimizer.step()
            iter_num = iter_num + 1
            # logging.info('iteration %d : loss : %f, loss_dice: %f' % (iter_num, loss.item(), loss_dice.item()))
            epoch_loss += loss.item()
            epoch_dice += 1 - loss_dice.item()
        # one epoch is done
        epoch_loss /= len(trainloader)
        epoch_dice /= len(trainloader)
        loss_train_list.append(epoch_loss)
        dice_train_list.append(epoch_dice)
        dice_train = dice_train_list[-1]

        lr_new = get_learning_rate(epoch_num, args)
        if lr_new != lr:
            lr = lr_new
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        max_vali_dice = 0
        if len(dice_vali_list) > 0:
            max_vali_dice = max(dice_vali_list)

        epoch_list.append(epoch_num)

        # Validation
        logger.info('performing validation')
        dice_vali, _ = inference_new(valiloader, model, device, num_classes, "mean")
        dice_vali_list.append(dice_vali)
        logger.info('validation is completed')

        # save the best model
        if dice_vali > max_vali_dice:
            # Test
            logger.info('performing test')
            dice_test, _ = inference_new(testloader, model, device, num_classes, "mean")
            dice_test_list.append(dice_test)
            logger.info('test is completed')
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            logger.info('epoch %d : train: %f, vali: %f, test: %f' % (epoch_num, dice_train, dice_vali, dice_test))
            data_save = {"args": args,
                         "epoch": epoch_num,
                         "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "rng_state_train": Dataset_train.rng.get_state(),
                         "loss_train_list": loss_train_list,
                         "dice_train_list": dice_train_list,
                         "dice_vali_list": dice_vali_list,
                         "dice_test_list": dice_test_list,
                         "epoch_list": epoch_list
                         }

            best_vali_epoch = epoch_num
            save_model_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(data_save, save_model_path)
            logger.info("{} save best model to {}".format("epoch " + str(epoch_num), save_model_path))
            logger.info("best validation dice: {}".format(str(dice_vali)))

        if (epoch_num + 1) % args.plot_training_epoch == 0:
            # plot and save training_process JPG
            fig, ax = plt.subplots()
            ax.plot(loss_train_list)
            ax.set_ylim(0, 1.0)
            ax.set_title('train_loss')
            plt.show()
            jpg_path = os.path.join(snapshot_path, 'train_loss_epoch%s.jpg' % str(epoch_num))
            fig.savefig(jpg_path)
            plt.close(fig)
            fig, ax = plt.subplots(1, 2, sharey=True)
            ax[0].plot(epoch_list, dice_vali_list, '-g')
            ax[0].plot(dice_train_list, '-b')
            ax[0].set_title('dice train(blue) vali(green)')
            ax[0].grid(True)
            ax[0].set_ylim(0.9, 1.0)
            ax[0].set_yticks(np.linspace(0.9, 1.0, 11))
            ax[1].plot(epoch_list, dice_test_list)
            ax[1].set_title('dice_test')
            ax[1].grid(True)
            plt.show()
            jpg_path = os.path.join(snapshot_path, 'vali_test_dice_epoch%s.jpg' % str(epoch_num))
            fig.savefig(jpg_path)
            plt.close(fig)
    # done
    logger.info("best_vali_epoch " + str(best_vali_epoch))


# %%
def model_selection(model_name, args):
    from model.symtc import SymTC
    if model_name == 'SymTC_s':
        args.patch_size = [16, 16, 4, 1]  # vit_patches_size'
        args.embed_dim = 256  # embeded dimension
        args.n_heads = 8  # number of heads in MHA
        args.dropout = 0  # dropout in MHA
        args.n_layers = [2, 2, 2, 2]  # number of layers in MHA
        args.alpha = 100  # alpha for sin position embed
        args.use_cnn = 1  # 1: use cnn in the model
        args.use_attn = 1  # 1: use attn in the model
        args.num_classes = 12  # output channel of network

    elif model_name == 'SymTC':
        args.patch_size = [16, 16, 4, 1]  # vit_patches_size'
        args.embed_dim = 512  # embeded dimension
        args.n_heads = 16  # number of heads in MHA
        args.dropout = 0  # dropout in MHA
        args.n_layers = [2, 2, 2, 2]  # number of layers in MHA
        args.alpha = 100  # alpha for sin position embed
        args.use_cnn = 1  # 1: use cnn in the model
        args.use_attn = 1  # 1: use attn in the model
        args.num_classes = 12  # output channel of network

    model = SymTC(args)
    return model
