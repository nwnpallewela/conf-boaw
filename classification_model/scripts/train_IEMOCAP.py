import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
# import tensorboard
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import argparse
import time
import pickle
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support
from model import BiModel, MaskedNLLLoss
from dataloader import IEMOCAPDataset

np.random.seed(1234)
details_labels = pd.read_csv("../../audio-features/data/final_g_emotion_df.csv")

id = []
label = []
new_pred = []
adv_pred = []

detailed = {}
for row in details_labels.iterrows():
    detailed[row[1][8]] = row[1]


# 'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5
def adjust_label(pred, label):
    if label == 0 or label == 4:
        if pred == 0 or pred == 4:
            return label
        else:
            return pred
    elif label == 3 or label == 5:
        if pred == 3 or pred == 5:
            return label
        else:
            return pred
    else:
        return pred


def get_value(pred):
    if pred == 0:
        return 0
    elif pred == 1:
        return 1
    elif pred == 2:
        return 2
    elif pred == 3:
        return 3
    elif pred == 4:
        return 4
    elif pred == 5:
        return 5


def advanced_pred(file, pred, label1):
    row = detailed.get(file)
    if row is None:
        return get_value(label1)
    hap = row[1]  # 0
    sad = row[2]  # 1
    fru = row[3]  # 5
    exited = row[4]  # 4
    anger = row[5]  # 3
    neu = row[6]  # 2
    emo = row[9]
    if pred == 0 and hap > 0:
        # print(hap)
        return label1
    elif pred == 1 and sad > 0:
        # print(sad)
        return get_value(label1)
    elif pred == 2 and neu > 0:
        # print(neu)
        return get_value(label1)
    elif pred == 3 and anger > 0:
        # print(anger)
        return get_value(label1)
    elif pred == 4 and exited > 0:
        # print(exited)
        return get_value(label1)
    elif pred == 5 and fru > 0:
        # print(fru)
        return get_value(label1)
    else:
        return get_value(pred)


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False, fold=0):
    trainset = IEMOCAPDataset(path=path,fold=fold)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, train=False, fold=fold)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    ids = []
    last_layer = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        vid = [d.cuda() for d in data[6:7]] if cuda else data[6:7][0]
        textf, visuf, acouf, qmask, umask, label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        length = umask.size()[1]
        log_prob, alpha, alpha_f, alpha_b, hidden = model(acouf, textf, visuf, qmask, umask,
                                                          att2=True)  # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        if str(loss).startswith('tensor(nan'):
            print(vid)
        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        ids.append(vid)
        last_layer.append(torch.reshape(log_prob.data, (log_prob.data.shape[0] * log_prob.data.shape[1],
                                                        log_prob.data.shape[2])).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    if not param[1].grad.dtype.is_floating_point:
                        print()
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
        last_layer = np.concatenate(last_layer)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), [], []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return ids, avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids], last_layer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=30, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=True,
                        help='active listener')
    parser.add_argument('--attention', default='simple', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='Enables tensorboard log')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        writer = SummaryWriter()

    batch_size = args.batch_size
    n_classes = 6
    # n_classes  = 6
    cuda = args.cuda
    n_epochs = args.epochs
    fold = 3
    D_m = 2000
    D_g = 500
    D_p = 500
    D_e = 300
    D_h = 300

    D_a = 300  # concat attention

    model = BiModel(D_m, D_g, D_p, D_e, D_h,
                    n_classes=n_classes,
                    listener_state=args.active_listener,
                    context_attention=args.attention,
                    dropout_rec=args.rec_dropout,
                    dropout=args.dropout)
    if cuda:
        model.cuda()
    loss_weights = torch.FloatTensor([
        # 1/0.207261,
        1 / 0.078526,
        1 / 0.145760,
        1 / 0.230020,
        1 / 0.161918,
        1 / 0.128735,
        1 / 0.255041,
    ])
    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_loader, valid_loader, test_loader = \
        get_IEMOCAP_loaders('../data/IEMOCAP_features_raw.pkl',
                            valid=0.0,
                            batch_size=batch_size,
                            num_workers=2, fold=fold)

    best_loss, best_label, best_pred, best_mask, best_train_last_layer, best_test_last_layer = None, None, None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_ids, train_loss, train_acc, _, _, _, train_fscore, _, train_last_layer = train_or_eval_model(model,
                                                                                                           loss_function,
                                                                                                           train_loader,
                                                                                                           e,
                                                                                                           optimizer,
                                                                                                           True)
        test_ids, test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_last_layer = train_or_eval_model(
            model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
            best_ids, best_loss, best_label, best_pred, best_mask, best_attn, best_test_last_layer, best_train_last_layer = \
                test_ids, test_loss, test_label, test_pred, test_mask, attentions, test_last_layer, train_last_layer

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)
        print(
            'epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} test_loss {} test_acc {} test_fscore {} time {}'. \
                format(e + 1, train_loss, train_acc, train_fscore, 0, 0, 0, test_loss, test_acc, test_fscore,
                       round(time.time() - start_time, 2)))
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    best_ids_1 = [item for sublist in best_ids for array in sublist for item in array]
    updated_ids = []
    count_id = 0
    for i in test_mask:
        if i == 1:
            updated_ids.append(best_ids_1[count_id])
            count_id = count_id + 1
        else:
            updated_ids.append("masked")
    print('Loss {} accuracy {}'.format(best_loss,
                                       round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100, 2)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
    best_prediction_test = pd.DataFrame(data={"video_id": updated_ids, "pred": best_pred, "label": best_label})
    best_last_layer_test = pd.DataFrame(data=best_test_last_layer).to_csv("best_test_last_layer.csv")
    best_prediction_test.to_csv("best_prediction_test.csv")
    for row in best_prediction_test.iterrows():
        id.append(row[1][0])
        label.append(row[1][2])
        new_pred.append(adjust_label(row[1][1], row[1][2]))
        adv_pred.append(advanced_pred(row[1][0], row[1][1], row[1][2]))
    print(classification_report(label, adv_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(label, adv_pred, sample_weight=best_mask))
