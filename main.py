import pickle
import config.config as cf
import torch
from model.Dataset import MyDataSet
import torch.utils.data as Data
import numpy as np
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from model.caculate import caculate_metric
from model.Model import Model,CnnModel,Transformer
import pandas as pd

def load_config():
    '''Set the required variables in the configuration'''

    # path_train_data = './dataset/Bitter Peptide/train/train.tsv'
    # path_test_data = './dataset/Bitter Peptide/test/test.tsv'

    # path_train_data = './dataset/Tumor Homing Peptide/train/Maintrain.tsv'
    # path_test_data = './dataset/Tumor Homing Peptide/test/Maintest.tsv'

    # path_train_data = './dataset/Tumor Homing Peptide/train/Main90train.tsv'
    # path_test_data = './dataset/Tumor Homing Peptide/test/Main90test.tsv'

    # path_train_data = './dataset/DPP-IV inhibitory peptide/train/train.tsv'
    # path_test_data = './dataset/DPP-IV inhibitory peptide/test/test.tsv'

    path_train_data = './dataset/Tumor Homing Peptide/train/Smalltrain.tsv'
    path_test_data = './dataset/Tumor Homing Peptide/test/Smalltest.tsv'

    '''Get configuration'''
    config = cf.get_train_config()

    '''Set other variables'''
    b = 0.06
    model_name = 'Model'

    '''initialize result folder'''
    result_folder = './parameter/' + config.learn_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    '''Save all variables in configuration'''
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data

    config.b = b
    config.if_multi_scaled = False
    config.model_name = model_name
    config.result_folder = result_folder

    return config

residue2idx = pickle.load(open('./dataset/meta_data/residue2idx.pkl', 'rb'))
cf.vocab_size = len(residue2idx)
cf.token2index = residue2idx

def save_model(model_dict, best_acc, save_dir, save_prefix,data_set):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = '{},{}, ACC[{:.4f}].pt'.format(save_prefix,data_set, best_acc)
    save_path_pt = os.path.join(save_dir, filename)
    print('save_path_pt', save_path_pt)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)
    print('Save Model Over: {},{}, ACC: {:.4f}'.format(save_prefix,data_set, best_acc))
    return save_path_pt


def adjust_model(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    pass


def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))

    return sequences, labels


def transform_token2index(sequences, config):
    token2index = residue2idx
    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)
    return token_list, max_len


def make_data_with_unified_length(token_list, labels, config):
    padded_max_len = config.max_len
    token2index = residue2idx

    data = []
    for i in range(len(labels)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]
        n_pad = padded_max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append([token_list[i], labels[i]])
    return data


def construct_dataset(data, config):
    cuda = config.cuda
    batch_size = config.batch_size

    input_ids, labels = zip(*data)


    if cuda:
        input_ids, labels = torch.cuda.LongTensor(input_ids), torch.cuda.LongTensor(labels)
    else:
        input_ids, labels = torch.LongTensor(input_ids), torch.LongTensor(labels)

    data_loader = Data.DataLoader(MyDataSet(input_ids, labels),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)

    return data_loader

def construct_dataset_text(data, config):
    cuda = config.cuda
    batch_size = config.batch_size

    input_ids, labels = zip(*data)


    if cuda:
        input_ids, labels = torch.cuda.LongTensor(input_ids), torch.cuda.LongTensor(labels)
    else:
        input_ids, labels = torch.LongTensor(input_ids), torch.LongTensor(labels)


    data_loader = Data.DataLoader(MyDataSet(input_ids, labels),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False)

    return data_loader

def load_data(config):
    path_data_train = config.path_train_data
    path_data_test = config.path_test_data

    sequences_train, labels_train = load_tsv_format_data(path_data_train)
    sequences_test, labels_test = load_tsv_format_data(path_data_test)

    token_list_train, max_len_train = transform_token2index(sequences_train, config)
    token_list_test, max_len_test = transform_token2index(sequences_test, config)

    config.max_len = 92

    data_train = make_data_with_unified_length(token_list_train, labels_train, config)
    data_test = make_data_with_unified_length(token_list_test, labels_test, config)

    data_loader_train = construct_dataset(data_train, config)
    data_loader_test = construct_dataset_text(data_test, config)

    return data_loader_train, data_loader_test

def get_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, config.num_class), label.view(-1))
    loss = (loss.float()).mean()

    loss = (loss - config.b).abs() + config.b

    return loss

def model_eval(data_iter, model, criterion, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    #测试之前
    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            if config.if_multi_scaled:
                input, origin_inpt, label = batch
                logits, output = model(input, origin_inpt)
            else:
                input, label = batch
                logits, output = model(input)

            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())

            loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            loss = (loss.float()).mean()
            avg_loss += loss

            pred_prob_all = F.softmax(logits, dim=1)
            # Prediction probability [batch_size, class_num]
            pred_prob_positive = pred_prob_all[:, 1]
            # Probability of predicting positive classes [batch_size]
            pred_prob_sort = torch.max(pred_prob_all, 1)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            # The location (class) of the predicted maximum probability in each sample [batch_size]
            corrects += (pred_class == label).sum()

            iter_size += label.shape[0]

            label_pred = torch.cat([label_pred, pred_class.float()])
            label_real = torch.cat([label_real, label.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

    metric, roc_data, prc_data = caculate_metric(label_pred, label_real, pred_prob)
    avg_loss /= iter_size
    # accuracy = 100.0 * corrects / iter_size
    accuracy = metric[0]
    print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                  accuracy,
                                                                  corrects,
                                                                  iter_size))

    return metric, avg_loss, repres_list, label_list, roc_data, prc_data


def periodic_test(test_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    test_metric, test_loss, test_repres_list, test_label_list, \
    test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, config)

    print('Model current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(test_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_test_interval.append(sum_epoch)
    test_acc_record.append(test_metric[0])
    test_loss_record.append(test_loss)

    return test_metric, test_loss, test_repres_list, test_label_list


def periodic_valid(valid_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)

    valid_metric, valid_loss, valid_repres_list, valid_label_list, \
    valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)

    print('validation current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(valid_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_valid_interval.append(sum_epoch)
    valid_acc_record.append(valid_metric[0])
    valid_loss_record.append(valid_loss)

    return valid_metric, valid_loss, valid_repres_list, valid_label_list


def get_traindata(config):
    path_data_train = config.path_train_data
    path_data_test = config.path_test_data


    sequences_train, labels_train = load_tsv_format_data(path_data_train)
    if path_data_test is not None:
        sequences_test, labels_test = load_tsv_format_data(path_data_test)

    token_list_train, max_len_train = transform_token2index(sequences_train, config)
    if path_data_test is not None:
        token_list_test, max_len_test = transform_token2index(sequences_test, config)
    else:
        max_len_test = 0
    # token_list_train: [[1, 5, 8], [2, 7, 9], ...]

    config.max_len = max(max_len_train, max_len_test)
    config.max_len_train = max_len_train
    config.max_len_test = max_len_test
    config.max_len = config.max_len + 2  # add [CLS] and [SEP]
    config.max_len = 92

    data_test = make_data_with_unified_length(token_list_test, labels_test, config)
    data_train = make_data_with_unified_length(token_list_train, labels_train, config)

    return data_train, data_test


def train_ACP(train_iter, valid_iter, test_iter, model, optimizer, criterion, config, iter_k):
    steps = 0
    best_acc = 0
    best_performance = 0

    _, shuju_test = get_traindata(config)
    sequence_test, label_test = zip(*shuju_test)
    sequences_test, labels_test = torch.cuda.LongTensor(sequence_test), torch.cuda.LongTensor(label_test)

    shuju_test = []
    shuju_test.append(sequences_test)
    shuju_test.append(labels_test)

    for epoch in range(1, config.epoch + 1):
        model.train()
        repres_list = []
        label_list = []
        text_test = []
        for batch in train_iter:
            if config.if_multi_scaled:
                input, origin_input, label = batch
                logits, output = model(input, origin_input)
            else:
                input, label = batch
                logits, output = model(input)

                repres_list.extend(output.cpu().detach().numpy())
                label_list.extend(label.cpu().detach().numpy())

            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

            '''Periodic Train Log'''
            if steps % config.interval_log == 0:
                corrects = (torch.max(logits, 1)[1] == label).sum()
                the_batch_size = label.shape[0]
                train_acc = 100.0 * corrects / the_batch_size
                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                        loss,
                                                                                        train_acc,
                                                                                        corrects,
                                                                                        the_batch_size))
                print()

                step_log_interval.append(steps)
                train_acc_record.append(train_acc)
                train_loss_record.append(loss)

        input_test, label_text = shuju_test
        logits_test, output_test = model(input_test, label_input=False, label=label_text.cpu().detach().numpy(),
                                         epoch_num=epoch)
        # probability = torch.softmax(logits2, dim=1).tolist()
        logits_test = torch.softmax(logits_test, dim=1).tolist()
        # logits_test = logits_test.tolist()
        text_test.extend([[sequence_test[i], logits_test[i], label_test[i]] for i in range(sequence_test.__len__())])

        df = pd.DataFrame(text_test)
        df.to_csv('./result/Epoch_{}.csv'.format(epoch))

        sum_epoch = iter_k * config.epoch + epoch

        '''Periodic Validation'''
        if valid_iter and sum_epoch % config.interval_valid == 0:
            valid_metric, valid_loss, valid_repres_list, valid_label_list = periodic_valid(valid_iter,
                                                                                           model,
                                                                                           criterion,
                                                                                           config,
                                                                                           sum_epoch)
            valid_acc = valid_metric[0]
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_performance = valid_metric

        '''Periodic Test'''
        if test_iter and sum_epoch % 1 == 0:
            time_test_start = time.time()

            test_metric, test_loss, test_repres_list, test_label_list = periodic_test(test_iter,
                                                                                      model,
                                                                                      criterion,
                                                                                      config,
                                                                                      sum_epoch)
            '''Periodic Save'''
            # save the model if specific conditions are met
            test_acc = test_metric[0]
            if test_acc > best_acc:
                best_acc = test_acc
                best_performance = test_metric
                if config.save_best and best_acc > config.threshold:
                    save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name,config.data_set)

            test_label_list = [x + 2 for x in test_label_list]
            repres_list.extend(test_repres_list)
            label_list.extend(test_label_list)

    return best_performance


def draw_figure_train_test(config, fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(train_loss_record):
        train_loss_record[i] = e.cpu().detach()

    for i, e in enumerate(test_acc_record):
        test_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(test_loss_record):
        test_loss_record[i] = e.cpu().detach()

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Test Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_test_interval, test_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Test Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_test_interval, test_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


def train_test(train_iter, test_iter, config):
    print('=' * 50, 'Model', '=' * 50)

    model = Model(config)

    if config.cuda: model.cuda()
    adjust_model(model)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.reg)
    criterion = nn.CrossEntropyLoss()

    print('=' * 50 + 'Start Training' + '=' * 50)
    best_performance = train_ACP(train_iter, None, test_iter, model, optimizer, criterion, config, 0)
    print('=' * 50 + 'Train Finished' + '=' * 50)

    print('*' * 60 + 'The Last Test' + '*' * 60)
    last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, \
    last_test_roc_data, last_test_prc_data = model_eval(test_iter, model, criterion, config)
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(last_test_metric.numpy())
    print('*' * 60 + 'The Last Test Over' + '*' * 60)

    return model, best_performance, last_test_metric

if __name__ == '__main__':
    print("----------------Model----------------------")
    record = []
    np.set_printoptions(linewidth=400, precision=4)
    time_start = time.time()

    '''load configuration'''
    config = load_config()

    '''set device'''
    torch.cuda.set_device(config.device)

    '''load data'''
    train_iter, test_iter = load_data(config)
    print('=' * 20, 'load data over', '=' * 20)

    '''draw preparation'''
    step_log_interval = []
    train_acc_record = []
    train_loss_record = []
    step_valid_interval = []
    valid_acc_record = []
    valid_loss_record = []
    step_test_interval = []
    test_acc_record = []
    test_loss_record = []

    '''train procedure'''
    valid_performance = 0
    best_performance = 0
    last_test_metric = 0

    if config.k_fold == -1:
        model, best_performance, last_test_metric = train_test(train_iter, test_iter, config)
        pass

    draw_figure_train_test(config, config.learn_name)

    '''report result'''
    print('*=' * 50 + 'Result Report' + '*=' * 50)
    if config.k_fold == -1:
        print('last T5_BERT_Model performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(last_test_metric))
        print()
        print('best_performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(best_performance))
    print('*=' * 50 + 'Report Over' + '*=' * 50)

    '''save train result'''
    # save the model if specific conditions are met
    if config.k_fold == -1:
        best_acc = best_performance[0]
        last_test_acc = last_test_metric[0]
        if last_test_acc > best_acc:
            best_acc = last_test_acc
            best_performance = last_test_metric
            if config.save_best and best_acc >= config.threshold:
                save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name,config.data_set)

    # save the model configuration
    with open(config.result_folder + '/Trainp[10], Test[10], Epoch[30], ACC[0.5420].pkl', 'wb') as file:
        pickle.dump(config, file)
    print('-' * 50, 'Config Save Over', '-' * 50)

    time_end = time.time()
    print('total time cost', time_end - time_start, 'seconds')

    record_file = open('./recode-2.txt', 'a')

    record_file.write(
        "lr:{} heads:{} epoch:{} acc:{} pre:{} sen:{} spe:{} F1:{} AUC:{} MCC:{}".format(config.lr, config.num_head,
                                                                                         config.epoch
                                                                                         , best_performance[0],
                                                                                         best_performance[1],
                                                                                         best_performance[2],
                                                                                         best_performance[3],
                                                                                         best_performance[4],
                                                                                         best_performance[5],
                                                                                         best_performance[6]))
    record_file.write('\n')

    record_file.close()