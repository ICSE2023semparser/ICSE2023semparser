import time
import sys
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from utils.metric import get_ner_fmeasure
# from model.seqlabel import SeqLabel
# from model.sentclassifier import SentClassifier
from utils.data import Data
from model.semparser import SemParser
from write_result import write
import pandas as pd
import json
from tqdm import tqdm
# sys.path.append('../Jinfer')
from pair_infer import joint_infer
import os


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, chars, combines, combine_labels],[words, chars, combine_labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    chars = [sent[1] for sent in input_batch_list]
    combines = [sent[2] for sent in input_batch_list]
    combine_labels = [sent[3] for sent in input_batch_list]
    # labels = [sent[3] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad = if_train).long()
    # label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad = if_train).long()
    # feature_seq_tensors = []
    # for idx in range(feature_num):
    #     feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad =  if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad = if_train).bool()
    for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    # for idx in range(feature_num):
    #     feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    # comb
    comb_seq_lengths = torch.LongTensor(list(map(len, combines)))
    max_comb_len = comb_seq_lengths.max().item()
    comb_seq_tensor = torch.zeros((batch_size, max_comb_len, 2), requires_grad=if_train).long()
    comb_label_seq_tensor = torch.zeros((batch_size, max_comb_len), requires_grad=if_train).long()
    comb_mask = torch.zeros((batch_size, max_comb_len), requireds_grad=if_train).bool()
    for idx, (comb, comb_len, comb_label) in enumerate(zip(combines, comb_seq_lengths, combine_labels)):
        comb_len = comb_len.item()
        comb_seq_tensor[idx, :comb_len, :] = torch.LongTensor(comb)
        comb_label_seq_tensor[idx, :] = torch.LongTensor(comb_label)
        comb_mask[idx, :comb_len] = torch.Tensor([1]*comb_len)

    comb_seq_lengths, comb_perm_idx = comb_seq_lengths.sort(0, descending=True)
    comb_seq_tensor = comb_seq_tensor[comb_perm_idx]
    comb_label_seq_tensor = comb_label_seq_tensor[comb_perm_idx]

    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    # if gpu:
    #     word_seq_tensor = word_seq_tensor.cuda()
    #     for idx in range(feature_num):
    #         feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
    #     word_seq_lengths = word_seq_lengths.cuda()
    #     word_seq_recover = word_seq_recover.cuda()
    #     label_seq_tensor = label_seq_tensor.cuda()
    #     char_seq_tensor = char_seq_tensor.cuda()
    #     char_seq_recover = char_seq_recover.cuda()
    #     mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, comb_seq_tensor, comb_seq_lengths, comb_label_seq_tensor, mask

def singlefy_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, chars, combines, combine_labels]]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [input_batch_list[0][0]]
    token_labels = [input_batch_list[0][4]]
    chars = [input_batch_list[0][1]]
    combines = [input_batch_list[0][2]]
    combine_labels = [input_batch_list[0][3]]
    word_features = [input_batch_list[0][5]]

    word_seq_length = torch.LongTensor([len(words[0])])

    word_seq_tensor = torch.zeros((1, word_seq_length), requires_grad = if_train).long()
    for idx, (seq, seqlen) in enumerate(zip(words, word_seq_length)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    pad_chars = [chars[idx] + [[0]] * (word_seq_length[0]-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, word_seq_length[0], max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[0].view(batch_size*word_seq_length[0],-1)
    char_seq_lengths = char_seq_lengths[0].view(batch_size*word_seq_length[0],)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    word_seq_recover = [0]

    token_labels_tensor = torch.tensor(token_labels)
    word_features_tensor = torch.tensor(word_features).float()
    # print(word_features_tensor.size())
    # if gpu:
    #     word_seq_tensor = word_seq_tensor.cuda()
    #     for idx in range(feature_num):
    #         feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
    #     word_seq_lengths = word_seq_lengths.cuda()
    #     word_seq_recover = word_seq_recover.cuda()
    #     label_seq_tensor = label_seq_tensor.cuda()
    #     char_seq_tensor = char_seq_tensor.cuda()
    #     char_seq_recover = char_seq_recover.cuda()
    #     mask = mask.cuda()
    return word_seq_tensor, word_seq_length, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, combines, combine_labels, token_labels_tensor, word_features_tensor

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train(data, model):
    print("Training model...")

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    for idx in range(data.HP_iteration):
        model.train()

        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        # batch_size = data.HP_batch_size
        batch_size = 1
        batch_id = 0
        train_num = len(data.train_Ids)
        # total_batch = train_num//batch_size+1
        total_batch = train_num
        total_loss = 0
        full_loss = 0
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_combine, batch_combine_labels, token_labels, word_features = singlefy_sequence_labeling_with_label(instance, data.HP_gpu, True)
            # loss = model.compute_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
            #                                  batch_combine, batch_combine_labels, token_labels)
            # loss.backward()
            # full_loss += loss.item()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # optimizer.step()
            # model.zero_grad()
            # print(batch_word)
            for combine, combine_label in zip(batch_combine[0], batch_combine_labels[0]):
                loss = model.compute_single_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, combine, combine_label, token_labels, word_features)
                # word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, comb_seq_tensor, comb_seq_lengths, comb_label_seq_tensor, mask = batchify_sequence_labeling_with_label(instance, data.HP_gpu, True)
                # loss = model.compute_batch_loss(word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, comb_seq_tensor, comb_seq_lengths, comb_label_seq_tensor)
                loss.backward()
                full_loss += loss.item()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                model.zero_grad()

        print('Loss:', full_loss)
        # if idx % 5 == 0 and idx!= 0:
        # optimizer = lr_decay(optimizer, idx*10, 0.05, 0.001)
        # print('alter lr!')
            # scheduler.step()

        # if loss.item() < 3.5:
        #     break

def test(data, model):

    model.eval()
    test_num = len(data.test_Ids)
    # total_batch = train_num//batch_size+1
    batch_size = 1
    total_batch = test_num

    outputs = []

    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > test_num:
            end = test_num
        instance = data.test_Ids[start:end]
        if not instance:
            continue

        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_combine, batch_combine_labels, token_labels, word_features = singlefy_sequence_labeling_with_label(instance, data.HP_gpu, False)
        predicts, predict_token_label = model.predict(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_combine, batch_combine_labels, word_features)

        # print(data.train_texts[start:end][0][0])
        # print(data.train_texts[start:end][0][0])
        # print(len(predicts))
        # print(instance[0][0])
        sentence = data.test_texts[start:end][0][0]
        # for predict in predicts:
        #     if predict[1] != 0:
        #         print(data.test_texts[start:end][0][0])
        #         print(data.test_texts[start:end][0][0][predict[0]], data.test_texts[start:end][0][0][predict[1]])
        # print(data.test_texts[start:end][0][0])
        # print(predict_token_label)
        # # print(batch_word)
        # # print(loss/batch_wordlen[0])
        #
        # print('*'*30)

        outputs.append(write(sentence, predicts, predict_token_label))

    #store as csv

    # df_outputs = pd.DataFrame(outputs)
    # df_outputs.to_csv(store_path, index=False)
    return outputs

            # total_loss += loss.item()


def data_initialization(data, word_emb_dir):
    # data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir, word_emb_dir)
    # data.build_alphabet(data.dev_dir)
    # data.build_alphabet(data.test_dir)
    data.fix_alphabet()

def train_base_model():

    word_emb_dir = '/Users/yintonghuo/PycharmProjects/DocLog/coref/data/w2v.emb'
    train_dir = '/Users/yintonghuo/PycharmProjects/DocLog/coref/data/annotated_log.v1.train'
    test_dir = '/Users/yintonghuo/PycharmProjects/DocLog/coref/data/RCA/tmp.txt'
    data = Data(train_dir, word_emb_dir)
    data_initialization(data, word_emb_dir)
    data.build_pretrain_emb()
    data.generate_instance('train')
    data.HP_iteration = 50


    model = CoRef(data)

    train(data, model)
    torch.save(model, 'model_820.pkl')
    # evaluate

    #model = torch.load('finetune_model.pkl')
    data.test_dir = test_dir
    #data_initialization(data)
    data.generate_instance('test')
    print(test(data, model))
    #data.build_pretrain_emb()
    #train(data, model)


    #test(data, model)

def finetune_model(train_dir, word_emb_dir, test_dir, model, finetuneset=None):

    data = Data(train_dir, word_emb_dir)
    data.generate_instance('finetune')
    data.HP_iteration = 30
    train(data, model)
    torch.save(model, '{}_finetune_model_820.pkl'.format(finetuneset))

    data.test_dir = test_dir
    data.generate_instance('test')
    outputs = test(data, model)
    print(outputs)

    # write results
    write_dir = '/Users/yintonghuo/PycharmProjects/DocLog/coref/data/Prediction'

    #formulate data to csv
    for idx, output in enumerate(outputs):
        # print(len(output['log']))
        outputs[idx]['log'] = output['log'][1:]
        # print(len(outputs[idx]['log']))
        outputs[idx]['concept'] = [i - 1 for i in output['concept']]
        outputs[idx]['instance'] = [i - 1 for i in output['instance']]
        tmp_pairs = []
        for pair in output['pairs']:
            tmp_pairs.append([pair[0] - 1, pair[1] - 1])
        outputs[idx]['pairs'] = tmp_pairs
        # print(outputs)

        # print(outputs)
        # entry: log, pairs, concept, instance

    logs = [item['log'] for item in outputs]
    pairs = [item['pairs'] for item in outputs]
    concepts = [item['concept'] for item in outputs]
    instances = [item['instance'] for item in outputs]

    structured_result = []

    real_pair, left_concept, left_instance, conceptualized, params = joint_infer(logs, pairs, concepts, instances)
    print('Inference END!')
    line_id = 1
    for log, pair, concept, instance, ctemplate, param in zip(logs, real_pair, concepts, instances, conceptualized, params):
        tmp_structure_log = {}
        tmp_structure_log['LineID'] = line_id
        tmp_structure_log['log'] = log
        tmp_structure_log['pair'] = pair
        tmp_structure_log['concept'] = [log[i] for i in concept]
        tmp_structure_log['instance'] = [log[i] for i in instance]
        tmp_structure_log['conceptualized'] = ctemplate
        tmp_structure_log['parameter'] = param
        structured_result.append(tmp_structure_log)
        line_id += 1

    df_strctured_result = pd.DataFrame(structured_result)
    df_strctured_result.to_csv(os.path.join(write_dir, '{}.csv'.format(finetuneset)), index=False)
    print('Save to ', os.path.join(write_dir, '{}.csv'.format(finetuneset)))



def test_model(model, test_dir=None):

    data = Data(train_dir=None, word_emb_dir=None)
    data.test_dir = test_dir

    data.generate_instance('test')
    outputs = test(data, model)

    return outputs



# train_base_model()


# with open('/Users/yintonghuo/PycharmProjects/DocLog/dataset/token_file.json') as f:
#     files = json.load(f)
#
# for file in tqdm(files):
    # store_path = file + '.label'

# for finetune_set in ['Andriod', 'BGL', 'Hadoop', 'HDFS', 'Linux', 'OpenStack','Spark', 'Zookeeper']:
finetune_set = 'Andriod'
train_dir = "/Users/yintonghuo/PycharmProjects/DocLog/coref/data/Finetune/{}.train".format(finetune_set)
word_emb_dir = '/Users/yintonghuo/PycharmProjects/DocLog/coref/data/w2v.emb'
# train_dir = "/Users/yintonghuo/PycharmProjects/DocLog/coref/data/RCA/tmp.txt"
# test_dir = "/Users/yintonghuo/PycharmProjects/DocLog/coref/data/RCA/tmp.txt"
test_dir = '/Users/yintonghuo/PycharmProjects/DocLog/coref/logs/tokenized/{}.log'.format(finetune_set)
model = torch.load('model_820.pkl')
# # #
# outputs = test_model(model, test_dir)

# #     # df_outputs = pd.DataFrame(outputs)
# #     # df_outputs.to_csv(store_path, index=False)
# #     # print('Store in ', store_path)
finetune_model(train_dir, word_emb_dir, test_dir, model, finetuneset=finetune_set)

# print(outputs)


