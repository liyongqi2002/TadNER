import json
import os
import random

import numpy as np
import torch

from torch import nn
from torch.utils.data import TensorDataset, Dataset, RandomSampler, DataLoader
from tqdm import tqdm
from torch.nn import functional as F, CrossEntropyLoss


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


################################################################
# model-related utils

def calculate_ce_loss(logits, label_ids, weight):
    ###################################
    loss_fct = CrossEntropyLoss(weight=weight)
    loss = loss_fct(logits, label_ids)
    return loss


#################################################################

# data preprocessing-related

def GetDataLoader(args, sentences, labels_ids, batch_size, ignore_o_sentence=True):
    sentences_filtered = []
    labels_ids_filtered = []
    if ignore_o_sentence:
        for sentence, label_ids in zip(sentences, labels_ids):
            if sum(label_ids) > 0:
                sentences_filtered.append(sentence)
                labels_ids_filtered.append(label_ids)
    else:
        sentences_filtered = sentences
        labels_ids_filtered = labels_ids

    features = []
    for sentence, label_ids in zip(sentences_filtered, labels_ids_filtered):
        features.append(convert_to_feature(sentence, label_ids, args))
    dataset = convert_features_to_dataset(features)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler,
                                  batch_size=batch_size)
    return train_dataloader


def convert_label_to_id(labels, args, strict_range=None):
    """
    :param labels: [["B-art-film","I-art-film","O"],["O","O"],]
    :return: [[2,2,0],[0,0],]
    """
    map2id = args.label2id

    labels_ids = []
    for item in labels:
        label_ids = []
        for label in item:
            if 'B-' in label:
                label_ids.append(map2id[label.split('B-')[1]])
            elif 'I-' in label:
                label_ids.append(map2id[label.split('I-')[1]])
            else:
                label_ids.append(map2id[label])
        labels_ids.append(label_ids)

    # In Domain Transfer settings,
    # if the train/test labels overlap, the restriction should be based on strict_range
    if strict_range is not None:
        label2id_test = {}
        for i, label in enumerate(args.id2label_test):
            label2id_test[label] = i + args.source_class_num

        label2id_train = {}
        for i, label in enumerate(args.id2label_train):
            label2id_train[label] = i

        duplicate_set = list(set(args.id2label_test).intersection(set(args.id2label_train)))
        new_labels_ids = []
        for label_ids in labels_ids:
            new_label_ids = []
            for label_id in label_ids:
                new_label_id = label_id
                if (label_id not in strict_range) and (args.id2label[label_id] in duplicate_set) and (label_id > 0):
                    if label_id < strict_range[0]:
                        new_label_id = label2id_test[args.id2label[label_id]]
                    elif label_id > strict_range[-1]:
                        new_label_id = label2id_train[args.id2label[label_id]]

                new_label_ids.append(new_label_id)
            new_labels_ids.append(new_label_ids)
        labels_ids = new_labels_ids
    return labels_ids


def convert_label_id_to_io(labels_ids_sentences):
    """
    :param labels_ids:[[2,2,0],[7,0],]
    :return:[[1,1,0],[1,0],]
    """
    label_io_sentences = []
    for label_id_sentence in labels_ids_sentences:
        label_io_sentence = [0] * len(label_id_sentence)
        for idx in range(len(label_id_sentence)):
            if (label_id_sentence[idx] > 0):
                label_io_sentence[idx] = 1
        label_io_sentences.append(label_io_sentence)

    return label_io_sentences


def convert_to_feature(sentence, label_ids, args):
    max_seq_length = args.max_seq_length
    sentence_tokens = []
    label_ids_tokens = []
    for word, label_id in zip(sentence, label_ids):
        word_tokens = args.tokenizer.tokenize(word)
        label_id_tokens = [label_id] + [-1] * (len(word_tokens) - 1)
        if len(word_tokens) == 0:  # Meet special space character
            word_tokens = args.tokenizer.tokenize('[UNK]')
            label_id_tokens = [label_id]
        sentence_tokens.extend(word_tokens)
        label_ids_tokens.extend(label_id_tokens)

    sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]
    label_ids_tokens = [-1] + label_ids_tokens + [-1]

    input_ids = args.tokenizer.convert_tokens_to_ids(sentence_tokens)

    padding_length = max_seq_length - len(input_ids)
    if padding_length >= 0:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids_padded = input_ids + [0] * padding_length
        label_ids_tokens += [-1] * padding_length
    else:
        attention_mask = ([1] * len(input_ids))[:max_seq_length]
        input_ids_padded = input_ids[:max_seq_length]
        label_ids_tokens = label_ids_tokens[:max_seq_length]

    token_type_ids = [0] * max_seq_length

    assert len(input_ids_padded) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(label_ids_tokens) == max_seq_length

    return InputFeature(input_ids_padded, token_type_ids, attention_mask, label_ids_tokens)


def convert_features_to_dataset(features):
    # convert to Tensors
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([feature.token_type_ids for feature in features], dtype=torch.long)
    all_attention_mask = torch.tensor([feature.attention_mask for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature.label_ids for feature in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)
    return dataset


##################################################
# reading file-related

def read_conll2003_format_data_from_file(filepath, data_name, read_samples=False):
    """

    :param filepath: conll2003_format file
    :param dataset_name:
    :return:
    """

    with open(filepath, "r", encoding='UTF-8') as f:  # 打开文件
        data = f.readlines()
        sentences_word = []
        sentences_label = []
        sentence_word = []
        sentence_label = []
        for line in data:
            line = line.replace('\t', ' ')
            if "-DOCSTART-" not in line:
                if len(line.split(' ')) == 1:  # division marks between sentences
                    sentences_word.append(sentence_word)
                    sentences_label.append(sentence_label)
                    sentence_word = []  # new sentence
                    sentence_label = []
                else:
                    sentence_word.append(line.replace('\n', '').split(' ')[0])
                    if read_samples or data_name in ['WNUT17', 'GUM', 'I2B2', 'FEW-NERD-INTRA', 'FEW-NERD-INTER']:
                        sentence_label.append(line.replace('\n', '').split(' ')[1])
                    else:
                        sentence_label.append(line.replace('\n', '').split(' ')[3])

        sentences_word = [item for item in sentences_word if len(item) > 0]
        sentences_label = [item for item in sentences_label if len(item) > 0]
        return sentences_word, sentences_label


def read_episodes_data_from_file(filepath, args, start=0, end=5000):
    episodes_data = []
    with open(filepath) as f:  # open file
        lines = f.readlines()
        print('--------getting episodes sentence and label ids---------')
        for line in tqdm(lines[start:end]):
            # print(line)

            line = json.loads(line)
            support_sentences = line["support"]["word"]
            support_labels = line["support"]["label"]
            query_sentences = line["query"]["word"]
            query_labels = line["query"]["label"]
            support_labels_ids = convert_label_to_id(support_labels, args)
            query_labels_ids = convert_label_to_id(query_labels, args)

            episode_data = {
                "support_sentences": support_sentences,
                "support_labels_ids": support_labels_ids,
                "query_sentences": query_sentences,
                "query_labels_ids": query_labels_ids,
            }
            episodes_data.append(episode_data)

    return episodes_data


def read_cross_domain_target_support_data_from_file(args):
    support_sentences_samples = []
    support_labels_samples = []
    samples_filepath = 'data_raw/' + args.dataset_target + '/samples-' + args.dataset_target + '-' + str(
        args.k_shot) + 'shot'
    samples_filenames = os.listdir(samples_filepath)
    for sample_filenames in samples_filenames:
        sample_filepath = samples_filepath + '/' + sample_filenames
        sentences, labels = read_conll2003_format_data_from_file(sample_filepath, args.dataset_target,
                                                                 read_samples=True)
        support_sentences_samples.append(sentences)
        support_labels_samples.append(labels)
    return support_sentences_samples, support_labels_samples


def read_labels_from_file(filepath, args):
    """
    :param filepath:  filepath of labels.jsonl
    :return: id2label=[]，label2id={}
    """
    with open(filepath) as f:
        labels_data = f.read()

    json_labels = json.loads(labels_data)
    id2label_train = json_labels["train"]
    id2label_dev = json_labels["dev"]
    id2label_test = json_labels["test"]

    id2proxy_label_train = json_labels["proxy_train"]
    id2proxy_label_dev = json_labels["proxy_dev"]
    id2proxy_label_test = json_labels["proxy_test"]

    id2label = []
    id2label.extend(id2label_train)
    id2label.extend(id2label_dev)
    id2label.extend(id2label_test)
    id2label.insert(0, "O")

    label2id = {}
    for i, label in enumerate(id2label):
        label2id[label] = i

    id2proxy_label = []
    id2proxy_label.extend(id2proxy_label_train)
    id2proxy_label.extend(id2proxy_label_dev)
    id2proxy_label.extend(id2proxy_label_test)
    id2proxy_label.insert(0, "other")

    proxy_label2id = {}
    for i, label in enumerate(id2proxy_label):
        proxy_label2id[label] = i

    return id2label, id2label_train, id2label_dev, id2label_test, label2id, \
        id2proxy_label, id2proxy_label_train, id2proxy_label_dev, id2proxy_label_test, proxy_label2id


def get_filepath(args):
    filepath_labels = ''
    filepath_source_train = ''
    filepath_source_dev = ''

    # used in FEW-NERD setting
    filepath_target_episodes = ''

    # used in Cross-Domain setting
    filepath_target = ''

    if args.dataset_target == 'FEW-NERD-INTRA':
        # filepath_labels = 'data_raw/FEW-NERD/intra/labels.jsonl'
        filepath_source_train = 'data_raw/FEW-NERD/intra/train.txt'
        filepath_source_dev = 'data_raw/FEW-NERD/intra/dev_' + args.n_way_k_shot + '.jsonl'
        filepath_target_episodes = 'data_raw/FEW-NERD/intra/test_' + args.n_way_k_shot + '.jsonl'

    elif args.dataset_target == 'FEW-NERD-INTER':
        # filepath_labels = 'data_raw/FEW-NERD/inter/labels.jsonl'
        filepath_source_train = 'data_raw/FEW-NERD/inter/train.txt'
        filepath_source_dev = 'data_raw/FEW-NERD/inter/dev_' + args.n_way_k_shot + '.jsonl'
        filepath_target_episodes = 'data_raw/FEW-NERD/inter/test_' + args.n_way_k_shot + '.jsonl'

    # Cross-Domain Setting
    else:
        # include train(source), dev(source), test(target) labels
        # filepath_labels = 'data_raw/' + args.dataset_target + '/labels.jsonl'
        filepath_source_train = 'data_raw/Ontonotes/train.txt'
        filepath_source_dev = 'data_raw/Ontonotes/train.txt'
        filepath_target = 'data_raw/' + args.dataset_target + '/test.txt'

    base_labels_json_file = 'labels.jsonl'

    if args.type_mode == 'original':
        base_labels_json_file = 'labels.jsonl'
    elif args.type_mode == 'meaningless':
        base_labels_json_file = 'meaningless_labels.jsonl'
    elif args.type_mode == 'misleading':
        base_labels_json_file = 'misleading_labels.jsonl'
    elif args.type_mode == 'variant1':
        base_labels_json_file = 'variant1_labels.jsonl'
    elif args.type_mode == 'variant2':
        base_labels_json_file = 'variant2_labels.jsonl'
    elif args.type_mode == 'variant3':
        base_labels_json_file = 'variant3_labels.jsonl'

    print('base_labels_json_file', base_labels_json_file)

    if args.dataset_target == 'FEW-NERD-INTRA':
        filepath_labels = 'data_raw/FEW-NERD/intra/' + base_labels_json_file
    elif args.dataset_target == 'FEW-NERD-INTER':
        filepath_labels = 'data_raw/FEW-NERD/inter/' + base_labels_json_file
    else:
        # include train(source), dev(source), test(target) labels
        filepath_labels = 'data_raw/' + args.dataset_target + '/' + base_labels_json_file
    return filepath_labels, filepath_source_train, filepath_source_dev, filepath_target_episodes, filepath_target


##################################################
# Dtaset-related
class InputFeature(object):

    def __init__(self, input_ids, token_type_ids, attention_mask, label_ids):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_ids = label_ids


class MyDataset(Dataset):  #
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


##################################################


#####################################################
# other utils

def get_original_prototypes(args, bert_encoder_pt, support_sentences, support_labels_ids, label_dict, label_types_id):
    """
    get prototypes by support
    """

    spans_emb = [[] for i in range(len(label_types_id))]
    for support_sentence, support_label_id in zip(support_sentences, support_labels_ids):
        feature = convert_to_feature(support_sentence, support_label_id, args)

        bert_encoder_outputs = \
            bert_encoder_pt(
                input_ids=torch.tensor([feature.input_ids]).to(args.device),
                token_type_ids=torch.tensor([feature.token_type_ids]).to(args.device),
                attention_mask=torch.tensor([feature.attention_mask]).to(args.device),
                output_hidden_states=True
            )

        bert_encoder_output = (torch.sum(torch.stack(bert_encoder_outputs.hidden_states[-4:]), 0) / 4).squeeze(1)
        bert_output_raw_flatten = torch.flatten(bert_encoder_output, start_dim=0, end_dim=1)[:]
        labels_flatten = torch.tensor(feature.label_ids)[:]
        filtered_indices = torch.where(labels_flatten >= 0)[0].cpu().numpy().tolist()

        filtered_bert_output_raw_flatten = bert_output_raw_flatten[filtered_indices]
        span_label_support = extract_entity_span_label(support_label_id)
        for span in span_label_support:
            span_emb = torch.sum(filtered_bert_output_raw_flatten[span["start"]:span["end"] + 1], 0) / (
                    span["end"] + 1 - span["start"])

            spans_emb[label_dict[span["label"]]].append(span_emb)

    proto_emb = []
    for item in spans_emb:
        item_ = torch.stack(item)
        proto_emb.append(torch.mean(item_, 0))

    proto_emb = torch.stack(proto_emb)
    proto_emb = F.normalize(proto_emb, p=2, dim=0)

    return proto_emb


def get_proxy_label_emb(args, ModelStage2, label_types_id):
    """
    get emb of proxy-labels
    """
    proxy_labels = [args.id2proxy_label[id_support] for id_support in label_types_id]
    labels_last_hidden_states = []
    for label in proxy_labels:
        # label = 'Entity Type' + label
        input_ids = args.tokenizer.encode(label, add_special_tokens=True)
        input_ids = torch.tensor([input_ids]).to(args.device)

        bert_encoder_outputs = ModelStage2.encoder(
            input_ids=input_ids,
            output_hidden_states=True
        )

        bert_encoder_output = (torch.sum(torch.stack(bert_encoder_outputs.hidden_states[-4:]), 0) / 4).squeeze(1)
        last_hidden_states = torch.flatten(bert_encoder_output, start_dim=0, end_dim=1)[:]

        if args.stage2_use_mlp:
            labels_last_hidden_states.append(ModelStage2.mlp(last_hidden_states[0]))
        else:
            labels_last_hidden_states.append(last_hidden_states[0])

    all_proto_emb_proxy = torch.stack(labels_last_hidden_states)
    return all_proto_emb_proxy


def extract_entity_span_label(query_label_ids):
    """
    :param query_label_ids: [[2,2,0],[7,0],] or [2,2,0]
    :return: [{"start":0,"end":1,"label":2},{"start":3,"end":3,"label":7}]或[{"start":0,"end":1,"label":2}]
    """
    query_label_ids_flatten = []
    if type(query_label_ids[0]) == list:
        for item in query_label_ids:
            query_label_ids_flatten.extend(item)
    else:
        query_label_ids_flatten = query_label_ids
    # Note here that it is important to handle both the common case of 4 4 4 0 3
    # and the case of 4 4 4 3, which is a different entity class but adjacent to each other
    span_label_golds = []
    span = {}
    last = 0

    for i in range(len(query_label_ids_flatten)):
        if query_label_ids_flatten[i] != last and last == 0:
            span["start"] = i
            last = query_label_ids_flatten[i]
        elif query_label_ids_flatten[i] != last and last > 0:
            span["end"] = i - 1
            span["label"] = query_label_ids_flatten[i - 1]
            span_label_golds.append(span)
            span = {}
            if query_label_ids_flatten[i] == 0:
                last = 0
            else:
                span["start"] = i
                last = query_label_ids_flatten[i]
    if query_label_ids_flatten[-1] > 0:  # To handle examples with entities at the end
        span["end"] = len(query_label_ids_flatten) - 1
        span["label"] = query_label_ids_flatten[-1]
        span_label_golds.append(span)

    return span_label_golds


def extract_entity_span(label_io_list):
    mention_spans = []
    if len(label_io_list) > 1:  # Only those longer than 1 will be considered next
        if label_io_list[0] == 1 and label_io_list[1] == 0:
            mention_spans.append({"start": 0, "end": 0})
        if label_io_list[0] == 1 and label_io_list[1] == 1:
            # If it is B, the span is stored temporarily and updated the next time it encounters E
            mention_spans.append({"start": 0, "end": -1})

    elif len(label_io_list) == 1:
        if label_io_list[0] == 1:
            mention_spans.append({"start": 0, "end": 0})
        return mention_spans

    for i in range(1, len(label_io_list) - 1):
        if label_io_list[i] == 1 and label_io_list[i - 1] == 0 and label_io_list[i + 1] == 0:
            # If it is S, then the mention is extracted directly
            mention_spans.append({"start": i, "end": i})
        elif label_io_list[i] == 1:
            if label_io_list[i - 1] == 0 and label_io_list[i + 1] == 1:
                # If it is B, the span is stored temporarily and updated the next time it encounters E
                mention_spans.append({"start": i, "end": -1})
            elif label_io_list[i - 1] == 1 and label_io_list[i + 1] == 0:
                # Meet E
                mention_spans[-1]["end"] = i
            # If it is 1 before or after, it is not processed

    if label_io_list[-1] == 1:  # If the last one is 1
        if len(label_io_list) > 1:  # Only those longer than 1 will be considered next
            if label_io_list[-2] == 0:  # If the last one is 1 and the previous one is 0
                mention_spans.append({"start": len(label_io_list) - 1, "end": len(label_io_list) - 1})
            elif label_io_list[-2] == 1:  # If the last one is 1 and the previous one is 1
                mention_spans[-1]["end"] = len(label_io_list) - 1

    return mention_spans
