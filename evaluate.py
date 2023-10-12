import copy
import json
import os
import random
import re
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm
from transformers import BertModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from model import BertModelStage2, BertModelStage1
from utils import read_episodes_data_from_file, convert_label_id_to_io, get_original_prototypes, convert_to_feature, \
    convert_features_to_dataset, extract_entity_span_label, extract_entity_span, convert_label_to_id, \
    read_conll2003_format_data_from_file, read_cross_domain_target_support_data_from_file, get_proxy_label_emb, \
    GetDataLoader, set_seeds, calculate_ce_loss


def adapt_predict_stage1_episode(args, episode_data, bert_model_stage1):
    ############ prepare data #####################
    episode_support_sentence = episode_data["support_sentences"]
    episode_support_label_id = episode_data["support_labels_ids"]
    # print(episode_support_sentence)
    # print(episode_support_label_id)

    episode_query_sentence = episode_data["query_sentences"]
    episode_query_label_id = episode_data["query_labels_ids"]

    episode_support_features_stage1 = []
    for sentence, label_id in zip(episode_support_sentence, episode_support_label_id):
        episode_support_features_stage1.append(convert_to_feature(sentence, label_id, args))

    episode_support_input_ids = torch.stack(
        [torch.tensor(feature.input_ids) for feature in episode_support_features_stage1])
    episode_support_attention_mask = torch.stack(
        [torch.tensor(feature.attention_mask) for feature in episode_support_features_stage1])
    episode_support_stage1 = torch.stack(
        [torch.tensor(feature.label_ids) for feature in episode_support_features_stage1])

    # stage1_finetune_time_a = time.time()

    ########################################
    optimizer_stage1 = torch.optim.Adam(bert_model_stage1.parameters(), lr=args.finetune_target_LR_stage1)

    bert_model_stage1.train()
    loss_min = 10000
    up = 0
    up_bound = 2
    if args.n_way_k_shot in ['5_5', '10_5']:
        up_bound = 6
    for i in range(100):
        optimizer_stage1.zero_grad()
        loss_stage1, _1, _2 = \
            bert_model_stage1(
                input_ids=episode_support_input_ids.to(args.device),
                attention_mask=episode_support_attention_mask.to(args.device),
                label_ids=episode_support_stage1.to(args.device),
            )
        if loss_stage1 > loss_min:
            up += 1
        else:
            up = 0
            loss_min = loss_stage1
        if up > up_bound:
            break
        loss_stage1.backward()
        optimizer_stage1.step()

    # stage1_finetune_time_b = time.time()
    # print('stage1_finetune_time_b-stage1_finetune_time_a', stage1_finetune_time_b - stage1_finetune_time_a)

    bert_model_stage1.eval()

    span_preds = []
    span_golds = []

    stage1_test_time_a = time.time()

    # for i in range(len(episode_query_input_ids)):
    #     # label_ids here is only used for identify the first token of a word,
    #     # so no label information is used for predicting
    #
    #     _, logits_stage1, label_ids_stage1 = \
    #         bert_model_stage1(
    #             input_ids=torch.stack([episode_query_input_ids[i]]).to(args.device),
    #             attention_mask=torch.stack([episode_query_attention_mask[i]]).to(args.device),
    #             label_ids=torch.stack([episode_query_stage1[i]]).to(args.device),
    #         )
    #
    #     preds_stage1 = torch.argmax(logits_stage1, dim=-1)
    #
    #     span_pred = bert_model_stage1.decode_label_ids(preds_stage1.cpu().numpy().tolist())
    #     span_gold = bert_model_stage1.decode_label_ids(label_ids_stage1)
    #
    #     span_preds.append(span_pred)
    #     span_golds.append(span_gold)

    episode_query_features_stage1 = []
    for sentence, label_id in zip(episode_query_sentence, episode_query_label_id):
        episode_query_features_stage1.append(convert_to_feature(sentence, label_id, args))

    episode_query_input_ids = torch.stack(
        [torch.tensor(feature.input_ids) for feature in episode_query_features_stage1])
    episode_query_attention_mask = torch.stack(
        [torch.tensor(feature.attention_mask) for feature in episode_query_features_stage1])
    episode_query_stage1 = torch.stack([torch.tensor(feature.label_ids) for feature in episode_query_features_stage1])

    _, logits_stage1, label_ids_stage1 = \
        bert_model_stage1(
            input_ids=episode_query_input_ids.to(args.device),
            attention_mask=episode_query_attention_mask.to(args.device),
            label_ids=episode_query_stage1.to(args.device),
        )

    preds_stage1 = torch.argmax(logits_stage1, dim=-1)
    preds_stage1 = preds_stage1.cpu().numpy().tolist()
    # print(len(preds_stage1))

    index_now = 0
    for sentence in episode_query_sentence:
        # print(sentence)
        len_sen = len(sentence)
        # print(len_sen)

        pred_io = preds_stage1[index_now:index_now + len_sen]
        gold_io = label_ids_stage1[index_now:index_now + len_sen]
        # print(pred_io)
        span_pred = bert_model_stage1.decode_label_ids(pred_io)
        span_gold = bert_model_stage1.decode_label_ids(gold_io)
        span_preds.append(span_pred)
        span_golds.append(span_gold)

        index_now += len_sen

    stage_1_inference_time = time.time() - stage1_test_time_a

    tp = 0
    num_pred = 0
    num_gold = 0
    for span_pred, span_gold in zip(span_preds, span_golds):
        for pred in span_pred:
            if pred in span_gold:
                tp += 1
        num_pred += len(span_pred)
        num_gold += len(span_gold)
    if tp == 0:
        precision_stage1_episode = 0
        recall_stage1_episode = 0
        f1_stage1_episode = 0
    else:
        precision_stage1_episode = tp / num_pred
        recall_stage1_episode = tp / num_gold
        f1_stage1_episode = 2 * precision_stage1_episode * recall_stage1_episode / (
                precision_stage1_episode + recall_stage1_episode)
    metric = {"f1": f1_stage1_episode,
              "precision": precision_stage1_episode,
              "recall": recall_stage1_episode,
              "num_true": tp,
              "num_pred": num_pred,
              "num_gold": num_gold,
              "stage_1_inference_time": stage_1_inference_time
              }
    return span_preds, metric


def adapt_predict_stage1_cross_domain(args, bert_model_stage1, support_sentences_sample, support_labels_ids_sample,
                                      query_sentences, query_labels_ids):
    optimizer_stage1 = torch.optim.Adam(bert_model_stage1.parameters(), lr=args.finetune_target_LR_stage1)
    bert_model_stage1.train()
    loss_min = 10000
    up = 0
    up_bound = 2
    if args.k_shot == 5:
        up_bound = 6

    if args.adapt_stage1:
        if args.dataset_target == 'I2B2' and args.k_shot == 5:
            # in this setting,
            # we cannot put all support data in a batch due to the GPU, so we split it
            dataloader_support_stage1 = GetDataLoader(args=args,
                                                      sentences=support_sentences_sample,
                                                      labels_ids=support_labels_ids_sample,
                                                      batch_size=8,
                                                      ignore_o_sentence=False)

            for i in range(args.finetune_target_epochs_stage1):
                loss_batch = 0
                for step, batch_stage1 in enumerate(dataloader_support_stage1):
                    optimizer_stage1.zero_grad()
                    loss_stage1, _1, _2 = \
                        bert_model_stage1(
                            input_ids=batch_stage1[0].to(args.device),
                            token_type_ids=batch_stage1[1].to(args.device),
                            attention_mask=batch_stage1[2].to(args.device),
                            label_ids=batch_stage1[3].to(args.device),
                        )
                    loss_batch += loss_stage1.item()
                    loss_stage1.backward()
                    optimizer_stage1.step()

                if loss_batch > loss_min:
                    up += 1
                else:
                    up = 0
                    loss_min = loss_batch
                if up > up_bound:
                    break
        else:
            support_stage1_features = []
            for sentence, label_stage1 in zip(support_sentences_sample, support_labels_ids_sample):
                support_stage1_features.append(convert_to_feature(sentence, label_stage1, args))

            support_data_stage1_feature = {
                "input_ids": torch.stack([torch.tensor(feature.input_ids) for feature in support_stage1_features]).to(
                    args.device),
                "token_type_ids": torch.stack(
                    [torch.tensor(feature.token_type_ids) for feature in support_stage1_features]).to(
                    args.device),
                "attention_mask": torch.stack(
                    [torch.tensor(feature.attention_mask) for feature in support_stage1_features]).to(
                    args.device),
                "label_ids": torch.stack([torch.tensor(feature.label_ids) for feature in support_stage1_features]).to(
                    args.device),
            }
            for i in range(args.finetune_target_epochs_stage1):
                optimizer_stage1.zero_grad()
                loss_stage1, _1, _2 = \
                    bert_model_stage1(
                        input_ids=support_data_stage1_feature["input_ids"],
                        attention_mask=support_data_stage1_feature["attention_mask"],
                        label_ids=support_data_stage1_feature["label_ids"],
                    )
                loss_stage1.backward()
                optimizer_stage1.step()

                if loss_stage1 > loss_min:
                    up += 1
                else:
                    up = 0
                    loss_min = loss_stage1
                if up > up_bound:
                    break
    else:
        pass

    span_preds = []
    span_golds = []

    for i in range(len(query_sentences)):
        query_sentence = query_sentences[i]
        query_label_ids = query_labels_ids[i]
        feature_stage1 = convert_to_feature(query_sentence, query_label_ids, args)
        # label_ids here is only used for identify the first token of a word,
        # so no label information is used for predicting
        _, logits_stage1, label_ids_stage1_stage1 = \
            bert_model_stage1(
                input_ids=torch.tensor([feature_stage1.input_ids]).to(args.device),
                attention_mask=torch.tensor([feature_stage1.attention_mask]).to(args.device),
                label_ids=torch.tensor([feature_stage1.label_ids]).to(args.device),
            )

        preds_stage1 = torch.argmax(logits_stage1, dim=-1)

        span_pred = bert_model_stage1.decode_label_ids(preds_stage1.cpu().numpy().tolist())
        span_gold = bert_model_stage1.decode_label_ids(label_ids_stage1_stage1)

        span_preds.append(span_pred)
        span_golds.append(span_gold)

    tp = 0
    num_pred = 0
    num_gold = 0
    for span_pred, span_gold in zip(span_preds, span_golds):
        for pred in span_pred:
            if pred in span_gold:
                tp += 1
        num_pred += len(span_pred)
        num_gold += len(span_gold)
    if tp == 0:
        precision_stage1 = 0
        recall_stage1 = 0
        f1_stage1 = 0
    else:
        precision_stage1 = tp / num_pred
        recall_stage1 = tp / num_gold
        f1_stage1 = 2 * precision_stage1 * recall_stage1 / (
                precision_stage1 + recall_stage1)
    metric = {"f1": f1_stage1,
              "precision": precision_stage1,
              "recall": recall_stage1,
              "num_true": tp,
              "num_pred": num_pred,
              "num_gold": num_gold,
              }
    return span_preds, metric


#
# def adapt_stage2_no_type_name(args, ModelStage2, sentences_support, labels_ids_support, label_types_id, label_dict):
#     ModelStage2.linear_layer = nn.Linear(args.pretrained_model_hidden_size, len(label_types_id)).to(args.device)
#
#     sentences_support_filtered = []
#     labels_ids_support_filtered = []
#     for sentence, label_ids in zip(sentences_support, labels_ids_support):
#         if sum(label_ids) > 0:
#             sentences_support_filtered.append(sentence)
#             new_label_ids = []
#             for id in label_ids:
#                 if id == 0:
#                     new_label_ids.append(id)
#                 else:
#                     # Here we add 1 for entity-token because we will minus 1 in the model
#                     new_label_ids.append(label_dict[id] + 1)
#             labels_ids_support_filtered.append(new_label_ids)
#
#     dataloader_support_io = GetDataLoader(args=args,
#                                           sentences=sentences_support_filtered,
#                                           labels_ids=labels_ids_support_filtered,
#                                           batch_size=32,
#                                           ignore_o_sentence=False)
#
#     optimizer = torch.optim.Adam(ModelStage2.parameters(), lr=args.train_source_LR_stage2)
#
#     ModelStage2.train()
#     flag = 0
#     loss_before = 1000
#     for i in range(args.finetune_target_epochs_stage2):
#         loss_batch = 0
#         for step, batch in enumerate(dataloader_support_io):
#             optimizer.zero_grad()
#             loss = \
#                 ModelStage2(
#                     input_ids=batch[0].to(args.device),
#                     token_type_ids=batch[1].to(args.device),
#                     attention_mask=batch[2].to(args.device),
#                     label_ids=batch[3].to(args.device),
#                     finetune=True,
#                 )
#             loss_batch += loss.item()
#             loss.backward()
#             optimizer.step()
#
#         avg_loss = loss_batch / len(sentences_support_filtered)
#         print('### Current avg_loss loss: ', avg_loss, ' ###')
#         if avg_loss > loss_before:
#             flag += 1
#         else:
#             flag = 0
#             loss_before = avg_loss
#         if flag > 0:
#             print('### Stop here: ', i, ' index batch ###')
#             break
#         # if avg_loss < args.finetune_target_threshold_stage2:
#         #     print('### Stop here: ', i, ' index batch ###')
#         #     print('### Current avg loss: ', avg_loss, ' ###')
#         #     break
#     return ModelStage2
#

def adapt_stage2(args, ModelStage2, sentences_support, labels_ids_support, label_types_id,
                 label_dict):
    # get the emb of type names and put it into the linear layer
    all_label_emb = get_proxy_label_emb(args, ModelStage2, label_types_id)
    ModelStage2.linear_layer = nn.Linear(args.pretrained_model_hidden_size, len(label_types_id)).to(args.device)

    ModelStage2.linear_layer.weight = nn.Parameter(all_label_emb.to(args.device))

    sentences_support_filtered = []
    labels_ids_support_filtered = []
    for sentence, label_ids in zip(sentences_support, labels_ids_support):
        if sum(label_ids) > 0:
            sentences_support_filtered.append(sentence)
            new_label_ids = []
            for id in label_ids:
                if id == 0:
                    new_label_ids.append(id)
                else:
                    # Here we add 1 for entity-token because we will minus 1 in the model
                    new_label_ids.append(label_dict[id] + 1)
            labels_ids_support_filtered.append(new_label_ids)

    # we only optimize the BERT of type classification
    optimizer = torch.optim.Adam(ModelStage2.encoder.parameters(), lr=args.train_source_LR_stage2)

    if args.dataset_target == 'I2B2':
        # in this setting,
        # we cannot put all support data in a batch due to the GPU, so we split it
        dataloader_support_io = GetDataLoader(args=args,
                                              sentences=sentences_support_filtered,
                                              labels_ids=labels_ids_support_filtered,
                                              batch_size=32,
                                              ignore_o_sentence=False)
        ModelStage2.train()
        flag = 0
        loss_before = 1000
        bert_stage2 = ModelStage2.encoder
        for i in range(args.finetune_target_epochs_stage2):
            loss_batch = 0
            for step, batch in enumerate(dataloader_support_io):
                optimizer.zero_grad()
                bert_encoder_outputs = \
                    bert_stage2(
                        input_ids=batch[0].to(args.device),
                        attention_mask=batch[2].to(args.device),
                        output_hidden_states=True
                    )
                bert_encoder_output = (torch.sum(torch.stack(bert_encoder_outputs.hidden_states[-4:]), 0) / 4).squeeze(
                    1)

                bert_output_raw_flatten = torch.flatten(bert_encoder_output, start_dim=0, end_dim=1)[:]
                labels_flatten = torch.flatten(batch[3].to(args.device), start_dim=0, end_dim=1)[:]

                filtered_indices_0 = torch.where(labels_flatten > 0)[0].cpu().numpy().tolist()
                entity_bert_output = bert_output_raw_flatten[filtered_indices_0]
                entity_label_ids = labels_flatten[filtered_indices_0] - 1
                all_label_emb = ModelStage2.linear_layer.weight

                logits = torch.matmul(entity_bert_output, all_label_emb.T)
                loss = calculate_ce_loss(logits=logits,
                                         label_ids=entity_label_ids,
                                         weight=None)
                avg_loss = loss / len(batch[0])
                if avg_loss < args.finetune_target_threshold_stage2:
                    print('### Stop here: ', step, ' index step ###')
                    print('### Current avg loss: ', avg_loss, ' ###')
                    flag += 1
                    break
                loss_batch += loss.item()
                loss.backward()
                optimizer.step()
            if flag > 0:
                print('### Stop here: ', i, ' index batch ###')
                break
            # avg_loss = loss_batch / len(sentences_support_filtered)
            # print('### Current avg_loss loss: ', avg_loss, ' ###')
            # if avg_loss > loss_before:
            #     flag += 1
            # else:
            #     flag = 0
            #     loss_before = avg_loss
            # if flag > 0:
            #     print('### Stop here: ', i, ' index batch ###')
            #     break
            # if avg_loss < args.finetune_target_threshold_stage2:
            #     print('### Stop here: ', i, ' index batch ###')
            #     print('### Current avg loss: ', avg_loss, ' ###')
            #     break
    else:
        features = []
        for sentence, label_ids in zip(sentences_support_filtered, labels_ids_support_filtered):
            features.append(convert_to_feature(sentence, label_ids, args))

        episode_support_input_ids = torch.stack([torch.tensor(feature.input_ids) for feature in features]).to(
            args.device)
        episode_support_attention_mask = torch.stack([torch.tensor(feature.attention_mask) for feature in features]).to(
            args.device)
        episode_support_label_ids = torch.stack([torch.tensor(feature.label_ids) for feature in features]).to(
            args.device)

        ModelStage2.train()
        flag = 0
        loss_before = 1000
        bert_stage2 = ModelStage2.encoder
        for i in range(args.finetune_target_epochs_stage2):
            optimizer.zero_grad()
            bert_encoder_outputs = \
                bert_stage2(
                    input_ids=episode_support_input_ids,
                    attention_mask=episode_support_attention_mask,
                    output_hidden_states=True
                )
            bert_encoder_output = (torch.sum(torch.stack(bert_encoder_outputs.hidden_states[-4:]), 0) / 4).squeeze(1)
            if args.stage2_use_mlp:
                bert_encoder_output = ModelStage2.mlp(bert_encoder_output)

            bert_output_raw_flatten = torch.flatten(bert_encoder_output, start_dim=0, end_dim=1)[:]
            labels_flatten = torch.flatten(episode_support_label_ids, start_dim=0, end_dim=1)[:]

            filtered_indices_0 = torch.where(labels_flatten > 0)[0].cpu().numpy().tolist()
            entity_bert_output = bert_output_raw_flatten[filtered_indices_0]
            entity_label_ids = labels_flatten[filtered_indices_0] - 1
            all_label_emb = ModelStage2.linear_layer.weight

            logits = torch.matmul(entity_bert_output, all_label_emb.T)
            loss = calculate_ce_loss(logits=logits,
                                     label_ids=entity_label_ids,
                                     weight=None)
            avg_loss = loss / len(features)
            print('### Current avg loss: ', avg_loss, ' ###')
            if avg_loss > loss_before:
                flag += 1
            else:
                flag = 0
                loss_before = avg_loss
            if flag > 0:
                print('### Stop here: ', i, ' index batch ###')
                break
            # if avg_loss < args.finetune_target_threshold_stage2:
            #     print('### Stop here: ', i, ' index batch ###')
            #     print('### Current avg loss: ', avg_loss, ' ###')
            #     break

            loss.backward()
            optimizer.step()

    return ModelStage2


def predict_stage2_cross_domain(args, bert_encoder_pt, all_proto_emb, span_preds, query_sentences, query_label_ids,
                                label_dict, label_types_id, span_threshold=0):
    tmp_all_logits = []
    preds = []
    stage2_test_time_a = time.time()

    for i, (span_pred, sentence, label_ids) in enumerate(tqdm(zip(span_preds, query_sentences, query_label_ids))):
        if span_pred == []:
            continue
        # label_ids here is only used for identify the first token of a word,
        # so no label information is used for predicting
        feature = convert_to_feature(sentence, label_ids, args)
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

        for span in span_pred:
            span_emb = torch.mean(filtered_bert_output_raw_flatten[span["start"]:span["end"] + 1], 0)
            if args.use_type_name:
                cat_span_emb = torch.cat((span_emb, span_emb), dim=-1)
                logit = torch.matmul(all_proto_emb, cat_span_emb)
            else:
                logit = torch.matmul(all_proto_emb, span_emb)

            max_logit = torch.max(logit).detach().cpu().numpy()
            pred = torch.argmax(logit).cpu().numpy()
            if args.filter:
                if max_logit / 2 > span_threshold:
                    preds.append(pred)
                else:
                    preds.append(-1)
            else:
                preds.append(pred)

    preds_label = []
    for item in preds:
        if item < 0:
            preds_label.append(-1)
        else:
            preds_label.append(label_types_id[item])

    idx_now = 0
    # all_preds_label corresponds to the unfolded "span_preds".
    # for example, span_preds=[[{"strat":1,"end":2},{"strat":3,"end":5}],], then all_preds_label=[[2,1],]
    all_preds_label = []

    for i, span_pred in enumerate(span_preds):
        tmp = []
        for j in range(idx_now, idx_now + len(span_pred)):
            tmp.append(preds_label[j])
        all_preds_label.append(tmp)
        idx_now += len(span_pred)

    stage_2_inference_time = time.time() - stage2_test_time_a
    print(' stage2_test_time_b - stage2_test_time_a', stage_2_inference_time)

    return all_preds_label, stage_2_inference_time


def predict_stage2_episode(args, bert_encoder_pt, all_proto_emb, span_preds, query_sentences, query_label_ids,
                           label_dict, label_types_id, span_threshold=0):
    stage2_test_time_a = time.time()

    tmp_all_logits = []
    preds = []

    query_features_input_ids = []
    query_features_token_type_ids = []
    query_features_attention_mask = []
    query_features_label_ids = []

    for i, (span_pred, sentence, label_ids) in enumerate(tqdm(zip(span_preds, query_sentences, query_label_ids))):
        feature = convert_to_feature(sentence, label_ids, args)
        query_features_input_ids.append(torch.tensor(feature.input_ids))
        query_features_token_type_ids.append(torch.tensor(feature.token_type_ids))
        query_features_attention_mask.append(torch.tensor(feature.attention_mask))
        query_features_label_ids.append(torch.tensor(feature.label_ids))

    query_features_input_ids = torch.stack(query_features_input_ids)
    query_features_token_type_ids = torch.stack(query_features_token_type_ids)
    query_features_attention_mask = torch.stack(query_features_attention_mask)
    query_features_label_ids = torch.stack(query_features_label_ids)

    bert_encoder_outputs = \
        bert_encoder_pt(
            input_ids=query_features_input_ids.to(args.device),
            token_type_ids=query_features_token_type_ids.to(args.device),
            attention_mask=query_features_attention_mask.to(args.device),
            output_hidden_states=True
        )
    # print((torch.sum(torch.stack(bert_encoder_outputs.hidden_states[-4:]), 0) / 4).size())

    bert_encoder_output = torch.sum(torch.stack(bert_encoder_outputs.hidden_states[-4:]), 0) / 4
    bert_output_raw_flatten = torch.flatten(bert_encoder_output, start_dim=0, end_dim=1)[:]
    # print(bert_output_raw_flatten.size())
    labels_flatten = torch.tensor(torch.flatten(query_features_label_ids, start_dim=0, end_dim=1)[:].numpy().tolist())[:]
    # print(labels_flatten.size())
    filtered_indices = torch.where(labels_flatten >= 0)[0].cpu().numpy().tolist()
    filtered_bert_output = bert_output_raw_flatten[filtered_indices]
    # print(filtered_bert_output.size())

    all_filtered_bert_output_raw_flatten = []
    index_now = 0
    for sentence in query_sentences:
        len_sen = len(sentence)
        filtered_bert_output_raw_flatten = filtered_bert_output[index_now:index_now + len_sen]
        all_filtered_bert_output_raw_flatten.append(filtered_bert_output_raw_flatten)
        index_now += len_sen


    for i, (span_pred, filtered_bert_output_raw_flatten) in enumerate(
            tqdm(zip(span_preds, all_filtered_bert_output_raw_flatten))):
        if span_pred == []:
            continue

        for span in span_pred:
            span_emb = torch.mean(filtered_bert_output_raw_flatten[span["start"]:span["end"] + 1], 0)
            if args.use_type_name:
                cat_span_emb = torch.cat((span_emb, span_emb), dim=-1)
                logit = torch.matmul(all_proto_emb, cat_span_emb)
            else:
                logit = torch.matmul(all_proto_emb, span_emb)

            tmp_all_logits.append(logit)

    logits = torch.stack(tmp_all_logits)
    logits_numpy = logits.detach().cpu().numpy()
    logits_normalize = F.normalize(logits, p=2, dim=0)
    raw_preds = torch.argmax(logits_normalize, -1).cpu().numpy().tolist()
    if args.filter:
        for logit_numpy, pred in zip(logits_numpy, raw_preds):
            if max(logit_numpy) / 2 > span_threshold:
                preds.append(pred)
            else:
                preds.append(-1)
    else:
        preds = raw_preds


    preds_label = []
    for item in preds:
        if item < 0:
            preds_label.append(-1)
        else:
            preds_label.append(label_types_id[item])

    idx_now = 0

    # all_preds_label corresponds to the unfolded "span_preds".
    # for example, span_preds=[[{"strat":1,"end":2},{"strat":3,"end":5}],], then all_preds_label=[[2,1],]
    all_preds_label = []

    for i, span_pred in enumerate(span_preds):
        tmp = []
        for j in range(idx_now, idx_now + len(span_pred)):
            tmp.append(preds_label[j])
        all_preds_label.append(tmp)
        idx_now += len(span_pred)

    stage_2_inference_time = time.time() - stage2_test_time_a
    print(' stage2_test_time_b - stage2_test_time_a', stage_2_inference_time)

    return all_preds_label, stage_2_inference_time


def cal_f1(preds_label, span_preds, query_labels_ids):
    tp = 0
    num_pred = 0
    num_gold = 0
    for pred_label, span_pred, query_label_id in zip(preds_label, span_preds, query_labels_ids):

        span_label_gold = extract_entity_span_label(query_label_id)

        span_label_pred = []
        for label, span in zip(pred_label, span_pred):
            if label > 0:
                span["label"] = label
                span_label_pred.append(span)

        num_pred += len(span_label_pred)
        num_gold += len(span_label_gold)
        for item in span_label_pred:
            if item in span_label_gold:
                tp += 1

    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = tp / num_pred
        recall = tp / num_gold
        f1 = 2 * precision * recall / (precision + recall)
    print('tp', tp)
    print('num_pred', num_pred)
    print('num_gold', num_gold)

    return f1, precision, recall


def evaluate_episodes(args):
    ################get episodes data ##################
    episodes_data = read_episodes_data_from_file(filepath=args.filepath_target_episodes,
                                                 args=args,
                                                 start=args.test_episodes_num_start,
                                                 end=args.test_episodes_num)
    #####################################################
    metric_stage1_all, metric_all_stages_all, metric_stage1_filtered_all = {}, {}, {}

    # 输出结果的文件
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    results_file = open(args.results_dir + args.dataset_target + args.n_way_k_shot + '.txt', 'a')
    print('----------------------------------------------', file=results_file)
    print('----------------------------------------------', file=results_file)
    print('----------------------------------------------', file=results_file)

    if not args.test_stage2_only:
        ckpt_dir_stage1 = './checkpoint/' \
                          + args.dataset_source \
                          + '-' + args.mode \
                          + '-' + args.type_mode \
                          + '-' + str(args.seed) \
                          + '/stage1/' \
                          + args.IO_mode + '-' + 'bert_model_stage1.ckpt'

        checkpoint_bert_model_stage1 = torch.load(ckpt_dir_stage1, map_location=args.device)
        ModelStage1 = BertModelStage1(args).to(args.device)
        ModelStage1.load_state_dict(checkpoint_bert_model_stage1['model_state_dict'])

    # 加载第二阶段TRAIN 完成的 bert模型
    if not args.test_stage1_only:
        ckpt_dir_stage2 = './checkpoint/' + args.dataset_source \
                          + '-' + args.mode \
                          + '-' + args.type_mode \
                          + '-' + str(args.seed) \
                          + '/stage2/bert_model_stage2.ckpt'

        checkpoint_bert_model_stage2 = torch.load(ckpt_dir_stage2, map_location=args.device)
        ModelStage2 = BertModelStage2(args).to(args.device)
        ModelStage2.load_state_dict(checkpoint_bert_model_stage2['model_state_dict'])

        bert_encoder_pt = ModelStage2.encoder

    # The test is given to the data of one episode at a time,
    # and the model trained on source is loaded before each finetune.
    num_true_stage1 = 0
    num_pred_stage1 = 0
    num_gold_stage1 = 0

    num_true_stage1_filtered = 0
    num_pred_stage1_filtered = 0
    num_gold_stage1_filtered = 0

    num_true_all_stages = 0
    num_pred_all_stages = 0
    num_gold_all_stages = 0

    num_fp_all_stages = 0
    num_fp_span_all_stages = 0
    num_fp_type_all_stages = 0

    num_fn_all_stages = 0
    num_fn_span_all_stages = 0
    num_fn_type_all_stages = 0

    sum_inference_time = 0

    for episode in tqdm(range(args.test_episodes_num - args.test_episodes_num_start)):
        set_seeds(args)
        if args.filter and (not args.test_stage1_only):
            # Only if a filtering strategy is used and not only testing stage 1, the ModelStage2 will be loaded again
            ModelStage2.linear_layer = nn.Linear(args.pretrained_model_hidden_size, args.source_class_num).to(
                args.device)
            ModelStage2.load_state_dict(checkpoint_bert_model_stage2['model_state_dict'])

        results_file = open(args.results_dir + args.dataset_target + args.n_way_k_shot + '.txt', 'a')

        print(args.dataset_target + args.n_way_k_shot, file=results_file)
        print(args.test_episodes_num_start + episode, 'episode', file=results_file)

        # no need to add args.test_episodes_num_start, because when we read episodes-date(read_episodes_data_from_file),
        # we start from args.test_episodes_num_start
        episode_data = episodes_data[episode]
        span_preds = []

        support_labels_ids = episode_data["support_labels_ids"]
        support_sentences = episode_data["support_sentences"]
        # 原型的获取
        label_dict = {}
        label_types_id = list(set([item for item_list in support_labels_ids for item in item_list]))
        label_types_id.remove(0)
        for i in range(len(label_types_id)):
            label_dict[label_types_id[i]] = i
        if not args.test_stage2_only:
            ######### reload the ModelStage1 ##########################
            ModelStage1.load_state_dict(checkpoint_bert_model_stage1['model_state_dict'])

            span_preds, metric_stage1 = adapt_predict_stage1_episode(args, episode_data, ModelStage1)

            sum_inference_time += metric_stage1["stage_1_inference_time"]

            num_true_stage1 += metric_stage1["num_true"]
            num_pred_stage1 += metric_stage1["num_pred"]
            num_gold_stage1 += metric_stage1["num_gold"]

            precision_stage1 = num_true_stage1 / num_pred_stage1
            recall_stage1 = num_true_stage1 / num_gold_stage1
            f1_stage1 = 2 * precision_stage1 * recall_stage1 / (precision_stage1 + recall_stage1)

            metric_stage1_all = {"f1": f1_stage1, "precision": precision_stage1, "recall": recall_stage1, }
            print('Currently to', args.test_episodes_num_start + episode)
            print('metric_stage1_all: ', metric_stage1_all)
            print('Currently to', args.test_episodes_num_start + episode, file=results_file)
            print('metric_stage1_all: ', metric_stage1_all, file=results_file)

        elif args.test_stage2_only:
            # If only the second stage is tested,
            # then the results of the first stage span detection will be obtained from the label by default
            span_preds = []
            query_labels_ids = episode_data["query_labels_ids"]
            query_label_io = convert_label_id_to_io(query_labels_ids)
            for label_io in query_label_io:
                span_preds.append(extract_entity_span(label_io))

        if not args.test_stage1_only:

            label_dict = {}
            label_types_id = list(set([item for item_list in support_labels_ids for item in item_list]))
            label_types_id.remove(0)
            for i in range(len(label_types_id)):
                label_dict[label_types_id[i]] = i

            if args.filter:
                # stage2_finetune_time_a = time.time()
                ModelStage2 = adapt_stage2(args=args,
                                           ModelStage2=ModelStage2,
                                           sentences_support=support_sentences,
                                           labels_ids_support=support_labels_ids,
                                           label_types_id=label_types_id,
                                           label_dict=label_dict,
                                           )

                # stage2_finetune_time_b = time.time()
                # print('stage2_finetune_time_b - stage2_finetune_time_a', stage2_finetune_time_b - stage2_finetune_time_a)

                bert_encoder_pt = ModelStage2.encoder
                all_proto_emb_support = get_original_prototypes(args,
                                                                ModelStage2.encoder,
                                                                support_sentences,
                                                                support_labels_ids,
                                                                label_dict, label_types_id)
                if args.use_type_name:
                    all_label_emb = get_proxy_label_emb(args, ModelStage2, label_types_id)
                    all_proto_emb = torch.cat((all_label_emb, all_proto_emb_support), dim=-1)
                    span_threshold = cal_span_threshold(args=args,
                                                        ModelStage2=ModelStage2,
                                                        all_label_emb=all_label_emb,
                                                        sentences_support=support_sentences,
                                                        labels_ids_support=support_labels_ids,
                                                        label_types_id=label_types_id,
                                                        label_dict=label_dict,
                                                        )
                else:
                    all_proto_emb = all_proto_emb_support
                    span_threshold = 0
            elif not args.filter:
                all_proto_emb_support = get_original_prototypes(args, bert_encoder_pt, support_sentences,
                                                                support_labels_ids,
                                                                label_dict, label_types_id)
                if args.use_type_name:
                    all_label_emb = get_proxy_label_emb(args, ModelStage2, label_types_id)
                    all_proto_emb = torch.cat((all_label_emb, all_proto_emb_support), dim=-1)
                else:
                    all_proto_emb = all_proto_emb_support
                span_threshold = 0

            query_sentences = episode_data["query_sentences"]
            query_labels_ids = episode_data["query_labels_ids"]

            if args.test_stage2_only:
                span_threshold = 0

            preds_label, stage_2_inference_time = predict_stage2_episode(args,
                                                                         bert_encoder_pt,
                                                                         all_proto_emb,
                                                                         span_preds,
                                                                         query_sentences,
                                                                         query_labels_ids,
                                                                         label_dict,
                                                                         label_types_id,
                                                                         span_threshold=span_threshold)
            sum_inference_time += stage_2_inference_time

            for pred_label, span_pred, query_label_id in zip(preds_label, span_preds, query_labels_ids):

                span_label_gold = extract_entity_span_label(query_label_id)

                span_label_pred = []
                for label, span in zip(pred_label, span_pred):
                    if label > 0:
                        span["label"] = label
                        span_label_pred.append(span)

                num_pred_all_stages += len(span_label_pred)
                num_gold_all_stages += len(span_label_gold)
                for item in span_label_pred:
                    if item in span_label_gold:
                        num_true_all_stages += 1

                # error analysis part
                spans_pred = [{"start": item["start"], "end": item["end"]} for item in span_label_pred]
                spans_gold = [{"start": item["start"], "end": item["end"]} for item in span_label_gold]
                # print(spans_pred)
                # print(spans_gold)
                for item in span_label_pred:
                    if item not in span_label_gold:
                        num_fp_all_stages += 1
                        span_item = {"start": item["start"], "end": item["end"]}
                        if span_item in spans_gold:
                            num_fp_type_all_stages += 1
                        else:
                            num_fp_span_all_stages += 1

                for item in span_label_gold:
                    if item not in span_label_pred:
                        num_fn_all_stages += 1
                        span_item = {"start": item["start"], "end": item["end"]}
                        if span_item in spans_pred:
                            # 因为type错了没召回
                            num_fn_type_all_stages += 1
                        else:
                            # 因为span错了没召回
                            num_fn_span_all_stages += 1

                if args.filter:
                    copy_span_pred = []
                    copy_span_gold = []
                    for pred in span_label_pred:
                        copy_span_pred.append({"start": pred["start"], "end": pred["end"]})
                    for pred in span_label_gold:
                        copy_span_gold.append({"start": pred["start"], "end": pred["end"]})

                    num_pred_stage1_filtered += len(copy_span_pred)
                    num_gold_stage1_filtered += len(copy_span_gold)

                    for pred in copy_span_pred:
                        if pred in copy_span_gold:
                            num_true_stage1_filtered += 1

            precision_all_stages = num_true_all_stages / num_pred_all_stages
            recall_all_stages = num_true_all_stages / num_gold_all_stages
            f1_all_stages = 2 * precision_all_stages * recall_all_stages / (precision_all_stages + recall_all_stages)

            metric_all_stages_all = {"f1": f1_all_stages, "precision": precision_all_stages,
                                     "recall": recall_all_stages, }
            print('Currently to', args.test_episodes_num_start + episode)
            print('metric_all_stages_all: ', metric_all_stages_all)
            print('Currently to', args.test_episodes_num_start + episode, file=results_file)
            print('metric_all_stages_all: ', metric_all_stages_all, file=results_file)

            print('sum_inference_time', sum_inference_time)
            print('episodes', episode + 1)

            if args.filter:
                precision_stage1_filtered = num_true_stage1_filtered / num_pred_stage1_filtered
                recall_stage1_filtered = num_true_stage1_filtered / num_gold_stage1_filtered
                f1_stage1_filtered = 2 * precision_stage1_filtered * recall_stage1_filtered / (
                        precision_stage1_filtered + recall_stage1_filtered)

                metric_stage1_filtered_all = {"f1": f1_stage1_filtered, "precision": precision_stage1_filtered,
                                              "recall": recall_stage1_filtered, }
                print('Currently to', args.test_episodes_num_start + episode)
                print('metric_stage1_filtered_all: ', metric_stage1_filtered_all)
                print('Currently to', args.test_episodes_num_start + episode, file=results_file)
                print('metric_stage1_filtered_all: ', metric_stage1_filtered_all, file=results_file)

    if args.test_stage2_only:
        results_metric_file = open(args.results_dir + 'test_stage2_only-fewnerd-results_metric.txt', 'a')
        print('####################################################################', file=results_metric_file)
        print(args.dataset_target + args.n_way_k_shot, file=results_metric_file)
        print('episodes: ', args.test_episodes_num - args.test_episodes_num_start, file=results_metric_file)
        print('metric_all_stages_all: ', metric_all_stages_all, file=results_metric_file)
        print('mode: ', args.mode, file=results_metric_file)
        return None, metric_all_stages_all, None

    results_metric_file = open(args.results_dir + 'fewnerd-results_metric.txt', 'a')
    print('####################################################################', file=results_metric_file)
    print(args.dataset_target + args.n_way_k_shot, file=results_metric_file)
    print('IO_mode: ', args.IO_mode, file=results_metric_file)
    print('mode: ', args.mode, file=results_metric_file)
    print('seed: ', args.seed, file=results_metric_file)

    print('type_mode: ', args.type_mode, file=results_metric_file)

    print('episodes: ', args.test_episodes_num - args.test_episodes_num_start, file=results_metric_file)
    print('Filter: ', args.filter, file=results_metric_file)

    print('Adapt Stage2 finetune_target_epochs: ', args.finetune_target_epochs_stage2, file=results_metric_file)

    print('metric_stage1_all: ', metric_stage1_all, file=results_metric_file)
    print('metric_all_stages_all: ', metric_all_stages_all, file=results_metric_file)

    print('num_fp_all_stages: ', num_fp_all_stages, file=results_metric_file)
    print('num_fp_span_all_stages: ', num_fp_span_all_stages, file=results_metric_file)
    print('num_fp_type_all_stages: ', num_fp_type_all_stages, file=results_metric_file)
    print('num_fn_all_stages: ', num_fn_all_stages, file=results_metric_file)
    print('num_fn_span_all_stages: ', num_fn_span_all_stages, file=results_metric_file)
    print('num_fn_type_all_stages: ', num_fn_type_all_stages, file=results_metric_file)

    if args.filter:
        print('metric_stage1_filtered_all: ', metric_stage1_filtered_all, file=results_metric_file)
    print('####################################################################', file=results_metric_file)

    return metric_stage1_all, metric_all_stages_all, metric_stage1_filtered_all


def cal_span_threshold(args, ModelStage2, all_label_emb, sentences_support, labels_ids_support, label_types_id,
                       label_dict):
    bert_encoder_pt = ModelStage2.encoder
    all_max_logit = []
    all_max_logit_o_tokens = []
    for sentence, label_ids in zip(sentences_support, labels_ids_support):
        feature = convert_to_feature(sentence, label_ids, args)
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

        filtered_indices_o = torch.where(labels_flatten == 0)[0].cpu().numpy().tolist()
        filtered_bert_output_raw_flatten_o = bert_output_raw_flatten[filtered_indices_o]
        logits_o_tokens = torch.matmul(filtered_bert_output_raw_flatten_o, all_label_emb.T).detach().cpu().numpy()
        max_logit_o_tokens = np.max(logits_o_tokens, axis=-1).tolist()
        all_max_logit_o_tokens.extend(max_logit_o_tokens)

        spans_gold = extract_entity_span_label(label_ids)

        for span in spans_gold:
            words_emb = filtered_bert_output_raw_flatten[span["start"]:span["end"] + 1]
            label_emb = all_label_emb[label_dict[span["label"]]]
            logit = torch.matmul(words_emb, label_emb).detach().cpu().numpy().tolist()
            all_max_logit.extend(logit)

    all_max_logit = np.array(all_max_logit)

    if args.dataset_target in ['I2B2']:
        # Because the entities in I2B2 are too sparse, we use the mean,
        # and for the rest we still take the min calculation
        span_threshold = np.mean(all_max_logit)
    else:
        span_threshold = np.min(all_max_logit)

    return span_threshold


def evaluate_cross_domain(args):
    support_sentences_samples, support_labels_samples = read_cross_domain_target_support_data_from_file(args)

    query_sentences, query_labels = read_conll2003_format_data_from_file(args.filepath_target, args.dataset_target)

    raw_size_dataset = len(query_sentences)
    query_sentences = query_sentences[:min(5000, raw_size_dataset)]
    query_labels = query_labels[:min(5000, raw_size_dataset)]

    # dev_labels is empty(because it is the same as train_labels) in cross-domain setting
    strict_range = [i for i in range(len(args.id2label_train), len(args.id2label_train) + len(args.id2label_test))]

    query_labels_ids = convert_label_to_id(query_labels, args, strict_range=strict_range)

    # 输出结果的文件
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if not os.path.exists(args.predict_results_dir):
        os.makedirs(args.predict_results_dir)

    results_file_path = args.results_dir + args.dataset_target + str(args.k_shot) + '.txt'
    with open(results_file_path, 'w') as f:
        f.write('')
    results_file = open(results_file_path, 'a')

    if args.test_stage2_only:
        results_file_path = args.results_dir + args.dataset_target + str(args.k_shot) + 'test_stage2_only.txt'
        with open(results_file_path, 'w') as f:
            f.write('')
        results_file = open(results_file_path, 'a')

    predict_results_file_path = args.predict_results_dir + args.dataset_target + str(args.k_shot) + '.txt'
    with open(predict_results_file_path, 'w') as f:
        f.write('')
    predict_results_file = open(predict_results_file_path, 'a')

    if not args.test_stage2_only:
        ckpt_dir_stage1 = './checkpoint/' \
                          + args.dataset_source \
                          + '-' + args.mode \
                          + '-' + args.type_mode \
                          + '-' + str(args.seed) \
                          + '/stage1/' \
                          + args.IO_mode + '-' + 'bert_model_stage1.ckpt'
        checkpoint_bert_model_stage1 = torch.load(ckpt_dir_stage1, map_location=args.device)
        ModelStage1 = BertModelStage1(args).to(args.device)
        ModelStage1.load_state_dict(checkpoint_bert_model_stage1['model_state_dict'])
    if not args.test_stage1_only:
        # 加载第二阶段TRAIN 完成的模型
        ckpt_stage2_path = './checkpoint/' \
                           + args.dataset_source \
                           + '-' + args.mode \
                           + '-' + args.type_mode \
                           + '-' + str(args.seed) \
                           + '/stage2/' + 'bert_model_stage2.ckpt'
        checkpoint_bert_model_stage2 = torch.load(ckpt_stage2_path, map_location=args.device)
        ModelStage2 = BertModelStage2(args).to(args.device)
        ModelStage2.load_state_dict(checkpoint_bert_model_stage2['model_state_dict'])
        # bert_encoder_pt_stage2 = ModelStage2.encoder

    print('----------------------------------------------', file=results_file)
    print('----------------------------------------------', file=results_file)
    print('if filter or not', str(args.filter), file=results_file)

    print('mode', str(args.mode), file=results_file)
    print('stage2_use_mlp', str(args.stage2_use_mlp), file=results_file)

    print('----------------------------------------------', file=results_file)

    all_metric_stage1 = []
    all_metric_all_stages = []
    all_metric_stage1_filtered = []

    for support_sentences_sample, support_labels_sample in zip(support_sentences_samples, support_labels_samples):
        num_true_stage1_filtered = 0
        num_pred_stage1_filtered = 0
        num_gold_stage1_filtered = 0

        num_true_all_stages = 0
        num_pred_all_stages = 0
        num_gold_all_stages = 0

        set_seeds(args)

        support_labels_ids_sample = convert_label_to_id(support_labels_sample, args, strict_range=strict_range)

        if not args.test_stage2_only:
            ModelStage1 = BertModelStage1(args).to(args.device)
            ModelStage1.load_state_dict(checkpoint_bert_model_stage1['model_state_dict'])
            span_preds, metric_stage1 \
                = adapt_predict_stage1_cross_domain(args=args,
                                                    bert_model_stage1=ModelStage1,
                                                    support_sentences_sample=support_sentences_sample,
                                                    support_labels_ids_sample=support_labels_ids_sample,
                                                    query_sentences=query_sentences,
                                                    query_labels_ids=query_labels_ids,
                                                    )
            all_metric_stage1.append(metric_stage1)
            print('precision of span detection', metric_stage1["precision"],
                  'recall of span detection', metric_stage1["recall"],
                  'f1 of span detection', metric_stage1["f1"])
            print('precision of span detection', metric_stage1["precision"],
                  'recall of span detection', metric_stage1["recall"],
                  'f1 of span detection', metric_stage1["f1"], file=results_file)
            del ModelStage1

        elif args.test_stage2_only:
            span_preds = []
            query_labels_io = convert_label_id_to_io(query_labels_ids)
            for label_io in query_labels_io:
                if sum(label_io) > 0:
                    span_preds.append(extract_entity_span(label_io))
                else:
                    span_preds.append([])

        if not args.test_stage1_only:

            label_dict = {}
            label_types_id = list(set([item for item_list in support_labels_ids_sample for item in item_list]))
            label_types_id.remove(0)
            for i in range(len(label_types_id)):
                label_dict[label_types_id[i]] = i

            if args.adapt_stage2:
                # 只要需要微调时才需要重新加载ModelStage2
                ModelStage2 = BertModelStage2(args).to(args.device)
                ModelStage2.load_state_dict(checkpoint_bert_model_stage2['model_state_dict'])
                if args.use_type_name:
                    ModelStage2 = adapt_stage2(args=args,
                                               ModelStage2=ModelStage2,
                                               sentences_support=support_sentences_sample,
                                               labels_ids_support=support_labels_ids_sample,
                                               label_types_id=label_types_id,
                                               label_dict=label_dict,
                                               )
                # else:
                #     ModelStage2 = adapt_stage2_no_type_name(args=args,
                #                                             ModelStage2=ModelStage2,
                #                                             sentences_support=support_sentences_sample,
                #                                             labels_ids_support=support_labels_ids_sample,
                #                                             label_types_id=label_types_id,
                #                                             label_dict=label_dict,
                #                                             )
            else:
                pass

            ModelStage2.eval()
            all_proto_emb_support = get_original_prototypes(args,
                                                            ModelStage2.encoder,
                                                            support_sentences_sample,
                                                            support_labels_ids_sample,
                                                            label_dict,
                                                            label_types_id)

            if args.use_type_name:
                all_label_emb = get_proxy_label_emb(args, ModelStage2, label_types_id)
                all_proto_emb = torch.cat((all_label_emb, all_proto_emb_support), dim=-1)
            else:
                all_proto_emb = all_proto_emb_support

            if args.filter:
                if args.use_type_name:
                    span_threshold = cal_span_threshold(args=args,
                                                        ModelStage2=ModelStage2,
                                                        all_label_emb=all_label_emb,
                                                        sentences_support=support_sentences_sample,
                                                        labels_ids_support=support_labels_ids_sample,
                                                        label_types_id=label_types_id,
                                                        label_dict=label_dict,
                                                        )
                else:
                    span_threshold = 0
            elif not args.filter:
                # if not filter, span_threshold=0
                span_threshold = 0

            if args.test_stage2_only:
                span_threshold = 0
            preds_label, stage_2_inference_time = predict_stage2_cross_domain(args,
                                                                              ModelStage2.encoder,
                                                                              all_proto_emb,
                                                                              span_preds,
                                                                              query_sentences,
                                                                              query_labels_ids,
                                                                              label_dict,
                                                                              label_types_id,
                                                                              span_threshold=span_threshold, )

            for pred_label, span_pred, query_label_id in zip(preds_label, span_preds, query_labels_ids):

                span_label_gold = extract_entity_span_label(query_label_id)

                span_label_pred = []
                for label, span in zip(pred_label, span_pred):
                    if label > 0:
                        span["label"] = label
                        span_label_pred.append(span)

                num_pred_all_stages += len(span_label_pred)
                num_gold_all_stages += len(span_label_gold)
                for item in span_label_pred:
                    if item in span_label_gold:
                        num_true_all_stages += 1

                if args.filter:
                    copy_span_pred = []
                    copy_span_gold = []
                    for pred in span_label_pred:
                        copy_span_pred.append({"start": pred["start"], "end": pred["end"]})
                    for pred in span_label_gold:
                        copy_span_gold.append({"start": pred["start"], "end": pred["end"]})

                    num_pred_stage1_filtered += len(copy_span_pred)
                    num_gold_stage1_filtered += len(copy_span_gold)

                    for pred in copy_span_pred:
                        if pred in copy_span_gold:
                            num_true_stage1_filtered += 1

            precision_all_stages = num_true_all_stages / num_pred_all_stages
            recall_all_stages = num_true_all_stages / num_gold_all_stages
            f1_all_stages = 2 * precision_all_stages * recall_all_stages / (precision_all_stages + recall_all_stages)

            metric_all_stages = {"f1": f1_all_stages, "precision": precision_all_stages, "recall": recall_all_stages, }
            print('metric_all_stages_all: ', metric_all_stages)
            print('metric_all_stages_all: ', metric_all_stages, file=results_file)
            all_metric_all_stages.append(metric_all_stages)

            if args.filter:
                precision_stage1_filtered = num_true_stage1_filtered / num_pred_stage1_filtered
                recall_stage1_filtered = num_true_stage1_filtered / num_gold_stage1_filtered
                f1_stage1_filtered = 2 * precision_stage1_filtered * recall_stage1_filtered / (
                        precision_stage1_filtered + recall_stage1_filtered)
                metric_stage1_filtered = {"f1": f1_stage1_filtered, "precision": precision_stage1_filtered,
                                          "recall": recall_stage1_filtered, }

                print('metric_stage1_filtered: ', metric_stage1_filtered)
                print('metric_stage1_filtered: ', metric_stage1_filtered, file=results_file)
                all_metric_stage1_filtered.append(metric_stage1_filtered)
    if args.test_stage2_only:
        results_metric_file = open(args.results_dir + 'test_stage2_only-domain_transfer_results_metric.txt', 'a')
        print(args.dataset_target + str(args.k_shot), file=results_metric_file)
        print('IO_mode: ', args.IO_mode, file=results_metric_file)
        print('mode: ', args.mode, file=results_metric_file)

        num_samples = len(all_metric_all_stages)
        mean_f1_all_stages = sum([item["f1"] for item in all_metric_all_stages]) / num_samples
        std_f1_all_stages = np.std([item["f1"] for item in all_metric_all_stages])

        mean_precision_all_stages = sum([item["precision"] for item in all_metric_all_stages]) / num_samples
        std_precision_all_stages = np.std([item["precision"] for item in all_metric_all_stages])

        mean_recall_all_stages = sum([item["recall"] for item in all_metric_all_stages]) / num_samples
        std_recall_all_stages = np.std([item["recall"] for item in all_metric_all_stages])

        mean_metric_all_stages = {"f1": mean_f1_all_stages,
                                  "precision": mean_precision_all_stages,
                                  "recall": mean_recall_all_stages,
                                  "std_f1": std_f1_all_stages,
                                  "std_precision": std_precision_all_stages,
                                  "std_recall": std_recall_all_stages,
                                  }

        print('mean_metric_only_stage2', mean_metric_all_stages, file=results_metric_file)
        return all_metric_all_stages

    results_metric_file = open(args.results_dir + 'domain_transfer_results_metric.txt', 'a')
    print('####################################################################', file=results_metric_file)
    print(args.dataset_target + str(args.k_shot), file=results_metric_file)
    print('IO_mode: ', args.IO_mode, file=results_metric_file)
    print('mode: ', args.mode, file=results_metric_file)
    print('type_mode: ', args.type_mode, file=results_metric_file)

    print('Filter: ', args.filter, file=results_metric_file)
    print('Adapt Stage1: ', args.adapt_stage1, file=results_metric_file)
    print('Adapt Stage2: ', args.adapt_stage2, file=results_metric_file)

    print('Adapt Stage1 finetune_target_epochs: ', args.finetune_target_epochs_stage1, file=results_metric_file)
    print('Adapt Stage2 finetune_target_epochs: ', args.finetune_target_epochs_stage2, file=results_metric_file)

    print('Adapt Stage2 finetune_target_threshold_stage2: ', args.finetune_target_threshold_stage2,
          file=results_metric_file)

    num_samples = len(all_metric_stage1)

    mean_f1_stage1 = sum([item["f1"] for item in all_metric_stage1]) / num_samples
    std_f1_stage1 = np.std([item["f1"] for item in all_metric_stage1])

    mean_precision_stage1 = sum([item["precision"] for item in all_metric_stage1]) / num_samples
    std_precision_stage1 = np.std([item["precision"] for item in all_metric_stage1])

    mean_recall_stage1 = sum([item["recall"] for item in all_metric_stage1]) / num_samples
    std_recall_stage1 = np.std([item["recall"] for item in all_metric_stage1])

    mean_metric_stage1 = {"f1": mean_f1_stage1,
                          "precision": mean_precision_stage1,
                          "recall": mean_recall_stage1,
                          "std_f1": std_f1_stage1,
                          "std_precision": std_precision_stage1,
                          "std_recall": std_recall_stage1,
                          }

    mean_f1_all_stages = sum([item["f1"] for item in all_metric_all_stages]) / num_samples
    std_f1_all_stages = np.std([item["f1"] for item in all_metric_all_stages])

    mean_precision_all_stages = sum([item["precision"] for item in all_metric_all_stages]) / num_samples
    std_precision_all_stages = np.std([item["precision"] for item in all_metric_all_stages])

    mean_recall_all_stages = sum([item["recall"] for item in all_metric_all_stages]) / num_samples
    std_recall_all_stages = np.std([item["recall"] for item in all_metric_all_stages])

    mean_metric_all_stages = {"f1": mean_f1_all_stages,
                              "precision": mean_precision_all_stages,
                              "recall": mean_recall_all_stages,
                              "std_f1": std_f1_all_stages,
                              "std_precision": std_precision_all_stages,
                              "std_recall": std_recall_all_stages,
                              }

    print('mean_metric_stage1', mean_metric_stage1, file=results_metric_file)
    print('mean_metric_all_stages', mean_metric_all_stages, file=results_metric_file)

    if args.filter:
        mean_f1_stage1_filtered = sum([item["f1"] for item in all_metric_stage1_filtered]) / num_samples
        std_f1_stage1_filtered = np.std([item["f1"] for item in all_metric_stage1_filtered])

        mean_precision_stage1_filtered = sum([item["precision"] for item in all_metric_stage1_filtered]) / num_samples
        std_precision_stage1_filtered = np.std([item["precision"] for item in all_metric_stage1_filtered])

        mean_recall_stage1_filtered = sum([item["recall"] for item in all_metric_stage1_filtered]) / num_samples
        std_recall_stage1_filtered = np.std([item["recall"] for item in all_metric_stage1_filtered])

        mean_metric_stage1_filtered = {"f1": mean_f1_stage1_filtered,
                                       "precision": mean_precision_stage1_filtered,
                                       "recall": mean_recall_stage1_filtered,
                                       "std_f1": std_f1_stage1_filtered,
                                       "std_precision": std_precision_stage1_filtered,
                                       "std_recall": std_recall_stage1_filtered,
                                       }

        print('mean_metric_stage1_filtered', mean_metric_stage1_filtered, file=results_metric_file)

    print('####################################################################', file=results_metric_file)

    return all_metric_all_stages


def evaluate(args):
    all_f1 = []
    if args.dataset_target == 'FEW-NERD-INTRA' or args.dataset_target == 'FEW-NERD-INTER':
        metric_stage1_all, metric_all_stages_all, metric_stage1_filtered_all = evaluate_episodes(args)
    # Domain Transfer setting
    else:
        all_f1 = evaluate_cross_domain(args)

    return all_f1

    # metric_all_stages = evaluate_episodes_dual_loss(args)
    # return metric_all_stages
