import torch
from torch import nn
from transformers import BertModel
import torch.nn.functional as F

from utils import calculate_ce_loss


class BertModelStage1(nn.Module):
    def __init__(self, args):
        super(BertModelStage1, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(args.pretrained_model)

        self.linear_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(args.pretrained_model_hidden_size, len(self.args.IO_mode)),
        )
        print('IO_mode length', len(self.args.IO_mode))

    def convert_label_id_to_bioes(self, label_ids):
        # 输入实体标签类别【0,12,12,0,45,0,0,46,46,46,0,0】，输出【0,1,3,0,4,0,0,1,2,3,0,0】
        # 写一段python代码，其中O对应于0，B对应于1，I对应于2，E对应于3，S对应于4
        output = []
        prev_label = None
        for idx, label_id in enumerate(label_ids):
            if label_id == 0:
                if prev_label is not None:
                    # revise the former one
                    output[idx - 1] = 3  # E -> 3

                output.append(0)  # O -> 0
                prev_label = None
            elif label_id != 0:
                if label_id == prev_label:
                    output.append(2)  # I -> 2
                else:
                    output.append(1)  # B -> 1

                    if prev_label is not None:
                        output[idx - 1] = 3  # E -> 3
                prev_label = label_id
        if prev_label is not None:
            output[-1] = 3  # E -> 3

        # 将单个的3转化为4
        label_bieos = []
        prev_label = None
        for idx, label_id in enumerate(output):
            if label_id == 0:
                label_bieos.append(0)
                prev_label = None
            elif label_id != 0:
                if label_id == 3 and prev_label is None:
                    label_bieos.append(4)
                else:
                    label_bieos.append(label_id)
                prev_label = label_id

        if label_bieos[-1] == 2:  # 最后一个是2-I，就应该转为3-E
            label_bieos[-1] = 3  # E -> 3
        elif label_bieos[-1] == 1:  # 最后一个是1，就应该转为4-S
            label_bieos[-1] = 4  # E -> 3

        return label_bieos

    def convert_label_id_to_bio(self, label_ids):
        # 写一段python代码，其中O对应于0，B对应于1，I对应于2
        label_bieos = self.convert_label_id_to_bioes(label_ids)
        # 将4->1,3->2
        label_bio = []
        for idx, label in enumerate(label_bieos):
            if label == 4:
                label_bio.append(1)
            elif label == 3:
                label_bio.append(2)
            else:
                label_bio.append(label)

        return label_bio

    def convert_label_id_to_io(self, label_ids):
        label_io = []
        for idx, label in enumerate(label_ids):
            if label > 0:
                label_io.append(1)
            else:
                label_io.append(0)
        return label_io

    def extract_entity_span_label_BIO(tags):
        """
        :param labels_id: [B-PER,I-PER,0]
        :return: [{"start":0,"end":1,"label":PER}]
        """

        spans_label = []
        entity_start = None
        entity_label = None
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                # 开始新的实体
                if entity_start is not None:
                    # 上一个实体还未结束，先将其添加到列表中
                    entity_end = i - 1
                    spans_label.append({"start": entity_start, "end": entity_end, "label": entity_label})
                entity_start = i
                entity_label = tag[2:]
            elif tag.startswith('I-'):
                # 实体内部
                if entity_start is None:
                    # 非法的标签序列，直接跳过。指的是O, I-ORG这种
                    continue
                if entity_label != tag[2:]:
                    # 非法的标签序列，将前面已有的作为一个预测。指的是B-LOC, I-ORG这种
                    entity_end = i - 1  # 最后一个实体的i - 1
                    spans_label.append({"start": entity_start, "end": entity_end, "label": entity_label})
                    entity_start = None
            else:
                # 标签为O，表示实体结束
                if entity_start is not None:
                    entity_end = i - 1  # 最后一个实体的i - 1
                    spans_label.append({"start": entity_start, "end": entity_end, "label": entity_label})
                    entity_start = None
                    entity_label = None
        if entity_start is not None:
            # 最后一个实体还未结束，将其添加到列表中
            entity_end = len(tags) - 1
            spans_label.append({"start": entity_start, "end": entity_end, "label": entity_label})
        return spans_label

    def extract_bioes_to_span(self, label_ids):
        spans = []
        entity_start = None
        for idx, tag in enumerate(label_ids):
            if tag == 1:  # B-
                # 开始新的实体
                entity_start = idx
            elif tag == 2:  # I-
                # 实体内部
                continue
            elif tag == 3:  # E-
                if entity_start is not None:
                    spans.append({"start": entity_start, "end": idx})
                entity_start = None
            elif tag == 4:  # S-
                spans.append({"start": idx, "end": idx})
                entity_start = None
        return spans

    def extract_bio_to_span(self, label_ids):
        spans = []
        entity_start = None

        for idx, tag in enumerate(label_ids):
            if tag == 1:
                # 开始新的实体
                if entity_start is not None:
                    # 上一个实体还未结束，先将其添加到列表中
                    entity_end = idx - 1
                    spans.append({"start": entity_start, "end": entity_end})
                entity_start = idx
            elif tag == 2:
                continue
            else:
                # 标签为O，表示实体结束
                if entity_start is not None:
                    entity_end = idx - 1  # 最后一个实体的i - 1
                    spans.append({"start": entity_start, "end": entity_end})
                    entity_start = None
        if entity_start is not None:
            # 最后一个实体还未结束，将其添加到列表中
            entity_end = len(label_ids) - 1
            spans.append({"start": entity_start, "end": entity_end})
        return spans

    def extract_io_to_span(self, label_io_list):
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

    def decode_label_ids(self, label_ids):
        if self.args.IO_mode == 'BIOES':
            spans = self.extract_bioes_to_span(label_ids)
        elif self.args.IO_mode == 'BIO':
            spans = self.extract_bio_to_span(label_ids)
        elif self.args.IO_mode == 'IO':
            spans = self.extract_io_to_span(label_ids)
        return spans

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, label_ids=None):
        bert_output_raw = \
            self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, )[0]
        logits = self.linear_layer(bert_output_raw)

        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=1)[:]

        label_ids_flatten = torch.flatten(label_ids, start_dim=0, end_dim=1)[:]

        # filter out those masked tokens
        filtered_indices = torch.where(label_ids_flatten >= 0)[0].cpu().numpy().tolist()

        filtered_logits_flatten = logits_flatten[filtered_indices]

        filtered_label_ids_flatten = label_ids_flatten[filtered_indices]
        if self.args.IO_mode == 'BIOES':
            converted_label_ids_for_stage1 = self.convert_label_id_to_bioes(filtered_label_ids_flatten)
        elif self.args.IO_mode == 'BIO':
            converted_label_ids_for_stage1 = self.convert_label_id_to_bio(filtered_label_ids_flatten)
        elif self.args.IO_mode == 'IO':
            converted_label_ids_for_stage1 = self.convert_label_id_to_io(filtered_label_ids_flatten)

        loss = calculate_ce_loss(filtered_logits_flatten,
                                 torch.tensor(converted_label_ids_for_stage1).to(self.args.device),
                                 weight=None)

        return loss, filtered_logits_flatten, converted_label_ids_for_stage1


class BertModelStage2(nn.Module):
    def __init__(self, args):
        super(BertModelStage2, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(args.pretrained_model)
        if self.args.traditional_contrastive:
            # in this mode, we add a mlp following previous supervised contrastive method to avoid model collapse
            self.mlp_pair_contrastive = nn.Sequential(
                nn.ReLU(),
                nn.Linear(args.pretrained_model_hidden_size, args.pretrained_model_hidden_size),
            )

        # we add a linear layer to work like parameters
        self.linear_layer = nn.Linear(args.pretrained_model_hidden_size, args.source_class_num)

        if self.args.stage2_use_mlp:
            self.mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(args.pretrained_model_hidden_size, args.pretrained_model_hidden_size),
            )

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, label_ids=None,
                finetune=False):
        bert_outputs_raw = \
            self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        bert_output_raw = bert_outputs_raw[0]

        # label_id of O-tokens will be -1 and be filtered later
        label_ids = label_ids - 1

        # (batch_size,n,768)->(batch_size*n,768), n is the length of padded sentence
        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        label_ids_flatten = torch.flatten(label_ids, start_dim=0, end_dim=1)[:]

        # we only select those label_id>=0 (filtering out masked tokens and non-entity tokens)
        filtered_indices = torch.where(label_ids_flatten >= 0)[0].cpu().numpy().tolist()
        filtered_bert_output_raw_flatten = bert_output_raw_flatten[filtered_indices]
        filtered_label_ids_flatten = label_ids_flatten[filtered_indices]

        if self.args.use_type_name:
            if finetune:
                labels_emb = self.linear_layer.weight
                words_emb = filtered_bert_output_raw_flatten
                logits = torch.matmul(words_emb, labels_emb.T)
                loss = calculate_ce_loss(logits=logits,
                                         label_ids=filtered_label_ids_flatten,
                                         weight=None)
                return loss
            else:
                # get labels_emb
                labels = self.args.id2proxy_label_train
                labels_last_hidden_states = []
                for label in labels:
                    input_ids = self.args.tokenizer.encode(label, add_special_tokens=True)
                    input_ids = torch.tensor([input_ids]).to(self.args.device)
                    last_hidden_states = self.encoder(input_ids)[0]
                    last_hidden_states = last_hidden_states.squeeze(0)
                    if self.args.stage2_use_mlp:
                        labels_last_hidden_states.append(self.mlp(last_hidden_states[0]))
                    else:
                        # use the [CLS] output for representing this label
                        labels_last_hidden_states.append(last_hidden_states[0])

                labels_emb = torch.stack(labels_last_hidden_states).to(self.args.device)

                words_emb = filtered_bert_output_raw_flatten
                words_corresponding_label_emb = labels_emb[filtered_label_ids_flatten]

                loss = self.calculate_type_aware_contrastive_loss(words_emb=words_emb,
                                                                  words_corresponding_label_emb=words_corresponding_label_emb,
                                                                  label_ids=filtered_label_ids_flatten)

                return loss
        else:
            if self.args.virtual_proxy:
                # virtual_labels
                # in this mode, we use random tensor to replace labels_emb for comparison
                virtual_labels_emb = self.linear_layer.weight
                words_emb = filtered_bert_output_raw_flatten
                words_corresponding_label_emb = virtual_labels_emb[filtered_label_ids_flatten]
                loss = self.calculate_type_aware_contrastive_loss(words_emb=words_emb,
                                                                  words_corresponding_label_emb=words_corresponding_label_emb,
                                                                  label_ids=filtered_label_ids_flatten)
                return loss

            elif self.args.traditional_contrastive:
                # in this mode, we calculate traditional supervised contrastive loss for comparison to our methods
                # we didn't pay much attention on it due to its bad performance
                filtered_bert_output_raw_flatten = self.mlp_pair_contrastive(bert_output_raw_flatten[filtered_indices])

                loss = self.calculate_CONTaiNER_contrastive_loss(features=filtered_bert_output_raw_flatten,
                                                                 labels=filtered_label_ids_flatten,
                                                                 args=self.args
                                                                 )
                return loss

    def calculate_type_aware_contrastive_loss(self, words_emb, words_corresponding_label_emb, label_ids):
        num_words = len(label_ids)
        pos_words_labels = torch.eq(label_ids.unsqueeze(1).repeat(1, num_words),
                                    label_ids.unsqueeze(0).repeat(num_words, 1)
                                    ).float().to(self.args.device)

        labels_words_emb = torch.cat((words_corresponding_label_emb, words_emb), dim=-1)
        words_labels_emb = torch.cat((words_emb, words_corresponding_label_emb), dim=-1)
        logits = torch.matmul(labels_words_emb, words_labels_emb.T)
        logits = F.normalize(logits, p=2, dim=0)
        logits = logits / torch.tensor(0.05)

        softmax_logits = torch.softmax(logits, dim=-1)
        log_softmax_logits = torch.log(softmax_logits)

        lines_loss = -torch.mean(log_softmax_logits * pos_words_labels, dim=-1)
        loss = torch.sum(lines_loss)

        return loss

    def calculate_CONTaiNER_contrastive_loss(self, features, labels, args, temperature=1):
        """
        calculate traditional supervised contrastive loss for comparison
        Reference: https://github.com/HobbitLong/SupContrast
        """
        diagonal = torch.eye(labels.shape[0], dtype=torch.bool).float().to(args.device)
        mask_label_equal = torch.eq(labels, labels.T).float().to(args.device)
        positive_mask = mask_label_equal - diagonal  # 1 only when label is same(not include itself)
        negtive_mask = 1. - mask_label_equal

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)  # 计算两两样本间点乘相似度
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # for every row, the num of positive pairs
        num_positives_per_row = torch.sum(positive_mask, dim=1)

        denominator = torch.sum(exp_logits * negtive_mask, dim=1, keepdim=True) + torch.sum(exp_logits * positive_mask,
                                                                                            dim=1, keepdim=True)

        log_probs = logits - torch.log(denominator)
        log_probs = torch.sum(log_probs * positive_mask, dim=1)[num_positives_per_row > 0] / num_positives_per_row[
            num_positives_per_row > 0]
        loss = -log_probs
        loss = loss.mean()

        return loss
