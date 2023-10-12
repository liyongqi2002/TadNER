import os

import torch
from tqdm import trange, tqdm

from transformers import get_linear_schedule_with_warmup

from model import BertModelStage1, BertModelStage2
from utils import convert_label_to_id, read_conll2003_format_data_from_file, convert_label_id_to_io, GetDataLoader


def train_stage1(args):
    # source数据处理
    sentences_train, labels_train = read_conll2003_format_data_from_file(args.filepath_source_train,
                                                                         args.dataset_source)

    labels_ids_train = []
    if args.dataset_target in ['FEW-NERD-INTRA', 'FEW-NERD-INTER']:
        labels_ids_train = convert_label_to_id(labels_train, args)
    elif args.dataset_target in ['WNUT17', 'CONLL2003', 'I2B2', 'GUM']:
        strict_range = [i for i in range(args.source_class_num)]
        labels_ids_train = convert_label_to_id(labels_train, args, strict_range=strict_range)

    dataloader_source_train = GetDataLoader(args=args,
                                            sentences=sentences_train,
                                            labels_ids=labels_ids_train,
                                            batch_size=args.batch_size_source,
                                            ignore_o_sentence=True)

    bert_model_stage1 = BertModelStage1(args).to(args.device)
    optimizer_stage1 = torch.optim.Adam(bert_model_stage1.parameters(), lr=args.train_source_LR_stage1)

    num_train_epochs = args.train_source_epochs_stage1
    num_update_steps_per_epoch = len(dataloader_source_train)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer_stage1,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    train_source_iterator = trange(0, 1, desc="train_source_epochs_stage1",
                                   disable=False)
    for epoch_iter in train_source_iterator:

        bert_model_stage1.train()
        batch_iterator_stage1 = tqdm(dataloader_source_train, desc="batch_iterator_stage1", disable=False)

        for step, batch_stage1 in enumerate(batch_iterator_stage1):
            optimizer_stage1.zero_grad()
            # print(batch_stage1)

            loss_stage1, _1, _2 = \
                bert_model_stage1(
                    input_ids=batch_stage1[0].to(args.device),
                    token_type_ids=batch_stage1[1].to(args.device),
                    attention_mask=batch_stage1[2].to(args.device),
                    label_ids=batch_stage1[3].to(args.device),
                )
            # compute gradient and do step
            loss_stage1.backward()
            optimizer_stage1.step()
            lr_scheduler.step()

    # due to the domain gap, validation needs very long time and useless,
    # thus we directly save the model after one epoch.
    ckpt_dir = './checkpoint/' \
               + args.dataset_source \
               + '-' + args.mode \
               + '-' + args.type_mode \
               + '-' + str(args.seed) + '/stage1/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'model_state_dict': bert_model_stage1.state_dict()},
               os.path.join(ckpt_dir,
                            args.IO_mode + '-' + 'bert_model_stage1.ckpt'))
    pass


def train_stage2(args):
    # read sentence and corresponding labels from the file
    sentences_train, labels_train = read_conll2003_format_data_from_file(filepath=args.filepath_source_train,
                                                                         data_name=args.dataset_source)
    if args.dataset_target in ['FEW-NERD-INTRA', 'FEW-NERD-INTER']:
        labels_ids_train = convert_label_to_id(labels=labels_train,
                                               args=args,
                                               strict_range=None)
    elif args.dataset_target in ['WNUT17', 'CONLL2003', 'I2B2', 'GUM']:
        # we need this strict range due to the overlapping of labels between the source domain and target domains
        strict_range = [i for i in range(args.source_class_num)]
        labels_ids_train = convert_label_to_id(labels=labels_train,
                                               args=args,
                                               strict_range=strict_range)

    dataloader_source_train = GetDataLoader(args,
                                            sentences_train,
                                            labels_ids_train,
                                            batch_size=args.batch_size_source,
                                            ignore_o_sentence=True)

    bert_model_stage2 = BertModelStage2(args).to(args.device)
    optimizer = torch.optim.Adam(bert_model_stage2.parameters(), lr=args.train_source_LR_stage2)

    num_train_epochs = args.train_source_epochs_stage2
    num_update_steps_per_epoch = len(dataloader_source_train)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    train_source_iterator = trange(0, int(args.train_source_epochs_stage2), desc="train_source_epochs_stage2",
                                   disable=False)

    for epoch_iter in train_source_iterator:
        bert_model_stage2.train()
        batch_iterator = tqdm(dataloader_source_train, desc="batch_iterator", disable=False)

        for step, batch in enumerate(batch_iterator):
            optimizer.zero_grad()
            loss = \
                bert_model_stage2(
                    input_ids=batch[0].to(args.device),
                    token_type_ids=batch[1].to(args.device),
                    attention_mask=batch[2].to(args.device),
                    label_ids=batch[3].to(args.device),
                    finetune=False,
                )

            # compute gradient and do step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    ckpt_dir = './checkpoint/' \
               + args.dataset_source \
               + '-' + args.mode \
               + '-' + args.type_mode \
               + '-' + str(args.seed) + '/stage2/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # we directly save the model after one epoch
    torch.save({'model_state_dict': bert_model_stage2.state_dict()}, os.path.join(ckpt_dir, 'bert_model_stage2.ckpt'))
    pass


def train(args):
    if not args.test_stage2_only:
        train_stage1(args)
    if not args.test_stage1_only:
        train_stage2(args)
    pass
