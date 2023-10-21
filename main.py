import argparse
import torch
from transformers import BertTokenizer

from evaluate import evaluate
# from zero_evaluate import evaluate


from train import train
from utils import get_filepath, read_labels_from_file, set_seeds


def parse_boolean_argument(arg, arg_item):
    if arg == 'True':
        return True
    elif arg == 'False':
        return False
    else:
        raise Exception(f'Please assign {arg_item} True or False')


if __name__ == "__main__":
    # 参数设置
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath_source", default=None, type=str, help="source file path of the data")
    parser.add_argument("--filepath_target_episodes", default=None, type=str, help="target file path of the data")

    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length")

    parser.add_argument("--batch_size_source", default=32, type=int, help="source training batch size")

    parser.add_argument("--train_source_LR_stage1", default=3e-5, type=float, help="train_source_LR_stage1")
    parser.add_argument("--finetune_target_LR_stage1", default=3e-5, type=float, help="train_target_LR_stage1")

    parser.add_argument("--train_source_LR_stage2", default=3e-5, type=float, help="train_source_LR_stage2")
    parser.add_argument("--finetune_target_LR_stage2", default=3e-5, type=float, help="finetune_target_LR_stage2")

    parser.add_argument("--finetune_target_threshold_stage2", default=1, type=float, help="finetune_target_threshold_stage2")

    parser.add_argument("--train_source_epochs_stage1", default=1, type=int, help="train_source_epochs_stage1")
    parser.add_argument("--finetune_target_epochs_stage1", default=100, type=int, help="train_target_epochs_stage1")

    parser.add_argument("--train_source_epochs_stage2", default=1, type=int, help="train_source_epochs_stage2")
    parser.add_argument("--finetune_target_epochs_stage2", default=100, type=int,
                        help="Fine tuning epochs for the second stage")

    parser.add_argument("--results_dir", default='./results/', type=str, help="results directory")
    parser.add_argument("--predict_results_dir", default='./predict_results/', type=str,
                        help="predict_results directory")

    #############################################################################################################
    parser.add_argument("--pretrained_model", default='bert-base-uncased', type=str, help="pretrained_model")
    parser.add_argument("--pretrained_model_hidden_size", default=768, type=int, help="hidden_size of PLM")
    #############################################################################################################
    parser.add_argument("--mode", default='use_type_name', type=str,
                        help="use_type_name/virtual_proxy/traditional_contrastive")


    parser.add_argument("--dataset_source", default=None, type=str, help="dataset_source")
    parser.add_argument("--dataset_target", default=None, type=str, help="dataset_target")

    parser.add_argument("--n_way_k_shot", default=None, type=str, help="n_way_k_shot")
    parser.add_argument("--test_episodes_num_start", default=0, type=int, help="test_episodes_num_start")
    parser.add_argument("--test_episodes_num", default=5000, type=int, help="test_episodes_num")

    parser.add_argument("--k_shot", default=None, type=int, help="k_shot")

    parser.add_argument("--train", default=None, type=str, help="train or not")
    parser.add_argument("--test_stage2_only", default='False', type=str, help="test_stage2_only")
    parser.add_argument("--test_stage1_only", default='False', type=str, help="test_stage1_only")

    parser.add_argument("--filter", default='True', type=str, help="filter or not")
    parser.add_argument("--adapt_stage1", default='True', type=str, help="adapt_stage1 or not")
    parser.add_argument("--adapt_stage2", default='True', type=str, help="adapt_stage2 or not")

    parser.add_argument("--stage2_use_mlp", default='False', type=str,
                        help="to examine if mlp layer can help contrastive learning")

    parser.add_argument("--seed", default=999, type=int, help="seed")
    parser.add_argument("--gpu_id", type=int, default=0, help="select on which gpu to train.")

    parser.add_argument("--zero_shot", default='False', type=str, help="zero_shot or not")

    parser.add_argument("--IO_mode", default='IO', type=str, help="IO or BIO or BIOES")

    parser.add_argument("--type_mode", default='original', type=str, help="type_mode, "
                                                                          "original "
                                                                          "or meaningless "
                                                                          "or misleading"
                                                                          "or variant_1")

    #############################################################################################################

    args = parser.parse_args()

    ######################################
    ######################################

    # choose one from the three below
    args.use_type_name = False
    args.virtual_proxy = False
    args.traditional_contrastive = False
    if args.mode == "use_type_name":
        args.use_type_name = True
    elif args.mode == "virtual_proxy":
        args.virtual_proxy = True
    elif args.mode == "traditional_contrastive":
        args.traditional_contrastive = True
    else:
        raise Exception('Please assign one mode use_type_name/virtual_proxy/traditional_contrastive')

    if args.dataset_source is None or args.dataset_target is None:
        raise Exception('Please note what is source domain? what is target domain?')

    if args.dataset_target in ['FEW-NERD-INTRA', 'FEW-NERD-INTER'] and args.n_way_k_shot is None:
        raise Exception('Please note n_way_k_shot in FEW-NERD settings,eg. 5_1')

    if args.dataset_target in ['WNUT17', 'CONLL2003', 'I2B2', 'GUM'] and args.k_shot is None:
        raise Exception('Please note k_shot in Cross-Domain settings eg. 5')

    # default for 1000 episodes in FEW-NERD
    args.test_episodes_num_start = 0
    args.test_episodes_num = 5000

    args.train = parse_boolean_argument(args.train, arg_item='train')

    args.test_stage2_only = parse_boolean_argument(args.test_stage2_only,
                                                   arg_item='test_stage2_only')
    args.test_stage1_only = parse_boolean_argument(args.test_stage1_only,
                                                   arg_item='test_stage1_only')
    args.filter = parse_boolean_argument(args.filter, arg_item='filter')
    args.adapt_stage1 = parse_boolean_argument(args.adapt_stage1, arg_item='adapt_stage1')
    args.adapt_stage2 = parse_boolean_argument(args.adapt_stage2, arg_item='adapt_stage2')
    args.stage2_use_mlp = parse_boolean_argument(args.stage2_use_mlp, arg_item='stage2_use_mlp')

    ############################################################################
    print('*********** reading files path *****************')
    filepath = get_filepath(args)

    file_mapping = {
        0: 'filepath_labels',
        1: 'filepath_source_train',
        2: 'filepath_source_dev',
        3: 'filepath_target_episodes',  # used in FEW-NERD setting
        4: 'filepath_target'  # used in Cross-Domain setting
    }

    for i, file_index in enumerate(range(5)):
        file_name = file_mapping.get(file_index)
        setattr(args, file_name, filepath[i])
    ############################################################################
    print('*********** reading labels from file: ', args.filepath_labels, ' *****************')
    labels_from_file = read_labels_from_file(args.filepath_labels, args)

    label_mapping = {
        0: 'id2label',
        1: 'id2label_train',
        2: 'id2label_dev',
        3: 'id2label_test',
        4: 'label2id',
        5: 'id2proxy_label',
        6: 'id2proxy_label_train',
        7: 'id2proxy_label_dev',
        8: 'id2proxy_label_test',
        9: 'proxy_label2id'
    }

    for i, label_index in enumerate(range(10)):
        label_name = label_mapping.get(label_index)
        setattr(args, label_name, labels_from_file[i])

    # args.source_class_num is only used in stage2, only train.txt is enough
    args.source_class_num = len(args.id2label_train)
    args.target_class_num = len(args.id2label_test)

    ############################################################################
    args.gpu_id = 2
    print('***************** working on gpu id: ', args.gpu_id, ' *****************')
    args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

    args.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    args.n_gpu = 0 if not torch.cuda.is_available else torch.cuda.device_count()
    args.n_gpu = min(1, args.n_gpu)
    set_seeds(args)
    ##########################################################################
    # print(args)
    ##########################################################################

    if args.train:
        train(args)
    else:
        all_f1 = evaluate(args)
