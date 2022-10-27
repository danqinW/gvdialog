import argparse
from email.policy import default
import numpy as np
import random
import torch
from base_trainer import Trainer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name", default='gvdialog', 
        choices=['recosa', 'hsan', 'hred', 'gvdialog'])
    parser.add_argument('--task', type=str, help="task name", default='gv', choices=['base', 'gv'])
    parser.add_argument('--corpus-path', type=str, help="path to dialog dataset", default='preprocessed_data/douban')
    parser.add_argument('--separator', type=str, help="symbol to split sentences in conversations", default=' __eou__ ')
    parser.add_argument('--vocab-path', type=str, help="path to vocabulary", default='cache/cornell')
    parser.add_argument('--cachedir', type=str, help="path to data cache", default='cache/cornell')
    parser.add_argument('--save-cache', action='store_true', help='cache processed data', default=True)
    parser.add_argument('--tokenizer-type', type=str, help="tokenizer to use", default='gpt2',
        choices=['gpt2','spacy', 'zh', 'base'])
    parser.add_argument('--load-ckpt', type=str, help='path to pretrained model checkpoint', default='')
    parser.add_argument('--save-ckpt', type=str, help='path to save model checkpoint', default='ckpt')
    parser.add_argument('--n-epoch', type=int, help="training model for n epoch", default=50)
    parser.add_argument('--batch_size', type=int, help="mini-batch size", default=4)
    parser.add_argument('--max_seq_len', type=int, help="max sequence length", default=30)
    parser.add_argument('--max_conv_len', type=int, help="max conversations length", default=10)
    parser.add_argument('--vocab_size', type=int, help="set vocabualry size", default=16000)
    parser.add_argument('--hidden_size', type=int, help="hidden embedding size", default=512)
    parser.add_argument('--context_size', type=int, help="context embedding size", default=512)
    parser.add_argument('--n-layers', type=int, help="RNN layers number", default=2)
    parser.add_argument('--d_model', type=int, help="transformer's embedding size", default=512)
    parser.add_argument('--n-head', type=int, help='multihead attention head', default=8)
    parser.add_argument('--dim-feedforward', type=int, help='feed forward embeeding size', default=1024)
    parser.add_argument('--num-encoder-blocks', type=int, help='number of encoder blocks', default=2)
    parser.add_argument('--num-decoder-blocks', type=int, help='number of decoder blocks', default=4)
    parser.add_argument('--use-latent', action='store_true', help='use latent embedding for GVDialog model', default=True)
    parser.add_argument('--latent-size', type=int, help="set latent size", default=64)
    parser.add_argument('--wait-steps',type=int, help="waiting for n steps to add kl loss", default=200)
    parser.add_argument('--kl-annealing', type=int, help="set how many steps to perform kl annealing", default=200)
    parser.add_argument('--word-drop', type=float, help="word drop ratio", default=0.5)
    parser.add_argument('--alpha', type=float, help="weight for KL loss", default=0.01)
    parser.add_argument('--lambda', type=float, help="weight for reconstruction loss", default=0.5)
    parser.add_argument('--chunk-size', type=int, help="chunk size for latent representation", default=1)
    parser.add_argument('--optimizer', type=str, help="optimizer to use", default='sgd')
    parser.add_argument('--act', type=str, help="activation functions to use", default='gelu')
    parser.add_argument('--learning-rate', type=float, help="set learning rate", default=1e-2)
    parser.add_argument('--label-smoothing', type=float, help="label smoothing ratio", default=0.1)
    parser.add_argument('--dropout', type=float, help="dropout ratio", default=0.2)
    parser.add_argument('--disable-scheduler', action='store_true', help="disable learning rate scheduler")
    parser.add_argument('--warmup-ratio', type=float, help="learning rate warm up ratio", default=0.1)
    parser.add_argument('--num-cycles', type=int, help="num cycles for scheduler to restart learning rate", default=1)
    parser.add_argument('--share-embedding', action='store_true', help='if share embedding', default=False)
    parser.add_argument('--norm-first', action='store_true', help='transformer norm first')
    parser.add_argument('--batch-first', action='store_true', help='use batch first')
    parser.add_argument('--print-every', type=int, help="print information every n steps", default=200)
    parser.add_argument('--top-p', type=int, help="nucleus sampling probabilities p", default=0.98)
    parser.add_argument('--top-k', type=int, help="top k sampling", default=5)
    parser.add_argument('--evaluate-every', type=int, help="evaluate model every n steps", default=20)
    parser.add_argument('--random-seed', type=int, help="set random seed", default=2022)

    config = parser.parse_args()
    return config

def main():
    config = get_config()
    # set random seed
    if config.random_seed:
        setup_seed(config.random_seed)
    trainer = Trainer.build_trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
    


