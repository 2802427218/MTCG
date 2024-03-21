from ..model import MTCGModel
import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import json
import wandb
import random


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
    parser.add_argument('--model_dir', default='')
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--latent_size", type=int, default=768)
    parser.add_argument("--latent_num", type=int, default=1)
    parser.add_argument("--seq_len_per_latent", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_fix", action="store_true")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--contrasitive_loss", type=float, default=None)
    parser.add_argument("--sparse_loss", type=float, default=None)
    parser.add_argument("--latent_classify_loss", type=float, default=None)
    parser.add_argument("--aspect_gap_loss", type=float, default=None)
    parser.add_argument("--variation", type=float, default=0)
    parser.add_argument("--classifier_head_num", type=int, default=3)
    parser.add_argument("--classifier_class_num_per_head", type=str, default='[2,2,4]')
    parser.add_argument("--classifier_mid_size", type=int, default=128)
    parser.add_argument("--classifier_head_type", type=str, default='multiple', choices=('single', 'multiple'))
    parser.add_argument("--aspect_gap_head_num", type=int, default=3)
    parser.add_argument("--aspect_gap_amplification", type=int, default=10)
    args = parser.parse_args()
    return args


def load_models_and_tokenizers(args):

    encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
    encoder = BertModel.from_pretrained(args.pretrained_encoder)
    decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
    decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    return encoder, decoder, encoder_tokenizer, decoder_tokenizer


def main():
    pass

if __name__ == '__main__':
    main()
