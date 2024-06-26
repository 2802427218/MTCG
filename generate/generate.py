import sys

sys.path.append(r'D:Code\GraduationDesign\MTCG')
sys.path.append(r'D:\Code\GraduationDesign\MTCG\model')
import argparse
import json
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
from tqdm import tqdm
from utils import KCenters
from model import MTCGModel


def load_data():
    imdb_dataset = [{'sent': []} for _ in range(2)]
    ag_dataset = [{'sent': []} for _ in range(4)]
    toxic_dataset = [{'sent': []} for _ in range(2)]

    with open('../data/IMDb/IMDb.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            label = int(line[0])
            imdb_dataset[label]['sent'].append(line[1].strip())

    with open('../data/ToxicComment/Toxic.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            label = int(line[0])
            toxic_dataset[label]['sent'].append(line[1].strip())

    with open('../data/AGnews/AG-data.txt', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            label = int(line[0])
            ag_dataset[label]['sent'].append(line[1].strip())

    return imdb_dataset, ag_dataset, toxic_dataset


def prepare_dataloaders(datasets, tokenizer, max_length, batch_size):
    dataloaders = []
    for dataset in datasets:
        mapped_dataset = dataset.map(
            lambda e: tokenizer(e['sent'], max_length=max_length, padding='max_length', truncation=True),
            batched=True
        )
        mapped_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
        dataloader = torch.utils.data.DataLoader(mapped_dataset, batch_size=batch_size)
        dataloaders.append(dataloader)
    return dataloaders


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
    encoder = BertModel.from_pretrained(args.pretrained_encoder)
    decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
    decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    model = MTCGModel(encoder=encoder, decoder=decoder, args=args)
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.eval()

    device = 'cpu' if args.no_cuda else 'cuda'
    model.to(device)

    imdb_dataset, ag_dataset, toxic_dataset = load_data()
    imdb_dataset = [Dataset.from_dict(i) for i in imdb_dataset]
    ag_dataset = [Dataset.from_dict(i) for i in ag_dataset]
    toxic_dataset = [Dataset.from_dict(i) for i in toxic_dataset]

    imdb_dataloader = prepare_dataloaders(imdb_dataset, encoder_tokenizer, args.max_length, 32)
    ag_dataloader = prepare_dataloaders(ag_dataset, encoder_tokenizer, args.max_length, 32)
    toxic_dataloader = prepare_dataloaders(toxic_dataset, encoder_tokenizer, args.max_length, 32)

    # The process of encoding and generating texts goes here.
    # This section has been omitted for brevity since it involves
    # extensive operations with the model and datasets.

    # Please insert the logic for encoding, K-Centers training, and text generation here.

    not_latents = None
    sentiment_latents = {0: None, 1: None}
    topic_latents = {0: None, 1: None, 2: None, 3: None}

    for i in range(2):
        for cnt in tqdm(iter(imdb_dataloader[i])):
            encoder_input_ids = cnt['input_ids']
            encoder_attention_mask = cnt['attention_mask']
            encoder_token_type_ids = cnt['token_type_ids']

            latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask,
                                                                   encoder_token_type_ids)
            if sentiment_latents[i] is None:
                sentiment_latents[i] = latent.squeeze().detach()
            else:
                sentiment_latents[i] = torch.cat((sentiment_latents[i], latent.squeeze().detach()), dim=0)

    for i in range(4):
        for cnt in tqdm(iter(ag_dataloader[i])):
            encoder_input_ids = cnt['input_ids']
            encoder_attention_mask = cnt['attention_mask']
            encoder_token_type_ids = cnt['token_type_ids']

            latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask,
                                                                   encoder_token_type_ids)
            if topic_latents[i] is None:
                topic_latents[i] = latent.squeeze().detach()
            else:
                topic_latents[i] = torch.cat((topic_latents[i], latent.squeeze().detach()), dim=0)

    for cnt in tqdm(iter(toxic_dataloader[1])):
        encoder_input_ids = cnt['input_ids']
        encoder_attention_mask = cnt['attention_mask']
        encoder_token_type_ids = cnt['token_type_ids']

        latent, encoder_output, past_key_values = model.encode(encoder_input_ids, encoder_attention_mask,
                                                               encoder_token_type_ids)
        if not_latents is None:
            not_latents = latent.squeeze().detach()
        else:
            not_latents = torch.cat((not_latents, latent.squeeze().detach()), dim=0)

    kcmodel = KCenters(num_centers=args.num_centers, latent_size=args.latent_size,
                       num_output_centers=args.num_output_centers, device='cuda')

    output_text = []
    labels = []

    for i in range(2):
        for j in range(4):
            weight = weight_dict[i][j]
            num_output_centers = args.num_output_centers[i][j]
            print(weight)
            print(num_output_centers)
            centers = kcmodel.train(
                [sentiment_latents[i].to('cuda'), topic_latents[j].to('cuda'), not_latents.to('cuda')],
                weight=weight,
                topk=args.topk,
                SDM_reinit=args.SDM_reinit,
                max_iter=args.max_iter,
                strategy=args.strategy,
                temperature=args.temperature,
                num_output_centers=num_output_centers
            ).cpu().numpy()
            centers = [torch.FloatTensor(k).unsqueeze(0) for k in centers]

            for prompts in tqdm(json.loads(args.pre_tokens)):
                tokens = decoder_tokenizer(prompts, return_tensors='pt')
                input_ids = tokens.input_ids
                attention_mask = tokens.attention_mask
                input_ids = input_ids.expand(args.batch_size, -1)
                attention_mask = attention_mask.expand(args.batch_size, -1)

                output = model.generate(
                    input_latent=random.choice(centers),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    variation=args.variation,
                    max_len=50,
                    rp=1.2
                )

                output_text.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
                labels.extend([[i, j, 1]] * args.batch_size)
                assert len(labels) == len(output_text)

    with open(args.output_dir, 'w') as f:
        for i in tqdm(range(len(output_text))):
            f.write(json.dumps([labels[i], output_text[i]]) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define arguments
    parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--latent_size", type=int, default=768)
    parser.add_argument("--latent_num", type=int, default=1)
    parser.add_argument("--seq_len_per_latent", type=int, default=20)
    parser.add_argument("--model_path", type=str, default='../checkpoints/checkpoint-4500/pytorch_model.bin')
    parser.add_argument("--output_dir", type=str, default="../results/predict_4500.txt")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--pre_tokens", type=str, default=json.dumps([
        'In summary', 'This essay discusses', 'Views on', 'The connection', 'Foundational to this is',
        'To review,', 'In brief,', 'An illustration of', 'Furthermore,', 'The central theme',
        'To conclude,', 'The key aspect', 'Prior to this', 'Emphasised are', 'To summarise',
        'The relationship', 'More importantly,', 'It has been shown', 'The issue focused on', 'In this essay',
        'Once upon a time', 'The book', 'The chicken', 'The city', 'The country',
        'The horse', 'The lake', 'The last time', 'The movie', 'The painting',
        'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910'
    ]))
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variation", type=float, default=1e-3)
    parser.add_argument("--num_centers", type=int, default=1000)
    parser.add_argument("--num_output_centers", type=int, default=10)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--strategy", type=str, default='none', choices=['none', 'weight'])
    parser.add_argument("--temperature", type=float, default=50)
    parser.add_argument("--SDM_reinit", type=bool, default=True)
    parser.add_argument("--weight", type=str, default=json.dumps([1, 5, 1]))
    parser.add_argument("--config", type=str, default="generate_config.json")

    args = parser.parse_args()

    # Load weight from JSON or use default
    weight = json.loads(args.weight)
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
            weight = config.get('weight', weight)
            args.num_output_centers = config.get('num_output_centers', args.num_output_centers)

    if isinstance(weight, dict):
        default_weight = weight['default']
        weight_dict = [[default_weight for jt in range(4)] for it in range(2)]
        for keys in weight:
            if keys != 'default':
                tmp_i = int(keys[0])
                tmp_j = int(keys[1])
                weight_dict[tmp_i][tmp_j] = weight[keys]
    else:
        weight_dict = [[weight for jt in range(4)] for it in range(2)]

    if isinstance(args.num_output_centers, int):
        args.num_output_centers = [[args.num_output_centers] * 4] * 2

    main(args)
