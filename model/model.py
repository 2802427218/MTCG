import torch
import torch.nn as nn
import wandb


class MTCGModel(nn.Module):
    """
        Encoder-Decoder架构模型
    """

    def __init__(self, encoder, decoder, args):
        super(MTCGModel, self).__init__()

        # 超参数
        self.args = args

        # 编码器
        self.encoder = encoder
        self.encoder_config = encoder.config

        # 潜在空间参数
        self.latent_size = args.latent_size
        self.seq_len_per_latent = args.seq_len_per_latent
        self.latent_num = args.latent_num
        self.seq_len = self.latent_num * self.seq_len_per_latent

        # 解码器
        self.decoder = decoder
        self.decoder_config = decoder.config
