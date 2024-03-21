import torch
import torch.nn as nn
import wandb

import logging

logger = logging.getLogger(__name__)


class MTCGModel(nn.Module):
    """
        Encoder-Decoder架构模型
    """

    def __init__(self, encoder, decoder, args):
        super(MTCGModel, self).__init__()

        # 超参数
        self.args = args
        self.head_num = None
        self.loss_list = None
        self.latent_classify_head = None
        self.aspect_gap_head = None

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
        self.decoder_num_layer = self.decoder_config.n_layer
        self.decoder_hidden_size = self.decoder_config.n_embd
        self.decoder_num_head = self.decoder_config.n_head

        # 连接器
        # 将BERT的输出（即隐藏状态）映射到潜在空间
        self.connector_encoder = nn.Sequential(
            torch.nn.Linear(self.encoder_hidden_size, self.latent_num * self.latent_size),
            torch.nn.Tanh(),
            nn.Dropout(self.decoder_config.attn_pdrop)
        )
        # 将潜在空间的表示转换成GPT-2模型进行解码时所需的表示
        self.connector_decoder = nn.Sequential(
            torch.nn.Linear(self.latent_num * self.latent_size,
                            self.seq_len * self.decoder_num_layer * 2 * self.decoder_hidden_size),
            nn.Dropout(self.decoder_config.attn_pdrop),
            Reshape(
                {'seq_len': self.seq_len, 'num_layer': self.decoder_num_layer, 'hidden_size': self.decoder_hidden_size,
                 'num_head': self.decoder_num_head})
        )

    def fix_decoder(self):
        """
        冻结decoder参数
        """
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def connect(self, encoder_output, variation=0):
        """
        连接编码层和解码层
        """
        tmp_latent = self.connector_encoder(encoder_output)
        eps = torch.zeros_like(tmp_latent).normal_(std=variation).to(tmp_latent.device)
        past_key_values = self.connector_decoder(tmp_latent + eps)
        latent = tmp_latent.view(-1, self.latent_num, self.latent_size)
        return past_key_values, latent

    def sparse_loss(self, latent, dim=None):
        """
        稀疏性损失:增加数据的稀疏性可以帮助提高模型的泛化能力和计算效率。
        """
        if len(latent) == 3 and dim is None:
            raise Exception('Expect latent to be dim 2.')
        loss_func = nn.L1Loss(reduction='mean')
        batch_size = latent.shape[0]
        if dim is not None:
            tmp_latent = latent[:, dim, :].squeeze()
            average = torch.sum(tmp_latent, dim=0) / batch_size
            loss = loss_func(latent, average.expand(batch_size, -1))
        else:
            average = torch.sum(latent, dim=0) / batch_size
            loss = loss_func(latent, average.expand(batch_size, -1))
        return -loss

    def contrasitive_loss(self, latent1, latent2, loss_func=nn.SmoothL1Loss(reduction='mean'), dim=None):
        """
        对比损失:用于增加两个潜在向量的距离，区分不同特征。
        """
        if dim is not None:
            loss = loss_func(latent1[:, dim, :].squeeze(), latent2[:, dim, :].squeeze())
        else:
            loss = loss_func(latent1, latent2)
        return -1 * loss

    def attribute_classify_loss(self, latent, pos_label, neg_labels, head_index=None):
        """
        属性分类损失:用于区分同一方面的各种不同的属性
        """
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)
        if self.latent_classify_head_type == 'single':
            probs = torch.softmax(self.latent_classify_head(latent), dim=-1)
            batch_size, class_num = probs.shape
            loss = 0
            neg_len = neg_labels.shape[-1]

            for i in range(batch_size):
                pos_prob = probs[i, pos_label[i]]
                if pos_prob < 1 / self.head_num:
                    loss += torch.log(pos_prob)
                loss += torch.log(1 - probs[i, neg_labels[i]]).sum()

            return -1 * loss / (batch_size * (neg_len + 1))
        elif self.latent_classify_head_type == 'multiple':
            if head_index is None:
                print("Warning: head_index not set for multiple classifier head, default to 0")
                head_index = 0
            device = latent.device
            logits = self.latent_classify_head[head_index](latent)
            loss = torch.nn.functional.cross_entropy(logits, pos_label.to(device))
            return loss
        else:
            raise Exception('Wrong latent classifier head type.')

    def set_attribute_classify_head(self, head_num=1, class_num_per_head=2, mid_size=128, head_type='single'):
        if head_type == 'single':
            self.latent_classify_head = nn.Sequential(
                nn.Linear(self.latent_num * self.latent_size, mid_size),
                nn.ReLU(),
                nn.Linear(mid_size, class_num_per_head * head_num)
            )
        elif head_type == 'multiple':
            if type(class_num_per_head) is list:
                self.latent_classify_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, head_num)
                    ) for head_num in class_num_per_head]
                )
            else:
                self.latent_classify_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, class_num_per_head)
                    ) for i in range(head_num)]
                )

    def aspect_gap_loss(self, latent, head_index):
        """
        方面分类损失:用于区分不同方面
        """
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)

        mean_latent = torch.mean(latent, dim=0)
        loss = None
        for i in range(self.aspect_head_num):
            if i != head_index and self.aspect_gap_head[i] is not None:
                if loss is None:
                    loss = torch.nn.functional.mse_loss(mean_latent,
                                                        self.aspect_gap_head[i]) * self.aspect_gap_loss_amplification
                else:
                    loss += torch.nn.functional.mse_loss(mean_latent,
                                                         self.aspect_gap_head[i]) * self.aspect_gap_loss_amplification
        self.set_aspect_gap_head(mean_latent, head_index)
        return loss

    def set_aspect_gap_head(self, latent, head_index):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)

        if len(latent.shape) == 2:
            mean_latent = torch.mean(latent.detach(), dim=0)
            if self.aspect_gap_head[head_index] is not None:
                assert self.aspect_gap_head[head_index].shape == mean_latent.shape
            self.aspect_gap_head[head_index] = mean_latent
        elif len(latent.shape) == 1:
            if self.aspect_gap_head[head_index] is not None:
                assert self.aspect_gap_head[head_index].shape == latent.shape
            self.aspect_gap_head[head_index] = latent.detach()

    def set_losslist(self,
                     losslist: dict,
                     latent_classify_args={'head_num': 1, 'class_num_per_head': 2, 'mid_size': 128,
                                           'head_type': 'single'},
                     aspect_gap_args={'head_num': 2, 'amplification': 5}
                     ):
        """
        用于在自动编码器模型中配置不同类型的损失函数和它们的参数。
        例如：{'contrasitive_loss': 0.001, 'sparse_loss': 0.001, 'attribute_classify_loss':0.1, 'aspect_gap_loss':0.1}
        """
        self.loss_list = losslist
        if 'attribute_classify_loss' in losslist:
            self.head_num = 1
            class_num_per_head = 2
            mid_size = 128
            head_type = 'single'
            if latent_classify_args is not None:
                if 'head_num' in latent_classify_args:
                    self.head_num = latent_classify_args['head_num']
                if 'class_num_per_head' in latent_classify_args:
                    class_num_per_head = latent_classify_args['class_num_per_head']
                if 'mid_size' in latent_classify_args:
                    mid_size = latent_classify_args['mid_size']
                if 'head_type' in latent_classify_args:
                    head_type = latent_classify_args['head_type']

            self.set_attribute_classify_head(head_num=self.head_num, class_num_per_head=class_num_per_head,
                                             mid_size=mid_size, head_type=head_type)

            self.latent_classify_head_type = head_type

        if 'aspect_gap_loss' in losslist:
            if 'attribute_classify_loss' in losslist:
                if self.latent_classify_head_type == 'multiple':
                    self.aspect_head_num = self.head_num
                elif self.latent_classify_head == 'single':
                    print('set aspect head num to {aspect_head_num}.')
                    self.aspect_head_num = aspect_gap_args['head_num']
            else:
                print('set aspect head num to {aspect_head_num}.')
                self.aspect_head_num = aspect_gap_args['head_num']

            self.aspect_gap_loss_amplification = aspect_gap_args['amplification']

            self.aspect_gap_head = [None for i in range(self.aspect_head_num)]

    def forward(self,
                encoder_input_ids,
                encoder_attention_mask,
                encoder_token_type_ids,
                decoder_input_ids,
                decoder_attention_mask,
                adv_input_ids=None,
                adv_attention_mask=None,
                adv_token_type_ids=None,
                pos_label=None,
                neg_labels=None,
                variation=None,
                head_index=None
                ):
        """
        前向传播方法，用于计算重建损失。
        参数说明：
            - encoder_input_ids: 编码器的输入ID，来源于BertTokenizer的输出。
            这是一个由文本转换成的数字序列，用于表示输入给BERT模型的文本。
            - encoder_attention_mask: 编码器的注意力遮罩，同样来源于BertTokenizer。
            这个遮罩用于指示哪些部分是真实的输入内容，哪些部分是填充的内容。
            - encoder_token_type_ids: 编码器的token类型ID，用于区分两种不同的序列（例如，在问答任务中区分问题和答案）。
            也是由BertTokenizer生成。
            - decoder_input_ids: 解码器的输入ID，来源于GPT2Tokenizer的输出。
            这是一个由文本转换成的数字序列，用于表示输入给GPT-2模型的文本。
            - decoder_attention_mask: 解码器的注意力遮罩，来源于GPT2Tokenizer。
            这个遮罩同样用于指示输入序列中哪些部分是有效的。
            - adv_input_ids (可选): 对抗性文本的输入ID，用于增强模型的鲁棒性。
            这些ID表示经过特定操作修改过的文本，旨在测试模型对抗性攻击的抵抗能力。
            - adv_attention_mask (可选): 对抗性文本的注意力遮罩。
            - adv_token_type_ids (可选): 对抗性文本的token类型ID。

        该方法会根据这些输入计算模型的重建损失，这个损失用于训练过程中优化模型参数，提高模型的生成质量和鲁棒性。

        """
        if len(encoder_input_ids.shape) == 3:
            encoder_input_ids = encoder_input_ids.view(encoder_input_ids.shape[1], encoder_input_ids.shape[2])
            encoder_attention_mask = encoder_attention_mask.view(encoder_attention_mask.shape[1],
                                                                 encoder_attention_mask.shape[2])
            encoder_token_type_ids = encoder_token_type_ids.view(encoder_token_type_ids.shape[1],
                                                                 encoder_token_type_ids.shape[2])
            decoder_input_ids = decoder_input_ids.view(decoder_input_ids.shape[1], decoder_input_ids.shape[2])
            decoder_attention_mask = decoder_attention_mask.view(decoder_attention_mask.shape[1],
                                                                 decoder_attention_mask.shape[2])
            if head_index is not None:
                head_index = head_index.item()
            if adv_input_ids is not None:
                adv_input_ids = adv_input_ids.view(adv_input_ids.shape[1], adv_input_ids.shape[2])
            if adv_attention_mask is not None:
                adv_attention_mask = adv_attention_mask.view(adv_attention_mask.shape[1], adv_attention_mask.shape[2])
            if adv_token_type_ids is not None:
                adv_token_type_ids = adv_token_type_ids.view(adv_token_type_ids.shape[1], adv_token_type_ids.shape[2])
            if pos_label is not None:
                pos_label = pos_label.view(pos_label.shape[1])
            if neg_labels is not None:
                neg_labels = neg_labels.view(neg_labels.shape[1], neg_labels.shape[2])

        if variation is None:
            variation = self.variation
        batch_size = decoder_input_ids.shape[0]
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(decoder_input_ids.device)
        decoder_attention_mask = torch.cat([infix_attn, decoder_attention_mask], dim=1)

        encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask,
                                      token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output
        past_key_values, latent = self.connect(encoder_output, variation)
        outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
                               labels=decoder_input_ids, past_key_values=past_key_values, return_dict=True)
        lm_loss = outputs.loss

        loss = 0
        loss_detail = {"lm_loss": lm_loss.detach()}
        w = 1
        if self.losslist is not None:
            if 'contrasitive_loss' in self.losslist:
                if adv_input_ids is None:
                    raise Exception('Expect adversarial inputs for contrasitive loss.')
                adv_encoder_output = self.encoder(input_ids=adv_input_ids, attention_mask=adv_attention_mask,
                                                  token_type_ids=adv_token_type_ids, return_dict=True).pooler_output
                adv_latent = self.trans1(adv_encoder_output)
                adv_loss = self.contrasitive_loss(latent, adv_latent)
                # TO DO: change arg `dim' in the future
                loss += adv_loss * self.losslist['contrasitive_loss']
                w -= self.losslist['contrasitive_loss']
                loss_detail["contrasitive_loss"] = adv_loss.detach()

            if 'sparse_loss' in self.losslist:
                spa_loss = self.sparse_loss(latent)
                # TO DO: change arg `dim' in the future
                loss += spa_loss * self.losslist['sparse_loss']
                w -= self.losslist['sparse_loss']
                loss_detail["sparse_loss"] = spa_loss.detach()

            if 'attribute_classify_loss' in self.losslist:
                lac_loss = self.attribute_classify_loss(latent, pos_label, neg_labels, head_index)
                if lac_loss.detach().item() < 0.1:
                    loss += lac_loss * 0.05
                    w -= 0.05
                else:
                    loss += lac_loss * self.losslist['attribute_classify_loss']
                    w -= self.losslist['attribute_classify_loss']
                loss_detail["attribute_classify_loss"] = lac_loss.detach()

            if 'aspect_gap_loss' in self.losslist:
                agp_loss = self.aspect_gap_loss(latent, head_index)
                if agp_loss is not None:
                    loss += agp_loss * self.losslist['aspect_gap_loss']
                    w -= self.losslist['aspect_gap_loss']
                    loss_detail["aspect_gap_loss"] = agp_loss.detach()

            wandb.log(loss_detail)
        if w < 0:
            w = 1

        loss += w * lm_loss

        return loss, latent, loss_detail

    def encode(self,
               encoder_input_ids,
               encoder_attention_mask=None,
               encoder_token_type_ids=None,
               ):
        """
        编码输入文本并获得潜在表示
        """
        device = next(self.parameters()).device
        encoder_input_ids = encoder_input_ids.to(device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)
        if encoder_token_type_ids is not None:
            encoder_token_type_ids = encoder_token_type_ids.to(device)
        encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask,
                                      token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output
        past_key_values, latent = self.connect(encoder_output)
        return latent, encoder_output, past_key_values

    def generate(
            self,
            input_latent,
            input_ids=None,
            attention_mask=None,
            batch_size=None,
            variation=None,
            min_len=30,
            max_len=50,
            do_sample=True,
            topk=5,
            topp=0.9,
            lp=1,
            rp=1.0,
            use_cache=True):
        '''
        给定潜在表示生成文本
        '''
        device = next(self.parameters()).device
        input_latent = input_latent.to(device)

        if len(input_latent.shape) == 3:
            tmp_batch_size, latent_num, latent_size = input_latent.shape
            input_latent = input_latent.view(tmp_batch_size, latent_num * latent_size)
        elif len(input_latent.shape) != 2:
            raise Exception('Shape of input_latent is expected to be [batch_size, latent_num, latent_size] \
                or [batch_size, latent_num * latent_size]')

        if batch_size is None:
            batch_size = input_latent.shape[0]
            if input_ids is not None:
                if input_ids.shape[0] > batch_size:
                    if batch_size == 1:
                        batch_size = input_ids.shape[0]
                        input_latent = input_latent.expand(batch_size, -1)
                    else:
                        raise Exception('Batch size of input_latent and input_ids mismatched')
                elif input_ids.shape[0] < batch_size and input_ids.shape[0] == 1:
                    input_ids = input_ids.expand(batch_size, -1)

        if input_latent.shape[0] < batch_size:
            input_latent.expand(batch_size, -1)

        if variation is not None:
            eps = torch.zeros_like(input_latent).normal_(std=variation).to(input_latent.device)
            input_latent = input_latent + eps

        past_key_values = self.trans2(input_latent)

        if input_ids is None:
            input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]] * batch_size).to(device),
                                              max_length=3, do_sample=True)[:, 1:]
            attention_mask = torch.ones(batch_size, 2).bool()
        else:
            input_ids = input_ids.to(device)
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, 2).bool()

        cur_len = input_ids.shape[1]
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(device)
        attention_mask = torch.cat([infix_attn, attention_mask.to(device)], dim=-1)

        if cur_len < 1:
            raise Exception('input length error')
        if cur_len == 1:
            result = self.decoder.generate(input_ids=input_ids, past=past_key_values, attention_mask=attention_mask,
                                           repetition_penalty=rp,
                                           do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp,
                                           max_length=max_len, min_length=min_len, use_cache=use_cache)
        else:
            past_key_values = self.decoder(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1],
                                           past_key_values=past_key_values, return_dict=True,
                                           use_cache=True).past_key_values
            result = self.decoder.generate(input_ids=input_ids, past=past_key_values, attention_mask=attention_mask,
                                           repetition_penalty=rp,
                                           do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp,
                                           max_length=max_len, min_length=min_len, use_cache=use_cache)

        return result

    def reconstruct(self,
                    encoder_input_ids,
                    decoder_input_ids=None,
                    encoder_attention_mask=None,
                    encoder_token_type_ids=None,
                    decoder_attention_mask=None,
                    do_sample=True,
                    max_len=50,
                    min_len=30,
                    topk=5,
                    topp=0.9,
                    lp=1.0,
                    use_cache=True):
        '''
        重建生成文本
        '''
        device = next(self.parameters()).device
        batch_size = encoder_input_ids.shape[0]
        encoder_input_ids = encoder_input_ids.to(device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)
        if encoder_token_type_ids is not None:
            encoder_token_type_ids = encoder_token_type_ids.to(device)
        encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask,
                                      token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output

        past_key_values, latent = self.connect(encoder_output)
        if decoder_input_ids is None:
            decoder_input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]] * batch_size).to(device),
                                                      max_length=3, do_sample=True)[:, 1:]
            decoder_attention_mask = torch.ones(batch_size, 2).bool()
        else:
            decoder_input_ids = decoder_input_ids.to(device)
            if decoder_attention_mask is None:
                decoder_attention_mask = torch.ones(batch_size, 2).bool()

        cur_len = decoder_input_ids.shape[1]

        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(device)
        decoder_attention_mask = torch.cat([infix_attn, decoder_attention_mask.to(device)], dim=-1)

        if cur_len < 1:
            raise Exception('input length error')
        if cur_len == 1:
            result = self.decoder.generate(input_ids=decoder_input_ids, past=past_key_values,
                                           attention_mask=decoder_attention_mask,
                                           do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp,
                                           max_length=max_len, min_length=min_len, use_cache=use_cache)
        else:
            past_key_values = self.decoder(input_ids=decoder_input_ids[:, :-1],
                                           attention_mask=decoder_attention_mask[:, :-1],
                                           past_key_values=past_key_values, return_dict=True,
                                           use_cache=True).past_key_values
            result = self.decoder.generate(input_ids=decoder_input_ids, past=past_key_values,
                                           attention_mask=decoder_attention_mask,
                                           do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp,
                                           max_length=max_len, min_length=min_len, use_cache=use_cache)
        return result


class Reshape(nn.Module):
    """
    将潜在空间的向量转为GPT模型接受的'past_key_values'格式

    `past_key_values`的格式
    `past_key_values`是在使用类似GPT这样的Transformer模型进行序列生成任务时，用于缓存之前计算过的注意力机制的键（key）和值（value）的一种数据结构。这使得模型在生成新的序列部分时，能够有效地复用之前步骤的计算结果，从而提高计算效率和生成质量。
    类型：`Tuple[Tuple[torch.Tensor]]`，是一个元组，其中包含多个元组，每个元组又包含两个PyTorch张量。
    长度：外层元组的长度等于模型配置中的层数（`config.n_layers`），这表示每一层解码器都有对应的键值对。
    内部结构：每个内层元组代表一层解码器中的键（key）和值（value），因此包含两个张量。每个张量的形状为`(batch_size, num_heads, sequence_length, embed_size_per_head)`：
    - `batch_size`：批次大小，代表一次处理多少个序列。
    - `num_heads`：注意力机制的头数，即模型在计算注意力时划分的子空间数量。
    - `sequence_length`：序列长度，代表当前处理的序列的长度。
    - `embed_size_per_head`：每个头的嵌入维度大小，即每个注意力头处理的特征维度。
    """

    def __init__(self, arg_dict):
        super(Reshape, self).__init__()
        self.seq_len = arg_dict['seq_len']
        self.num_layer = arg_dict['num_layer']
        self.hidden_size = arg_dict['hidden_size']
        self.num_head = arg_dict['num_head']

    def forward(self, x):
        batch_size = x.shape[0]
        assert self.hidden_size % self.num_head == 0
        embed_size_per_head = self.hidden_size // self.num_head
        x = x.view(batch_size, self.num_layer, 2, self.num_head, self.seq_len, embed_size_per_head).permute(1, 2, 0, 3,
                                                                                                            4, 5)
        past_key_values = []
        for i in range(self.num_layer):
            past_key_values.append((x[i][0], x[i][1],))
        assert past_key_values[0][0].requires_grad == True
        return tuple(past_key_values)
