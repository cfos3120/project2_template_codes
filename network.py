'''
this script is for the network of Project 2.

You can change any parts of this code

-------------------------------------------
'''
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
import math
from os.path import join as pjoin
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

# Transformer Module 
# from https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
# implementation of Transformer: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Alexey Dosovitskiy et al. 

class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels,
        norm_layer = None,
        activation_layer = torch.nn.ReLU,
        inplace = True,
        bias: bool = True,
        dropout: float = 0.0
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        #_log_api_usage_once(self)

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer,
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class Network(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size = None,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs = None,
    ):
        super().__init__()
        #_log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        # if conv_stem_configs is not None:
        #     # As per https://arxiv.org/abs/2106.14881
        #     seq_proj = nn.Sequential()
        #     prev_channels = 3
        #     for i, conv_stem_layer_config in enumerate(conv_stem_configs):
        #         seq_proj.add_module(
        #             f"conv_bn_relu_{i}",
        #             Conv2dNormActivation(
        #                 in_channels=prev_channels,
        #                 out_channels=conv_stem_layer_config.out_channels,
        #                 kernel_size=conv_stem_layer_config.kernel_size,
        #                 stride=conv_stem_layer_config.stride,
        #                 norm_layer=conv_stem_layer_config.norm_layer,
        #                 activation_layer=conv_stem_layer_config.activation_layer,
        #             ),
        #         )
        #         prev_channels = conv_stem_layer_config.out_channels
        #     seq_proj.add_module(
        #         "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
        #     )
        #     self.conv_proj: nn.Module = seq_proj
        # else:
        #     self.conv_proj = nn.Conv2d(
        #         in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        #     )

        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

# MLP-Mixer Module 
# from https://github.com/QiushiYang/MLP-Mixer-Pytorch/blob/main/mlp_mixer.py 
# implementation of MLP-Mixer: An all-MLP Architecture for Vision by Ilya Tolstikhin et al. 
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.Dense(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.LayerNorm_0 = nn.LayerNorm(hidden_dim)
        self.token_mixing = MlpBlock(num_tokens, tokens_mlp_dim)
        self.LayerNorm_1 = nn.LayerNorm(hidden_dim)
        self.channel_mixing = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        out = self.LayerNorm_0(x).transpose(1, 2)
        x = x + self.token_mixing(out).transpose(1, 2)
        out = self.LayerNorm_1(x)
        x = x + self.channel_mixing(out)
        return x
    
    def load_from(self, weights, n_block):
        ROOT = f"MixerBlock_{n_block}"
        with torch.no_grad():
            LayerNorm_0_scale = np2th(weights[pjoin(ROOT, 'LayerNorm_0', "scale")]).t()
            LayerNorm_0_bias = np2th(weights[pjoin(ROOT, 'LayerNorm_0', "bias")]).view(-1)
            LayerNorm_1_scale = np2th(weights[pjoin(ROOT, 'LayerNorm_1', "scale")]).t()
            LayerNorm_1_bias = np2th(weights[pjoin(ROOT, 'LayerNorm_1', "bias")]).view(-1)
            
            self.LayerNorm_0.weight.copy_(LayerNorm_0_scale)
            self.LayerNorm_0.bias.copy_(LayerNorm_0_bias)
            self.LayerNorm_1.weight.copy_(LayerNorm_1_scale)
            self.LayerNorm_1.bias.copy_(LayerNorm_1_bias)
            
            
            token_mixing_0_scale = np2th(weights[pjoin(ROOT, 'token_mixing', "Dense_0/kernel")]).t()
            token_mixing_0_bias = np2th(weights[pjoin(ROOT, 'token_mixing', "Dense_0/bias")]).view(-1)
            token_mixing_1_scale = np2th(weights[pjoin(ROOT, 'token_mixing', "Dense_1/kernel")]).t()
            token_mixing_1_bias = np2th(weights[pjoin(ROOT, 'token_mixing', "Dense_1/bias")]).view(-1)
            channel_mixing_0_scale = np2th(weights[pjoin(ROOT, 'channel_mixing', "Dense_0/kernel")]).t()
            channel_mixing_0_bias = np2th(weights[pjoin(ROOT, 'channel_mixing', "Dense_0/bias")]).view(-1)
            channel_mixing_1_scale = np2th(weights[pjoin(ROOT, 'channel_mixing', "Dense_1/kernel")]).t()
            channel_mixing_1_bias = np2th(weights[pjoin(ROOT, 'channel_mixing', "Dense_1/bias")]).view(-1)
            
            if self.token_mixing.Dense[0].weight.shape == token_mixing_0_scale.shape:
                self.token_mixing.Dense[0].weight.copy_(token_mixing_0_scale)
                self.token_mixing.Dense[0].bias.copy_(token_mixing_0_bias)
                self.token_mixing.Dense[2].weight.copy_(token_mixing_1_scale)
                self.token_mixing.Dense[2].bias.copy_(token_mixing_1_bias)
            
            if self.channel_mixing.Dense[0].weight.shape == channel_mixing_0_scale.shape:
                self.channel_mixing.Dense[0].weight.copy_(channel_mixing_0_scale)
                self.channel_mixing.Dense[0].bias.copy_(channel_mixing_0_bias)
                self.channel_mixing.Dense[2].weight.copy_(channel_mixing_1_scale)
                self.channel_mixing.Dense[2].bias.copy_(channel_mixing_1_bias)

class MlpMixer(nn.Module):
    def __init__(self, 
                num_classes=10, 
                num_blocks=12, 
                patch_size=16, 
                hidden_dim=768, 
                tokens_mlp_dim=384, 
                channels_mlp_dim=3072, 
                image_size=224):
        super(MlpMixer, self).__init__()
        num_tokens = (image_size // patch_size)**2

        self.stem = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.MixerBlock = nn.Sequential(*[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.MixerBlock(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    
    def load_from(self, weights):
        with torch.no_grad():
            if self.stem.weight.shape == np2th(weights["stem/kernel"],conv=True).shape:
                self.stem.weight.copy_(np2th(weights["stem/kernel"],conv=True))
            self.ln.weight.copy_(np2th(weights["pre_head_layer_norm/scale"]))
            self.ln.bias.copy_(np2th(weights["pre_head_layer_norm/bias"]))
            
            for bname, block in self.MixerBlock.named_children():
                block.load_from(weights, n_block=bname)

# Custom Warmup Scheduler to go with MLP-Mixer model
# Author: CoinCheung
# Source: https://github.com/CoinCheung/fixmatch-pytorch/blob/master/lr_scheduler.py
class WarmupCosineLrScheduler(_LRScheduler):
    '''
    This is different from official definition, this is implemented according to
    the paper of fix-match
    '''
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio