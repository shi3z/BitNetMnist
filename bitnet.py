import torch
from torch import nn
import torch.nn.functional as F

class BitRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BitRMSNorm is equivalent to LlamaRMSNorm and T5LayerNorm
        refers: https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L76C1-L90C59
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        print(d)

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


def activation_quant(x):
   scale = 127.0 / x.abs().max(dim=-1,keepdim=True).values.clamp_(min=1e-5)
   y = (x*scale).round().clamp_(-128,127)/scale
   return y

def weight_quant(w):
   scale = 1.0/w.abs().mean().clamp_(min=1e-5)
   u = (w*scale).round().clamp_(-1,1) / scale
   return u

class BitLinearOriginal(nn.Linear):
   def __init__(self,in_features,out_features,bias=False,flg_before_linear=True,bits=8):
       super(BitLinearOriginal, self).__init__(in_features, out_features, bias)
       self.layernorm = nn.LayerNorm(in_features)
       self.RMSNorm = BitRMSNorm(in_features)
       self.bits = bits
   def forward(self,x):
       w=self.weight
       x_norm = self.RMSNorm(x)
       x_quant = x_norm + (activation_quant(x_norm)-x_norm).detach()
       w_quant = w+(weight_quant(w)-w).detach()
       y = F.linear(x_quant,w_quant)
       return y
