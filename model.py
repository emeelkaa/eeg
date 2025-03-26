import torch
import torch.nn.functional as F 
from torch import nn 
from torch import Tensor
from typing import Tuple, Union, Optional

from einops import rearrange
from einops.layers.torch import Reduce

class TSception(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel, stride, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel, stride=stride),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
            nn.Dropout(self.dropout_p)
        )
    
    def __init__(
        self,
        emb_size: int,
        input_size: Tuple[int, int, int],
        sampling_rate: int,
        dropout_p = 0.3

    ):
        super().__init__()
        self.inception_windows = [0.5, 0.25, 0.125]
        self.pool = 8
        self.dropout_p = dropout_p
        self.temporal1 = self.conv_block(1, emb_size, (1, int(self.inception_windows[0] * sampling_rate)), 1, self.pool)
        self.temporal2 = self.conv_block(1, emb_size, (1, int(self.inception_windows[1] * sampling_rate)), 1, self.pool)
        self.temporal3 = self.conv_block(1, emb_size, (1, int(self.inception_windows[2] * sampling_rate)), 1, self.pool)   

        self.spatial1 = self.conv_block(emb_size, emb_size, (int(input_size[1]), 1), 1, int(self.pool*0.25))
        self.spatial2 = self.conv_block(emb_size, emb_size, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool*0.25))

        self.bn_temporal = nn.BatchNorm2d(emb_size)

        self.fusion_layer = self.conv_block(
                in_channels=emb_size,
                out_channels=emb_size,
                kernel=(3, 1),
                stride=1,
                pool=4,
        )

        self.bn_spatial = nn.BatchNorm2d(emb_size)
        self.bn_fusion = nn.BatchNorm2d(emb_size)

    def forward(self, x: Tensor) -> Tensor:
        y = self.temporal1(x)
        out = y
        y = self.temporal2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.temporal3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.bn_temporal(out)

        z = self.spatial1(out)
        out_ = z 
        z = self.spatial2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.bn_spatial(out_)
        
        out = self.fusion_layer(out)
        out = self.bn_fusion(out)
        
        # Reshape output
        out = rearrange(out, 'b c (h) (t) -> b (h t) c')
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)  # [batch, seq_len, emb_size] -> [batch, num_heads, seq_len, head_dim]
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # [batch, num_heads, query_len, key_len]
        
        # Apply Mask (if provided)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")    # [batch, seq_len, emb_size]
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):    
    def __init__(self, emb_size: int, expansion: float, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, int(emb_size * expansion)),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(int(emb_size * expansion), emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):    
    def __init__(self, emb_size: int, num_heads: int, drop_p: float = 0.2):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=4.0, drop_p=drop_p),
                nn.Dropout(drop_p),
            ))
        )

class TransformerEncoder(nn.Sequential):    
    def __init__(self, depth: int, emb_size: int, num_heads: int):
        super().__init__(
            *[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)]
        )  
         
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes: int = 1):
        super().__init__()
        self.classifier = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )        

    def forward(self, x):
        out = self.classifier(x)
        return out

class Conformer(nn.Sequential):
    def __init__(self, 
                 emb_size: int = 40, 
                 depth: int = 2, 
                 n_classes: int = 1,
                 input_size: Tuple[int, int, int] = (1, 18, 2560),
                 sampling_rate: int = 256,
                 num_heads: int = 4,
    ):
        super().__init__()
        self.model = nn.Sequential(
            TSception(emb_size, input_size, sampling_rate),
            TransformerEncoder(depth, emb_size, num_heads),
            ClassificationHead(emb_size, n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


if __name__=="__main__":
    model = Conformer()
    x = torch.randn(1, 1, 18, 2560)  
    output = model(x)
    print(output.shape)



