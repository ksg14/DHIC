from torch import nn

from .model_utils import PatchEmbedding
from .encoder import TransformerEncoder

class VisionTransformerEncoder(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        )

# patches_embedded = PatchEmbedding()(x)
# TransformerEncoderBlock()(patches_embedded).shape