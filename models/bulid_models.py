""" NoisyNN in PyTorch
'NoisyNN: Exploring the Influence of Information Entropy Change in Learning Systems'
- https://arxiv.org/pdf/2309.10625v2.pdf
"""
import torch
# from timm.models.vision_transformer import VisionTransformer
from timm.models import register_model
from .vit import VisionTransformer


def quality_matrix(k, alpha=0.3):
    """r
    Quality matrix Q. Described in the eq (17) so that eps = QX, where X is the input.
    Alpha is 0.3, as mentioned in Appendix D.
    """
    identity = torch.diag(torch.ones(k))
    shift_identity = torch.zeros(k, k)
    for i in range(k):
        shift_identity[(i + 1) % k, i] = 1
    opt = -alpha * identity + alpha * shift_identity
    return opt


def optimal_quality_matrix(k):
    """r
    Optimal Quality matrix Q. Described in the eq (19) so that eps = QX, where X is the input.
    Suppose 1_(kxk) is torch.ones
    """
    return torch.diag(torch.ones(k)) * -k / (k + 1) + torch.ones(k, k) / (k + 1)



class NoisyViT(VisionTransformer):
    """r
    Args:
        optimal: Determine the linear transform noise is produced by the quality matrix or the optimal quality matrix.
        res: Inference resolution. Ensure the aspect ratio = 1
    """

    def __init__(self, optimal: bool, img_size: int, patch_size: int, **kwargs):
        self.stage3_res = img_size // patch_size

        if optimal:
            linear_transform_noise = optimal_quality_matrix(self.stage3_res)
        else:
            linear_transform_noise = quality_matrix(self.stage3_res)
        super().__init__(**kwargs)

        self.linear_transform_noise = torch.nn.Parameter(linear_transform_noise, requires_grad=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            return super().forward_features(x)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # Add noise only when training
        if self.training:
            x = self.blocks[:-1](x)
            # Suppose the token dim = 1
            token = x[:, 0, :].unsqueeze(1)
            x = x[:, 1:, :].permute(0, 2, 1)
            B, C, L = x.shape
            x = x.reshape(B, C, self.stage3_res, self.stage3_res).contiguous()
            x = self.linear_transform_noise @ x + x
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat([token, x], dim=1)
            x = self.blocks[-1](x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x


@register_model
def NoisyViT_base_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model = NoisyViT(optimal=True,
                     img_size=224,
                     patch_size=16,
                     embed_dim=768,
                     depth=12,
                     num_heads=12,
                     representation_size=None,
                     **kwargs)
    return model


@register_model
def NoisyViT_base_patch16_224_in21k(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                               has_logits: bool = True, **kwargs):
    model = NoisyViT(optimal=True,
                     img_size=224,
                     patch_size=16,
                     embed_dim=768,
                     depth=12,
                     num_heads=12,
                     representation_size=768 if has_logits else None,
                     **kwargs)
    return model


@register_model
def NoisyViT_base_patch32_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = NoisyViT(optimal=True,
                     img_size=224,
                     patch_size=32,
                     embed_dim=768,
                     depth=12,
                     num_heads=12,
                     representation_size=None,
                     **kwargs)
    return model


@register_model
def NoisyViT_base_patch32_224_in21k(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                               has_logits: bool = True, **kwargs):
    model = NoisyViT(optimal=True,
                     img_size=224,
                     patch_size=32,
                     embed_dim=768,
                     depth=12,
                     num_heads=12,
                     representation_size=768 if has_logits else None,
                     **kwargs)
    return model


@register_model
def NoisyViT_large_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = NoisyViT(optimal=True,
                     img_size=224,
                     patch_size=16,
                     embed_dim=1024,
                     depth=24,
                     num_heads=16,
                     representation_size=None,
                     **kwargs)
    return model


@register_model
def NoisyViT_large_patch16_224_in21k(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                                has_logits: bool = True, **kwargs):
    model = NoisyViT(optimal=True,
                     img_size=224,
                     patch_size=16,
                     embed_dim=1024,
                     depth=24,
                     num_heads=16,
                     representation_size=1024 if has_logits else None,
                     **kwargs)
    return model


@register_model
def NoisyViT_large_patch32_224_in21k(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                                has_logits: bool = True, **kwargs):
    model = NoisyViT(optimal=True,
                     img_size=224,
                     patch_size=32,
                     embed_dim=1024,
                     depth=24,
                     num_heads=16,
                     representation_size=1024 if has_logits else None,
                     **kwargs)
    return model


@register_model
def NoisyViT_huge_patch14_224_in21k(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                               has_logits: bool = True, **kwargs):
    model = NoisyViT(optimal=True,
                     img_size=224,
                     patch_size=14,
                     embed_dim=1280,
                     depth=32,
                     num_heads=16,
                     representation_size=1280 if has_logits else None,
                     **kwargs)
    return model


# Easy test
if __name__ == '__main__':
    from torchinfo import summary

    net = NoisyViT_base_patch16_224(num_classes=5)
    summary(net, input_size=(1, 3, 224, 224))