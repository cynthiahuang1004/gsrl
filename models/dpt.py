import torch
import torch.nn as nn
import torch.nn.functional as F


class SITRMultiScale(nn.Module):
    """
    Wraps a SITR encoder to extract multi-scale features from
    intermediate transformer blocks [3, 6, 9, 12] (1-indexed).

    Output: list of 4 tensors, each (B, num_patches, embed_dim).

    unfreeze_last_n: number of transformer blocks to unfreeze from the end.
        0 = fully frozen (default), 4 = last 4 blocks trainable, etc.
    """

    def __init__(self, sitr, layer_indices=(2, 5, 8, 11),
                 unfreeze_last_n=0):
        super().__init__()
        self.sitr = sitr
        self.layer_indices = sorted(layer_indices)
        self.unfreeze_last_n = unfreeze_last_n

        num_blocks = len(self.sitr.blocks)
        unfreeze_start = num_blocks - unfreeze_last_n

        for p in self.sitr.parameters():
            p.requires_grad = False

        if unfreeze_last_n > 0:
            for i in range(unfreeze_start, num_blocks):
                for p in self.sitr.blocks[i].parameters():
                    p.requires_grad = True
            for p in self.sitr.norm.parameters():
                p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.unfreeze_last_n == 0:
            self.sitr.eval()
        else:
            self.sitr.eval()
            num_blocks = len(self.sitr.blocks)
            for i in range(num_blocks - self.unfreeze_last_n, num_blocks):
                self.sitr.blocks[i].train(mode)
            self.sitr.norm.train(mode)
        return self

    def cache_calibration(self, c):
        """Delegate to underlying SITR encoder."""
        self.sitr.cache_calibration(c)

    def forward(self, x, c):
        s = self.sitr

        x = s.patch_embed(x)
        x = x + s.pos_embed[:, 1:, :]

        cls_token = s.cls_token + s.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        num_patches = s.patch_embed.num_patches

        if s._calib_cache is not None:
            x = torch.cat((x, s._calib_cache.expand(x.shape[0], -1, -1)), dim=1)
        elif s.num_calibration > 0:
            c = s.c_patch_embed(c)
            c = c + s.c_pos_embed
            x = torch.cat((x, c), dim=1)

        num_blocks = len(s.blocks)
        unfreeze_start = num_blocks - self.unfreeze_last_n

        features = []
        for i, blk in enumerate(s.blocks):
            if i < unfreeze_start:
                with torch.no_grad():
                    x = blk(x)
            else:
                x = blk(x)
            if i in self.layer_indices:
                tokens = x[:, 1:num_patches + 1, :]
                features.append(tokens)

        return features


class Reassemble(nn.Module):
    """Per-scale: LayerNorm -> reshape to 2D -> 1x1 conv -> resize."""

    def __init__(self, embed_dim, out_channels, scale_factor):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, tokens):
        B, N, D = tokens.shape
        h = w = int(N ** 0.5)

        tokens = self.norm(tokens)
        x = tokens.permute(0, 2, 1).reshape(B, D, h, w)
        x = self.proj(x)

        if self.scale_factor != 1.0:
            x = F.interpolate(
                x, scale_factor=self.scale_factor,
                mode="bilinear", align_corners=True,
            )
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Fuse skip connection, refine, and upsample 2x."""

    def __init__(self, features):
        super().__init__()
        self.rcu1 = ResidualConvUnit(features)
        self.rcu2 = ResidualConvUnit(features)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class DPTDecoder(nn.Module):
    """
    Dense Prediction Transformer decoder.

    Takes 4 multi-scale ViT features (each B, 196, 768) and produces
    dense depth (B,1,H,W) and normal (B,3,H,W) predictions.

    Pipeline:
        Reassemble (tokens -> 2D multi-scale maps)
        -> Progressive Fusion (coarse-to-fine, RefineNet-style)
        -> Prediction Heads (depth + normal)
    """

    def __init__(self, embed_dim=768, features=256, dropout=0.0):
        super().__init__()

        self.reassemble = nn.ModuleList([
            Reassemble(embed_dim, features, scale_factor=4.0),
            Reassemble(embed_dim, features, scale_factor=2.0),
            Reassemble(embed_dim, features, scale_factor=1.0),
            Reassemble(embed_dim, features, scale_factor=0.5),
        ])

        self.fusion = nn.ModuleList([
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
        ])

        self.drop = nn.Dropout2d(p=dropout)

        self.depth_head = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        self.normal_head = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, features):
        maps = [r(f) for f, r in zip(features, self.reassemble)]

        x = self.fusion[0](maps[3])
        x = self.fusion[1](x, maps[2])
        x = self.fusion[2](x, maps[1])
        x = self.fusion[3](x, maps[0])

        x = self.drop(x)

        depth  = self.depth_head(x)
        normal = self.normal_head(x)
        return depth, normal


class SITRWithDPT(nn.Module):
    """
    Full Stage-2 model: frozen SITR encoder + trainable DPT decoder.

    Input:  tactile image (B,3,224,224) + calibration (B,C,224,224)
    Output: dict with 'depth' (B,1,224,224) and 'normal' (B,3,224,224)
    """

    def __init__(self, sitr, embed_dim=768, features=256,
                 layer_indices=(2, 5, 8, 11), unfreeze_last_n=0, dropout=0.0):
        super().__init__()
        self.encoder = SITRMultiScale(sitr, layer_indices,
                                      unfreeze_last_n=unfreeze_last_n)
        self.decoder = DPTDecoder(embed_dim, features, dropout=dropout)

    def forward(self, x, c, return_latent=False):
        features = self.encoder(x, c)
        depth, normal = self.decoder(features)
        out = {"depth": depth, "normal": normal}
        if return_latent:
            out["latent"] = features[-1]
        return out

    def forward_encoder_full(self, x, c):
        """Run encoder once, return (multi_scale_features, full_latent).
        full_latent includes cls+patch tokens for pose head."""
        s = self.encoder.sitr
        x_enc = s.patch_embed(x)
        x_enc = x_enc + s.pos_embed[:, 1:, :]
        cls_token = s.cls_token + s.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_enc.shape[0], -1, -1)
        x_enc = torch.cat((cls_tokens, x_enc), dim=1)
        num_patches = s.patch_embed.num_patches
        if s._calib_cache is not None:
            x_enc = torch.cat((x_enc, s._calib_cache.expand(x_enc.shape[0], -1, -1)), dim=1)
        elif s.num_calibration > 0:
            c_enc = s.c_patch_embed(c)
            c_enc = c_enc + s.c_pos_embed
            x_enc = torch.cat((x_enc, c_enc), dim=1)
        features = []
        for i, blk in enumerate(s.blocks):
            with torch.no_grad():
                x_enc = blk(x_enc)
            if i in self.encoder.layer_indices:
                features.append(x_enc[:, 1:num_patches + 1, :])
        x_enc = s.norm(x_enc)
        if s.num_calibration > 0:
            latent = x_enc[:, :num_patches + 1, :]
        else:
            latent = x_enc
        return features, latent


class DINOv2MultiScale(nn.Module):
    """
    Wraps a frozen DINOv2 encoder to extract multi-scale features from
    intermediate transformer blocks.

    Output: list of 4 tensors, each (B, num_patches, embed_dim).
    No calibration input — single RGB image only.
    """

    def __init__(self, model_name='dinov2_vitb14', layer_indices=(2, 5, 8, 11)):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        self.layer_indices = sorted(layer_indices)
        self.embed_dim = self.dinov2.embed_dim

        for p in self.dinov2.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.dinov2.eval()
        return self

    def forward(self, x):
        features = self.dinov2.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False, return_class_token=False,
        )
        return list(features)


class DINOv2WithDPT(nn.Module):
    """
    Frozen DINOv2 encoder + trainable DPT decoder.

    Input:  tactile image (B,3,224,224); calibration argument accepted but ignored.
    Output: dict with 'depth' (B,1,224,224) and 'normal' (B,3,224,224)
    """

    def __init__(self, model_name='dinov2_vitb14', features=256,
                 layer_indices=(2, 5, 8, 11), dropout=0.0):
        super().__init__()
        self.encoder = DINOv2MultiScale(model_name, layer_indices)
        embed_dim = self.encoder.embed_dim
        self.decoder = DPTDecoder(embed_dim, features, dropout=dropout)

    def forward(self, x, c=None):
        features = self.encoder(x)
        depth, normal = self.decoder(features)
        h, w = x.shape[2:]
        if depth.shape[2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=True)
            normal = F.interpolate(normal, size=(h, w), mode='bilinear', align_corners=True)
        return {"depth": depth, "normal": normal}


class DAv2MultiScale(nn.Module):
    """Frozen Depth Anything V2 encoder (depth-finetuned DINOv2) for multi-scale features."""

    def __init__(self, model_name='dinov2_vitb14', weights=None,
                 layer_indices=(2, 5, 8, 11)):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        self.layer_indices = sorted(layer_indices)
        self.embed_dim = self.dinov2.embed_dim

        if weights is not None:
            print(f'  [encoder] loading DAv2 weights from {weights}')
            ck = torch.load(weights, map_location='cpu', weights_only=False)
            backbone_sd = {k.replace('pretrained.', ''): v
                           for k, v in ck.items() if k.startswith('pretrained.')}
            missing, unexpected = self.dinov2.load_state_dict(backbone_sd, strict=False)
            print(f'  [encoder] loaded {len(backbone_sd) - len(unexpected)} / '
                  f'{sum(1 for _ in self.dinov2.parameters())} parameters')
            if missing:
                print(f'  [encoder] WARNING: {len(missing)} missing keys')

        for p in self.dinov2.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.dinov2.eval()
        return self

    def forward(self, x):
        features = self.dinov2.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False, return_class_token=False,
        )
        return list(features)


class DAv2WithDPT(nn.Module):
    """Frozen DAv2 encoder + trainable DPT decoder for depth + normal."""

    def __init__(self, model_name='dinov2_vitb14', weights=None, features=256,
                 layer_indices=(2, 5, 8, 11), dropout=0.0):
        super().__init__()
        self.encoder = DAv2MultiScale(model_name, weights, layer_indices)
        embed_dim = self.encoder.embed_dim
        self.decoder = DPTDecoder(embed_dim, features, dropout=dropout)

    def forward(self, x, c=None):
        features = self.encoder(x)
        depth, normal = self.decoder(features)
        h, w = x.shape[2:]
        if depth.shape[2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=True)
            normal = F.interpolate(normal, size=(h, w), mode='bilinear', align_corners=True)
        return {"depth": depth, "normal": normal}


def _count_blocks(backbone):
    blocks = getattr(backbone, 'blocks', None)
    if blocks is None:
        return None
    try:
        return len(blocks)
    except TypeError:
        return sum(1 for _ in blocks)


def auto_layer_indices(backbone, k=4):
    depth = _count_blocks(backbone)
    if not depth or depth < k:
        return (2, 5, 8, 11)
    step = depth / k
    idx = sorted({min(depth - 1, int(round((i + 1) * step)) - 1) for i in range(k)})
    while len(idx) < k:
        for cand in range(depth - 1, -1, -1):
            if cand not in idx:
                idx.append(cand)
                break
        idx = sorted(set(idx))
    return tuple(idx[-k:])


def _convert_dinov3_hf_to_hub(state_dict, model):
    """Auto-convert HuggingFace DINOv3 keys to torch.hub format if needed."""
    hub_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    if not (hub_keys - set(state_dict.keys())):
        return state_dict

    if not any(k.startswith('embeddings.') or k.startswith('layer.') for k in state_dict):
        return state_dict

    hub_sd = {}
    _map = {
        'embeddings.cls_token': 'cls_token',
        'embeddings.patch_embeddings.weight': 'patch_embed.proj.weight',
        'embeddings.patch_embeddings.bias': 'patch_embed.proj.bias',
        'embeddings.register_tokens': 'storage_tokens',
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
    }
    for hf_k, hub_k in _map.items():
        if hf_k in state_dict:
            hub_sd[hub_k] = state_dict[hf_k]
    if 'embeddings.mask_token' in state_dict:
        t = state_dict['embeddings.mask_token']
        hub_sd['mask_token'] = t.squeeze(0) if t.dim() == 3 else t

    num_blocks = max((int(k.split('.')[1]) for k in state_dict if k.startswith('layer.')), default=-1) + 1
    for i in range(num_blocks):
        ph, pb = f'layer.{i}', f'blocks.{i}'
        for n in ['norm1', 'norm2']:
            for s in ['weight', 'bias']:
                k = f'{ph}.{n}.{s}'
                if k in state_dict:
                    hub_sd[f'{pb}.{n}.{s}'] = state_dict[k]
        if f'{ph}.layer_scale1.lambda1' in state_dict:
            hub_sd[f'{pb}.ls1.gamma'] = state_dict[f'{ph}.layer_scale1.lambda1']
        if f'{ph}.layer_scale2.lambda1' in state_dict:
            hub_sd[f'{pb}.ls2.gamma'] = state_dict[f'{ph}.layer_scale2.lambda1']
        q_w = state_dict.get(f'{ph}.attention.q_proj.weight')
        k_w = state_dict.get(f'{ph}.attention.k_proj.weight')
        v_w = state_dict.get(f'{ph}.attention.v_proj.weight')
        if q_w is not None:
            hub_sd[f'{pb}.attn.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
        q_b = state_dict.get(f'{ph}.attention.q_proj.bias')
        v_b = state_dict.get(f'{ph}.attention.v_proj.bias')
        if q_b is not None:
            hub_sd[f'{pb}.attn.qkv.bias'] = torch.cat([q_b, torch.zeros_like(q_b), v_b], dim=0)
        for s in ['weight', 'bias']:
            k = f'{ph}.attention.o_proj.{s}'
            if k in state_dict:
                hub_sd[f'{pb}.attn.proj.{s}'] = state_dict[k]
            k = f'{ph}.mlp.up_proj.{s}'
            if k in state_dict:
                hub_sd[f'{pb}.mlp.fc1.{s}'] = state_dict[k]
            k = f'{ph}.mlp.down_proj.{s}'
            if k in state_dict:
                hub_sd[f'{pb}.mlp.fc2.{s}'] = state_dict[k]

    print(f'  [encoder] converted {len(state_dict)} HF keys -> {len(hub_sd)} hub keys')
    return hub_sd


class DINOv3MultiScale(nn.Module):
    def __init__(self, model_name='dinov3_vitl16', weights=None,
                 layer_indices=None):
        super().__init__()
        if weights is None:
            raise ValueError(
                'DINOv3 weights are gated. Pass a local .pth path via weights=.')

        self.dinov3 = torch.hub.load(
            'facebookresearch/dinov3', model_name, pretrained=False)

        print(f'  [encoder] loading weights from {weights}')
        state_dict = torch.load(weights, map_location='cpu', weights_only=True)
        state_dict = _convert_dinov3_hf_to_hub(state_dict, self.dinov3)
        missing, unexpected = self.dinov3.load_state_dict(state_dict, strict=False)
        num_buf_missing = sum(1 for k in missing
                              if k in dict(self.dinov3.named_buffers()))
        num_param_missing = len(missing) - num_buf_missing
        if num_param_missing:
            print(f'  [encoder] WARNING: {num_param_missing} parameter keys '
                  f'NOT loaded (missing)')
        if unexpected:
            print(f'  [encoder] unexpected keys: {len(unexpected)}')
        print(f'  [encoder] loaded {len(state_dict) - len(unexpected)} / '
              f'{sum(1 for _ in self.dinov3.parameters())} parameters')

        if layer_indices is None:
            layer_indices = auto_layer_indices(self.dinov3, k=4)
            print(f'  [encoder] depth={_count_blocks(self.dinov3)} '
                  f'-> auto layer_indices={layer_indices}')
        self.layer_indices = sorted(layer_indices)

        self.embed_dim = getattr(self.dinov3, 'embed_dim', None)
        if self.embed_dim is None:
            self.embed_dim = self.dinov3.norm.normalized_shape[0]
        for p in self.dinov3.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.dinov3.eval()
        return self

    def forward(self, x):
        features = self.dinov3.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False, return_class_token=False)
        return list(features)


class DINOv3WithDPT(nn.Module):
    def __init__(self, model_name='dinov3_vitl16', weights=None, features=256,
                 layer_indices=None, dropout=0.0):
        super().__init__()
        self.encoder = DINOv3MultiScale(model_name, weights, layer_indices)
        embed_dim = self.encoder.embed_dim
        self.decoder = DPTDecoder(embed_dim, features, dropout=dropout)

    def forward(self, x, c=None):
        features = self.encoder(x)
        depth, normal = self.decoder(features)
        h, w = x.shape[2:]
        if depth.shape[2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=True)
            normal = F.interpolate(normal, size=(h, w), mode='bilinear', align_corners=True)
        return {'depth': depth, 'normal': normal}
