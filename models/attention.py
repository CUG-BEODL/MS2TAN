import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from models.rotary import AxialRotaryEmbedding, RotaryEmbedding, apply_rot_emb

calc_flops = False
calc_attn = False

# helpers


def exists(val):
    return val is not None


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# feedforward


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(dim, dim * mult * 2),
            # GEGLU(),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):

        if calc_flops:
            flops_linear1 = 2 * x.size(0) * x.size(1) * \
                x.size(2) * self.net[0].out_features
            flops_gelu = x.size(0) * x.size(1) * x.size(2)
            flops_dropout = 0  # Assuming Dropout doesn't add significant FLOPs
            flops_linear2 = 2 * x.size(0) * x.size(1) * self.net[3].in_features

            total_flops = flops_linear1 + flops_gelu + flops_dropout + flops_linear2
            print(f"{total_flops}\t{self.__class__.__name__}\t{x.shape}")

        return self.net(x)

# mask


def obs_to_time_mask(obs, patch_size=10, num_head=5, nozero_ratio=0.5):
    obs_patchs = rearrange(
        obs, 'b t c (hpn hps) (wpn wps) -> b t c hpn wpn hps wps', hps=patch_size, wps=patch_size)
    # print(obs_patchs.shape)
    obs_patchs_mean = reduce(
        obs_patchs, 'b t c hpn wpn hps wps -> b (hpn wpn) t', 'mean')
    # print(obs_patchs_mean.shape)
    obs_patchs_bool = (obs_patchs_mean > nozero_ratio) * 1.
    # obs_patchs_mask = obs_patchs_bool.unsqueeze(
    #     -1) @ obs_patchs_bool.unsqueeze(-2)
    obs_patchs_mask = repeat(
        obs_patchs_bool, 'b t n2 -> b t n1 n2', n1=obs_patchs_bool.shape[-1])
    # print(obs_patchs_mask.shape)
    obs_patchs_mask = repeat(
        obs_patchs_mask, 'b p tn1 tn2 -> (b h p) tn1 tn2', h=num_head)
    return obs_patchs_mask


def obs_to_space_mask(obs, patch_size=10, num_head=5, nozero_ratio=0.5):
    obs_patchs = rearrange(
        obs, 'b t c (hpn hps) (wpn wps) -> b t c hpn wpn hps wps', hps=patch_size, wps=patch_size)
    # print(obs_patchs.shape)
    obs_patchs_mean = reduce(
        obs_patchs, 'b t c hpn wpn hps wps -> b t (hpn wpn)', 'mean')
    # print(obs_patchs_mean.shape)
    obs_patchs_bool = (obs_patchs_mean > nozero_ratio) * 1.
    # obs_patchs_mask = obs_patchs_bool.unsqueeze(
    #     -1) @ obs_patchs_bool.unsqueeze(-2)
    obs_patchs_mask = repeat(
        obs_patchs_bool, 'b t n2 -> b t n1 n2', n1=obs_patchs_bool.shape[-1])
    # print(obs_patchs_mask.shape)
    obs_patchs_mask = repeat(
        obs_patchs_mask, 'b t pn1 pn2 -> (b h t) pn1 pn2', h=num_head)
    return obs_patchs_mask


# attention


# def attn(q, k, v, mask, diag):
#     sim = einsum("b i d, b j d -> b i j", q, k)

#     max_neg_value = -torch.finfo(sim.dtype).max

#     if diag:
#         # 对角mask, 1表示遮掩
#         diag_mask = torch.eye(sim.shape[1], device=sim.device).bool()
#         sim.masked_fill_(diag_mask, max_neg_value)
#     if exists(mask):
#         # 缺失值mask, 0表示遮掩
#         missing_mask = (mask == 0).bool()
#         sim.masked_fill_(missing_mask, max_neg_value)

#     # print(f'sim.shape={sim.shape}, mask.shape={mask.shape}')
#     attn = sim.softmax(dim=-1)
#     out = einsum("b i j, b j d -> b i d", attn, v)
#     # print(f'attn.shape={attn.shape}, out.shape={out.shape}')
#     return out


def attn(q, k, v, mask, diag):
    if calc_flops:
        # 获取输入张量的形状
        q_shape = q.shape
        k_shape = k.shape
        v_shape = v.shape

    # 使用 torch.einsum 计算 sim
    sim = einsum("b i d, b j d -> b i j", q, k)

    max_neg_value = -torch.finfo(sim.dtype).max

    if diag:
        diag_mask = torch.eye(sim.shape[1], device=sim.device).bool()
        sim.masked_fill_(diag_mask, max_neg_value)
    if mask is not None:
        missing_mask = (mask == 0).bool()
        sim.masked_fill_(missing_mask, max_neg_value)

    # 使用 softmax 计算 attn
    attn = sim.softmax(dim=-1)

    # 使用 einsum 计算 out
    out = einsum("b i j, b j d -> b i d", attn, v)

    # 计算 FLOPs
    if calc_flops:
        flops_qk = 2 * q_shape[0] * q_shape[1] * k_shape[1] * k_shape[2]
        flops_softmax = 2 * attn.shape[0] * \
            attn.shape[1] * attn.shape[2]  # Softmax operation
        flops_attn_out = 2 * attn.shape[0] * attn.shape[1] * \
            attn.shape[2] + 2 * attn.shape[0] * attn.shape[1] * v.shape[2]

        total_flops = flops_qk + flops_softmax + flops_attn_out
        print(f"{total_flops}\t{q_shape}\t{k_shape}")
    # print(f"flops_qk={clever_format(flops_qk)}, "
    #     f"flops_softmax={clever_format(flops_softmax)}, "
    #     f"flops_attn_out={clever_format(flops_attn_out)}")
    # print(f"Total FLOPs for attn function: {clever_format(total_flops)}")

    if calc_attn:
        return out, attn

    return out


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x,
        einops_from,
        einops_to,
        mask,
        diag,
        rot_emb=None,
        **einops_dims,
    ):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # print(f"q, k, v.shape={q.shape, k.shape, v.shape}")

        q = q * self.scale

        # splice out classification token at index 1
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
        #     lambda t: (t[:, :1], t[:, 1:]), (q, k, v)
        # )
        q_, k_, v_ = q, k, v

        # let classification token attend to key / values of all patches across time and space
        # cls_out = attn(cls_q, k, v, mask=cls_mask)

        # rearrange across time or space
        q_, k_, v_ = map(
            lambda t: rearrange(
                t, f"{einops_from} -> {einops_to}", **einops_dims),
            (q_, k_, v_),
        )

        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        # expand cls token keys and values across time or space and concat
        # r = q_.shape[0] // cls_k.shape[0]
        # cls_k, cls_v = map(
        #     lambda t: repeat(t, "b () d -> (b r) () d", r=r), (cls_k, cls_v)
        # )

        # k_ = torch.cat((cls_k, k_), dim=1)
        # v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        # 对角遮掩
        # mask = torch.eye(q.shape[1])==0
        if calc_attn:
            out, attn_map = attn(q_, k_, v_, mask=mask, diag=diag)
        else:
            out = attn(q_, k_, v_, mask=mask, diag=diag)

        # merge back time or space
        out = rearrange(out, f"{einops_to} -> {einops_from}", **einops_dims)

        # concat back the cls token
        # out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        # combine heads out
        out = self.to_out(out)

        if calc_attn:
            return out, attn_map

        return out


# main classes


class MFE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        image_size=224,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        rotary_emb=True,
        missing_mask=True,
        diag_mask=True,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim_in = in_channels * patch_size**2
        patch_dim_out = out_channels * patch_size**2

        print(
            f"num_patches={num_patches}, \
            num_positions={num_positions}, \
            patch_dim_in={patch_dim_in}, \
            patch_dim_out={patch_dim_out}"
        )

        self.heads = heads
        self.patch_size = patch_size
        # patch -> embedding
        self.to_patch_embedding = nn.Linear(patch_dim_in, dim)
        # embedding -> patch
        self.from_patch_embedding = nn.Linear(dim, patch_dim_out)
        # self.from_patch_embedding = nn.Sequential(
        #     nn.Linear(dim, 2 * patch_dim_out),
        #     nn.GELU(),
        #     nn.Linear(2 * patch_dim_out, patch_dim_out)
        # )
        self.missing_mask = missing_mask
        self.diag_mask = diag_mask

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            # self.pos_emb = nn.Embedding(num_positions + 1, dim)
            self.pos_emb = nn.Embedding(num_positions, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout=ff_dropout)
            time_attn = Attention(
                dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
            )
            spatial_attn = Attention(
                dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
            )

            time_attn, spatial_attn, ff = map(
                lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff)
            )

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))
            # self.layers.add_module('time_attn', time_attn)
            # self.layers.add_module('spatial_attn', spatial_attn)
            # self.layers.add_module('ff', ff)

    def forward(self, seq, obs, return_attn=False):
        b, f, _, h, w, *_, device, p = *seq.shape, seq.device, self.patch_size
        assert (
            h % p == 0 and w % p == 0
        ), f"height {h} and width {w} of seq must be divisible by the patch size {p}"

        # calculate num patches in height and width dimension, and number of total patches (n)

        hp, wp = (h // p), (w // p)
        n = hp * wp

        # seq to patch embeddings

        # seq = seq / 255.0 - self.mean

        # print(f"seq.shape={seq.shape}")

        seq_merge = rearrange(
            seq, "b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)", p1=p, p2=p
        )
        tokens = self.to_patch_embedding(seq_merge)

        # print(f"seq.shape={seq.shape}")
        # print(f"seq_merge.shape={seq_merge.shape}")
        # print(f"tokens.shape={tokens.shape}")

        # add cls token

        # cls_token = repeat(self.cls_token, "n d -> b n d", b=b)
        # x = torch.cat((cls_token, tokens), dim=1)
        x = tokens

        # positional embedding

        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device=device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device=device)
            image_pos_emb = self.image_rot_emb(hp, wp, device=device)

        # calculate masking for uneven number of frames

        # frame_mask = None
        # cls_attn_mask = None
        # if exists(mask):
        #     mask_with_cls = F.pad(mask, (1, 0), value=True)

        #     frame_mask = repeat(mask_with_cls, "b f -> (b h n) () f", n=n, h=self.heads)

        #     cls_attn_mask = repeat(mask, "b f -> (b h) () (f n)", n=n, h=self.heads)
        #     cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value=True)
        temporal_mask = None
        spatial_mask = None
        if self.missing_mask:
            # 1表示有效，0表示遮掩
            temporal_mask = obs_to_time_mask(
                obs, self.patch_size, self.heads).to(device)
            spatial_mask = obs_to_space_mask(
                obs, self.patch_size, self.heads).to(device)

        # time and space attention
        if calc_attn:
            attn_list = []
        else:
            attn_list = None

        for time_attn, spatial_attn, ff in self.layers:
            # print(f'x.shape={x.shape}')
            if calc_attn:
                res = x
                x, t_attn_map = time_attn(
                                        x,
                                        "b (f n) d",
                                        "(b n) f d",
                                        n=n,
                                        mask=temporal_mask,
                                        diag=self.diag_mask,
                                        rot_emb=frame_pos_emb,
                                    )
                x = x + res
                res = x
                x, s_attn_map = spatial_attn(
                                        x,
                                        "b (f n) d",
                                        "(b f) n d",
                                        f=f,
                                        mask=spatial_mask,
                                        diag=self.diag_mask,
                                        rot_emb=image_pos_emb,
                                    )
                x = x + res
                t_attn_map, s_attn_map = t_attn_map.detach(), s_attn_map.detach()
                attn_list.append([t_attn_map, s_attn_map])
            else:
                # print('time_attn:')
                x = (
                    time_attn(
                        x,
                        "b (f n) d",
                        "(b n) f d",
                        n=n,
                        mask=temporal_mask,
                        diag=self.diag_mask,
                        rot_emb=frame_pos_emb,
                    )
                    + x
                    )
                # print('spatial_attn')
                x = (
                    spatial_attn(
                        x,
                        "b (f n) d",
                        "(b f) n d",
                        f=f,
                        mask=spatial_mask,
                        diag=self.diag_mask,
                        rot_emb=image_pos_emb,
                    )
                    + x
                )
                
            # print('feedforward')
            x = ff(x) + x

        # cls_token = x[:, 0]
        # seq_out = x[:, 1:]
        seq_out = x

        # seq_out = self.seq_proj(seq_out)
        seq_out = self.from_patch_embedding(seq_out)

        seq_out = rearrange(
            seq_out,
            "b (f h w) (p1 p2 c) -> b f c (h p1) (w p2)",
            p1=p,
            p2=p,
            f=f,
            h=hp,
            w=wp,
        )

        # seq_out = seq_out.contiguous().reshape(b, f, -1, h, w)

        # seq = seq + seq_out

        if return_attn:
            return seq_out, attn_list

        return seq_out
