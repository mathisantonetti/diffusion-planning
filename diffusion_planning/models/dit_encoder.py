import torch.nn as nn
import torch
import numpy as np
import einops
import math
from timm.models.vision_transformer import Attention, Mlp
import diffusion_planning.utils as utils
import torch.nn as nn


class CustomDiTEncoder(nn.Module):
    """
    A DiT Based Traj Encoder for the Ovlp Part, 
    We just need the Cls-Token
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_size=256,
        depth=1,
        num_heads=4,
    ):
        super().__init__()

        self.x_embedder = nn.Linear(in_features=in_dim, out_features=hidden_size)
        self.t_embedder = TimestepEmbedder(self.out_dim)

        self.attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.out_dim = out_dim
        self.hidden_dim = hidden_size
                              

    def forward(self, x, time):
        """
        Forward pass of DiT-based Trajectory Encoder
        x: (B, H, obs_dim) a batch of trajs
        time: (B, T, 1) tensor of diffusion timesteps

        Returns (B, T, out_dim)
        """

        pos_embed = torch.zeros(1, x.shape[1], self.hidden_dim, requires_grad=False)
        tmp_pos_arr = np.arange(x.shape[1], dtype=np.int32 )
        pos_embed.data.copy_(torch.from_numpy(get_1d_sincos_pos_embed_from_grid(embed_dim=self.hidden_dim, pos=tmp_pos_arr)).float().unsqueeze(0))


        ## (B, H, D), e.g., [4, 160, 384]
        x_input_emb = self.x_embedder(x)

        ## (B, H, D)
        x_input_emb = x_input_emb + pos_embed
        
        t_feat = self.t_embedder(time.view(-1)).view(x.shape[0], -1, 1).repeat(1, 1, self.out_dim)

        attn_output, _ = self.attn(t_feat, x_input_emb, x_input_emb)
        
        return attn_output

class DiT1D_Traj_Time_Encoder(nn.Module):
    """
    A DiT Based Traj Encoder for the Ovlp Part, 
    We just need the Cls-Token
    """
    def __init__(
        self,
        c_traj_hzn,
        in_dim,
        out_dim,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        w_init_type='dit1d',
    ):
        super().__init__()

        self.c_traj_hzn = c_traj_hzn
        self.out_dim = out_dim
        self.transition_dim = in_dim
        self.hidden_size = hidden_size
        
        self.out_channels = self.transition_dim
        
        self.num_heads = num_heads

        ### ------------------------------------------------------------------------
        ## -- Define How Condition Signal are fused with the Transformer Denoiser --

        self.time_dim = hidden_size
        self.t_embedder = TimestepEmbedder(self.time_dim)

        ## ---------- Init the DiT 1D Backbone -------------

        ## PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        ## similar to diffusion policy, just one Linear, might upgrade to an MLP
        self.x_embedder = nn.Linear(in_features=self.transition_dim, out_features=hidden_size)

        ## plus 1 for the cls_token
        self.num_patches = self.c_traj_hzn + 1
        
        # Will use fixed sin-cos embedding: (will be init later)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        ## Will be init later
        self.cls_token = nn.Parameter(data=torch.randn(1, 1, hidden_size,))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                     cond_dim=hidden_size) for _ in range(depth)
        ])
        
        # pdb.set_trace()
        ## final_layer of the all tokens
        self.final_layer = FinalLayer(hidden_size, 
                                out_channels=hidden_size, cond_dim=hidden_size,)

        assert out_dim == hidden_size, 'for now'
        ## we only care about the final cls-token
        # self.mlp_head = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            # nn.Linear(hidden_size, out_dim)
        # )

        ## -----------
        # pdb.set_trace()
        if w_init_type == 'dit1d':
            self.initialize_weights()
        elif w_init_type == 'no':
            pass
        else:
            raise NotImplementedError

        # pdb.set_trace()
        utils.print_color(f'[DiT1D_Traj_Time_Encoder] {self.num_patches=}, '
                          f'{hidden_size=}, {depth=}, {out_dim=},', c='c')
        self.num_params = utils.report_parameters(self, topk=0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        assert (self.num_patches - 1) == self.c_traj_hzn, 'for now'
        tmp_pos_arr = np.arange(self.num_patches, dtype=np.int32 )
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.hidden_size, pos=tmp_pos_arr)
        
        ## pos_embed will still be float32
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w) ## no need to reshape, since already a Linear
        nn.init.constant_(self.x_embedder.bias, 0)

        # pdb.set_trace()
        # Initialize the cls-token
        nn.init.normal_(self.cls_token.data, std=0.02)

        # pdb.set_trace()

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, time,):
        """
        Forward pass of DiT-based Trajectory Encoder
        x: (B, H, Dim) a batch of trajs
        time: (B,) tensor of diffusion timesteps

        Returns (B, H, dim)
        """

        ## (B, H, D), e.g., [4, 160, 384]
        x_input_emb = self.x_embedder(x)
        b_s, _, _ = x.shape
        cls_tokens = einops.repeat(self.cls_token, '1 1 d -> b 1 d', b = b_s)


        # pdb.set_trace()
        x_input_emb = torch.cat( (cls_tokens, x_input_emb), dim=1 )
        ## (B, H, D)
        x_input_emb = x_input_emb + self.pos_embed

        # pdb.set_trace()


        ## ------------------------------------------------------
        ## ------------- obtain Condition feature ---------------

        

        ## ------------ create cond_feat for denoiser --------------

        ## (B, tot_cond_dim)
        t_feat = self.t_embedder(time)
        c_feat_all = t_feat

        # pdb.set_trace()

        ## ------------ Denoising BackBone ----------------

        x = x_input_emb ## prev: x.shape=B,H,obs_dim

        for block in self.blocks:
            x = block(x, c_feat_all)  # (B, T, hid_D)
        
        x = self.final_layer(x, c_feat_all)  # (B, T, obs_dim)
        
        ## (B, out_dim)
        cls_latents = x[:, 0, :]
        # pdb.set_trace() ## check shape

        return cls_latents
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, 
                 cond_dim=None,
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        cond_dim = hidden_size if cond_dim is None else cond_dim
        # pdb.set_trace()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    out_channels should directly be the env transition dim, e.g., 29 for antMaze
    """
    def __init__(self, hidden_size, out_channels, cond_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        cond_dim = hidden_size if cond_dim is None else cond_dim
        # pdb.set_trace()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
#################################################################################
#                                 Positional Embedding                          #
#################################################################################

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    print(embed_dim)
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

