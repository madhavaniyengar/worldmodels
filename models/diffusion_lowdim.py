import torch
import torch.nn as nn
import math

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) timesteps
        returns: (B, dim)
        """
        if t.dim() == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * -(math.log(10000.0) / (half - 1))
        )
        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, emb.new_zeros(emb.size(0), 1)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.fc1(self.act(self.norm1(x)))
        h = self.fc2(self.act(self.norm2(h)))
        return x + h

class DiffusionMLP(nn.Module):
    """
    Denoiser for a set of generated modalities, conditioned on others.
    """
    def __init__(
        self,
        gen_dims: dict,         
        cond_dims: dict,        
        hidden_dim: int = 512,
        num_blocks: int = 4,
        t_embed_dim: int = 128,
        dropout: float = 0.0,
        return_dict: bool = True,
        fusion_method: str = "add"
    ):
        super().__init__()
        self.gen_dims = gen_dims
        self.cond_dims = cond_dims
        self.return_dict = return_dict
        self.fusion_method = fusion_method


        # modalities to generate
        self.gen_projs = nn.ModuleDict({
            k: nn.Linear(d, hidden_dim) for k, d in gen_dims.items()
        })
        
        # modalities to condition on
        self.cond_projs = nn.ModuleDict({
            k: nn.Linear(d, hidden_dim) for k, d in cond_dims.items()
        })

        self.t_embed = SinusoidalTimestepEmbedding(t_embed_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # fusion dim: depends on fusion method
        if fusion_method == "add":
            # sum of all projected modalities + t_emb 
            fusion_dim = hidden_dim
        elif fusion_method == "concat":
            # concat all projected modalities + t_emb
            num_gen_modalities = len(gen_dims)
            num_cond_modalities = len(cond_dims)
            fusion_dim = hidden_dim * (num_gen_modalities + num_cond_modalities + 1)  # +1 for t_emb
        else:
            raise ValueError(f"fusion_method must be 'add' or 'concat', got {fusion_method}")
        
        self.fusion_dim = fusion_dim
        
        # we just sum projected features modality-wise (another option is concat)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ResidualBlock(fusion_dim) for _ in range(num_blocks)])

        # output heads
        self.heads = nn.ModuleDict({
            k: nn.Linear(fusion_dim, d) for k, d in gen_dims.items()
        })

    def forward(self, gen_inputs: dict, cond_inputs: dict, t: torch.Tensor):
        """
        gen_inputs: dict of tensors, each (B, D_gen[k])
        cond_inputs: dict of tensors, each (B, D_cond[k])
        t: (B,) or (B,1)
        """
        B = t.shape[0]

        if self.fusion_method == "add":
            h = 0
            for k, xk in gen_inputs.items():
                h = h + self.gen_projs[k](xk)

            for k, ck in cond_inputs.items():
                h = h + self.cond_projs[k](ck)

            t_emb = self.t_mlp(self.t_embed(t))
            h = h + t_emb

        elif self.fusion_method == "concat":
            gen_features = [self.gen_projs[k](xk) for k, xk in gen_inputs.items()]
            cond_features = [self.cond_projs[k](ck) for k, ck in cond_inputs.items()]
            
            t_emb = self.t_mlp(self.t_embed(t))
            
            all_features = gen_features + cond_features + [t_emb]
            h = torch.cat(all_features, dim=-1)

        h = self.dropout(h)
        for block in self.blocks:
            h = block(h)

        if self.return_dict:
            out = {k: self.heads[k](h) for k in self.gen_dims}
        else:
            outs = [self.heads[k](h) for k in self.gen_dims]
            out = torch.cat(outs, dim=-1)

        return out
