import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    Baseline MLP model for direct prediction of future force and velocity.
    Uses MSE loss instead of diffusion.
    """
    def __init__(
        self,
        gen_dims: dict,
        cond_dims: dict,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.0,
        return_dict: bool = True,
        fusion_method: str = "add"
    ):
        super().__init__()
        self.gen_dims = gen_dims
        self.cond_dims = cond_dims
        self.return_dict = return_dict
        self.fusion_method = fusion_method

        # modalities to condition on
        self.cond_projs = nn.ModuleDict({
            k: nn.Linear(d, hidden_dim) for k, d in cond_dims.items()
        })

        # fusion dim: depends on fusion method
        if fusion_method == "add":
            # sum of all projected modalities
            fusion_dim = hidden_dim
        elif fusion_method == "concat":
            # concat all projected modalities
            num_cond_modalities = len(cond_dims)
            fusion_dim = hidden_dim * num_cond_modalities
        else:
            raise ValueError(f"fusion_method must be 'add' or 'concat', got {fusion_method}")
        
        self.fusion_dim = fusion_dim
        
        # core mlp blocks
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ResidualBlock(fusion_dim) for _ in range(num_blocks)])

        # output heads
        self.heads = nn.ModuleDict({
            k: nn.Linear(fusion_dim, d) for k, d in gen_dims.items()
        })

    def forward(self, cond_inputs: dict):
        """
        cond_inputs: dict of tensors, each (B, D_cond[k])
        returns: dict of tensors, each (B, D_gen[k]) or concatenated tensor
        """
        if self.fusion_method == "add":
            # project and sum all conditioning modalities
            h = 0
            for k, ck in cond_inputs.items():
                h = h + self.cond_projs[k](ck)

        elif self.fusion_method == "concat":
            # project all conditioning modalities
            cond_features = [self.cond_projs[k](ck) for k, ck in cond_inputs.items()]
            
            # concat all features
            h = torch.cat(cond_features, dim=-1)

        #  residual mlp
        h = self.dropout(h)
        for block in self.blocks:
            h = block(h)

        # heads
        if self.return_dict:
            out = {k: self.heads[k](h) for k in self.gen_dims}
        else:
            outs = [self.heads[k](h) for k in self.gen_dims]
            out = torch.cat(outs, dim=-1)

        return out


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