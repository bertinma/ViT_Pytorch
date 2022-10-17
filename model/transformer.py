import numpy as np 

import torch.nn as nn
import torch

np.random.seed(0)
torch.manual_seed(0)


def get_positional_embeddings(sequence_length, d):
    """
    Returns positional embeddings of shape (sequence_length, d)
    """
    result = torch.ones((sequence_length, d))
    for i in range(sequence_length):
        for j in range(d):
            if j % 2 == 0:
                result[i, j] = np.sin(i / (10000 ** (j / d)))
            else:
                result[i, j] = np.cos(i / (10000 ** ((j - 1) / d)))
    return result


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


class ViT(nn.Module):
    """
    ViT class that implement the model architecture
    Parameters:
        input_shape: tuple of ints, shape of input image
        n_patches: int, number of patches to divide the image into
        hidden_dim: int, dimension of hidden layer
        patch_size: tuple of ints, size of patches to divide the image into
        input_dim: int, dimension of input layer
    """
    def __init__(
        self, 
        input_shape, 
        n_patches = 7, 
        hidden_dim = 8, 
        n_heads = 2, 
        n_blocks = 2,
        out_dim = 10
        ):
        # Super constructor 
        super(ViT, self).__init__()

        # Input and patches shapes 
        self.input_shape = input_shape
        self.n_patches = n_patches
        self.patch_size = (input_shape[1] // n_patches, input_shape[2] // n_patches)
        self.input_dim = input_shape[0] * self.patch_size[0] * self.patch_size[1]
        self.hidden_dim = hidden_dim

        # Assertion 
        assert input_shape[1] % n_patches == 0 and input_shape[2] % n_patches == 0, "Input shape must be divisible by n_patches"

        # Linear mapper 
        self.linear_mapper = nn.Linear(self.input_dim, self.hidden_dim)

        # Classification token 
        self.classification_token = nn.Parameter(torch.randn(1, self.hidden_dim))

        # Positional embeddings
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, self.hidden_dim), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(self.hidden_dim, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Dividing images into patches 
        n, c, w, h = x.shape
        patches = patchify(x, self.n_patches)

        # Runnning linear mapper for tokenization 
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.classification_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        # outs_blocks = torch.Tensor()
        attention_scores = []
        for block in self.blocks:
            out = block(out)
            attention_scores.append(block.attention)

        att_mat = torch.stack(attention_scores).squeeze(1)
        self.att_mat = torch.mean(att_mat, dim=1)
        out = out[:, 0]

        return self.mlp(out) # Map to output dimension, output category distribution
        
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        self.attention = self.mhsa.attention
        return out 



class MSA(nn.Module):
    def __init__(self, hidden_dim, n_heads = 2) -> None:
        super(MSA, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        assert hidden_dim % n_heads == 0, "Hidden dimension must be divisible by n_heads"

        self.d_head = int(self.hidden_dim // n_heads)

        self.q_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, sequences):
        result = []

        for sequence in sequences:
            seq_results = []
            for head_index, (q_mapping, k_mapping, v_mapping) in enumerate(zip(self.q_mappings, self.k_mappings, self.v_mappings)):

                seq = sequence[:, head_index * self.d_head : (head_index + 1) * self.d_head]
                
                q = q_mapping(seq)
                k = k_mapping(seq)
                v = v_mapping(seq)

                self.attention = self.softmax(torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.d_head))

                seq_results.append(torch.matmul(self.attention, v))
            result.append(torch.cat(seq_results, 1))
    
        # return torch.stack(result)
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])