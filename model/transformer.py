import numpy as np 

import torch.nn as nn
import torch

np.random.seed(0)
torch.manual_seed(0)


def get_posiitonnal_embeddings(sequence_length, d):
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
        # In forward function

        # normalization Layer 1
        self.normalization_layer_1 = nn.LayerNorm((self.n_patches**2 + 1, self.hidden_dim))

        self.msa = MSA(self.hidden_dim, n_heads)

        # normalization Layer 2
        self.normalization_layer_2 = nn.LayerNorm((self.n_patches**2 + 1, self.hidden_dim))

        # Encoder MLP 
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU()
        )
    
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_dim, out_dim), 
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Dividing images into patches 
        n, c, w, h = x.shape
        patches = x.reshape(n, self.n_patches ** 2, self.input_dim)

        # Runnnong linear mapper for tokenization 
        tokens = self.linear_mapper(patches)

        # Adding classification token
        tokens = torch.stack([torch.vstack((self.classification_token, token)) for token in tokens])

        # Adding positional embeddings
        positional_embeddings = get_posiitonnal_embeddings(tokens.shape[1], self.hidden_dim).repeat(tokens.shape[0], 1, 1)
        tokens += positional_embeddings

        # Normalization
        tokens = self.normalization_layer_1(tokens)

        # Multi Head Self Attention
        out_msa = tokens + self.msa(tokens)


        # Layer Normalization + MLP + residual connection
        out_mlp = self.encoder_mlp(self.normalization_layer_2(out_msa))

        # Classification head
        out_clf = self.classification_head(out_mlp[:, 0])
        return out_clf


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
            for head_index in range(self.n_heads):
                q_mapping = self.q_mappings[head_index]
                k_mapping = self.k_mappings[head_index]
                v_mapping = self.v_mappings[head_index]

                seq = sequence[:, head_index * self.d_head : (head_index + 1) * self.d_head]
                
                q = q_mapping(seq)
                k = k_mapping(seq)
                v = v_mapping(seq)

                attention = self.softmax(torch.matmul(q, k.T) / np.sqrt(self.d_head))

                seq_results.append(torch.matmul(attention, v))

            result.append(torch.hstack(seq_results))

        # return torch.stack(result)
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

