"""
Module implementing the layers and models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        """
        Args:
            d_model (int): dimension of the query, key and value vectors.
        """
        super().__init__()
        ...  # TODO: do not forget the scaling factor

    def forward(self, query, key, value, mask=None):
        """
        Compute the scaled dot-product attention.

        Args:
            query (torch.Tensor): tensor of shape (batch_size, seq_len, d_model) containing the query vectors.
            key (torch.Tensor): tensor of shape (batch_size, seq_len, d_model) containing the key vectors.
            value (torch.Tensor): tensor of shape (batch_size, seq_len, d_model) containing the value vectors.
            mask (torch.Tensor): tensor of shape (batch_size, seq_len, seq_len) containing the mask 
                for the attention scores. (optional)
        Returns:
            (torch.Tensor): tensor of shape (batch_size, seq_len, d_model)
                containing the weighted sum of the value vectors.
        """
        # Compute the scores and scale them
        scores = ...  # TODO

        if mask is not None:
            # Mask the scores so that they are -infinity where mask == 0
            # Hint: you can look into .masked_fill()
            ...  # TODO

        # Compute the attention weights and then weighted output
        ...  # TODO

        return ...
    

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        """
        Args:
            d_model (int): dimension of the query, key and value vectors.
        """
        super().__init__()

        # Query, key and value projections
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = ...  # TODO
        self.Wv = ...

        # Scaled dot-product attention
        self.attention = ...

        # Output projection
        self.Wo = ...

    def forward(self, query, key, value, mask=None):
        """
        Compute the self-attention (single-head).

        Args:
            query (torch.Tensor): tensor of shape (batch_size, seq_len, d_model) containing the query vectors.
            key (torch.Tensor): tensor of shape (batch_size, seq_len, d_model) containing the key vectors.
            value (torch.Tensor): tensor of shape (batch_size, seq_len, d_model) containing the value vectors.
            mask (torch.Tensor): tensor of shape (batch_size, seq_len, seq_len) containing the mask 
                for the attention scores. (optional)
        Returns:
            output (torch.Tensor): tensor of shape (batch_size, seq_len, d_model)
        """
        # Query, key and value projections
        Q = self.Wq(query)
        ...  # TODO

        # Self-attention
        attention_output = ...

        # Output projection
        output = ...

        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, feedforward_dim, dropout=0.1):
        """
        Args:
            d_model (int): dimension of the input (and output) vectors for each token position.
            feedforward_dim (int): dimension of the hidden layer in the feedforward network.
            dropout (float): dropout rate.
        """
        super().__init__()
        self.attention = ...  # TODO
        self.feed_forward = nn.Sequential(  # 2-layers feed-forward network with a ReLU activation
            ...
        )
        self.norm1 = nn.LayerNorm(...)
        self.norm2 = ...
        self.dropout = ...

    def forward(self, x, mask=None):
        """
        Compute the forward pass for a Transformer layer.

        Note: the input x is used as query, key and value vectors for the self-attention.

        Args:
            x (torch.Tensor): tensor of shape (batch_size, seq_len, d_model) containing the input vectors.
            mask (torch.Tensor): tensor of shape (batch_size, seq_len, seq_len) containing the mask
                for the attention scores. (optional)
        Returns:
            output (torch.Tensor): tensor of shape (batch_size, seq_len, d_model)
        """
        # Single-head self-attention
        attention_output = ...  # TODO
        # Residual connection and layer normalization
        x = self.norm1(x + ...)  # TODO
        # Feed-forward network
        ...
        # Residual connection and layer normalization
        ...
        return ...


class Transformer(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model, feedforward_dim, num_layers, dropout, 
                 device, max_seq_len, token2idx, idx2token):
        """
        Args:
            vocab_size (int): size of the vocabulary.
            pad_idx (int): index of the padding token.
            d_model (int): dimension of the input (and output) vectors for each token position.
            feedforward_dim (int): dimension of the hidden layer in the feedforward network.
            num_layers (int): number of transformer layers.
            dropout (float): dropout rate.
            device (torch.device): device to use for the model.
            max_seq_len (int): maximum input sequence length.
            token2idx (dict): mapping from tokens to indices.
            idx2token (dict): mapping from indices to tokens.
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.device = device
        self.max_seq_len = max_seq_len
        self.token2idx = token2idx
        self.idx2token = idx2token

        # Input embedding
        # Hint: use nn.Embedding(), a simple look-up table of learnable vectors
        # that can be accessed through their indices.
        self.embedding = nn.Embedding(...)   # TODO: what should be the number of token embeddings and their dimension?
        self.pos_embedding = nn.Embedding(...)

        # Transformer layers
        # Hint: we want "num_layers" transformer layers.
        ...

        # Output layer
        # Hint: a single linear layer to map the output of the last transformer layer
        # to class probabilities. We classsify tokens here.
        ...
        self.to(device)

    def forward(self, x_ids):
        """
        Predict the next tokens in the sequences.

        Args:
            x_ids (torch.LongTensor): tensor of shape (batch_size, seq_len)
                containing the indices of the input tokens.
        Returns:
            logits (torch.FloatTensor): tensor of shape (batch_size, seq_len, vocab_size)
                containing the logits for each output token.
        """
        # Check that the input sequence is not too long
        assert x_ids.shape[1] <= self.max_seq_len, f"Input sequence is too long for the model! ({x_ids.shape[1]} > {self.max_seq_len})"

        # Embed the input sequence
        # Hint: you need to embed the token and add the position embedding.
        # For the later, you will need to generate a tensor of position indices.
        embedded_input = ...
        embedded_pos = ...
        x = embedded_input + embedded_pos

        # Prepare the attention mask
        mask = get_mask(...).to(x.device)

        # Apply the transformer layers
        ...

        # Generate logits for the next token prediction
        logits = ...
        return logits
    
    def generate_sequence(self, max_length=None):
        """
        Generate a random protein sequence.

        Args:
            max_length (int): maximum length of the generated sequence. (optional, default: self.max_seq_len)
        Returns:
            sequence (list): list of tokens (as strings) for the generated sequence.
        """
        self.eval()
        max_length = max_length if max_length is not None else self.max_seq_len

        # Start with a <bos> token
        input_ids = torch.LongTensor([[self.token2idx['<bos>']]]).to(self.device)

        # Generate the sequence iteratively
        with torch.no_grad():
            for _ in range(self.max_seq_len - 1):
                # 1. Predict token probabilities (for the last position)
                ...
                # 2. Sample the next token (look into torch.multinomial())
                next_token = ...
                # 3. Concatenate the sampled token to the input sequence
                # and stop if <eos> has been sampled
                input_ids = ...
                ...

        # Decode the sequence of indices back into tokens
        input_ids = input_ids.squeeze(0).detach().cpu().numpy()
        sequence = [self.idx2token[i] for i in input_ids]
        return sequence