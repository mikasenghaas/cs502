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
        self.d_model = d_model
        self.scale = d_model ** (-0.5)

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
        scores = self.scale * (query @ key.permute(0, 2, 1))

        if mask is not None:
            scores[mask == 0.0] = -float("inf")

        # Compute the attention scores
        attention_scores = F.softmax(scores, dim=-1)

        # Compute the weighted sum of the value vectors
        output = attention_scores @ value

        return output


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        """
        Args:
            d_model (int): dimension of the query, key and value vectors.
        """
        super().__init__()

        # Query, key and value projections
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(d_model)

        # Output projection
        self.Wo = nn.Linear(d_model, d_model)

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
        K = self.Wk(key)
        V = self.Wv(value)

        # Self-attention
        attention_output = self.attention(Q, K, V, mask=mask)

        # Output projection
        output = self.Wo(attention_output)

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

        # Save parameters
        self.d_model = d_model
        self.feedforward_dim = feedforward_dim

        # Attention block
        self.attention = SingleHeadAttention(d_model)

        # Feed-forward block
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model),
        )  # 2-layers feed-forward network with a ReLU activation

        # Layer normalisation and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

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
        attention_output = self.attention(x, x, x, mask=mask)

        # Residual connection and layer normalisation
        x = self.norm1(x + self.dropout(attention_output))

        # Feed-forward network
        ff_output = self.feed_forward(x)

        # Residual connection and layer normalisation
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        d_model,
        feedforward_dim,
        num_layers,
        dropout,
        device,
        max_seq_len,
        token2idx,
        idx2token,
    ):
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
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx, device=device
        )

        # Positional embedding
        self.pos_embedding = nn.Embedding(
            max_seq_len, d_model, padding_idx=pad_idx, device=device
        )

        # Transformer layers
        # Hint: we want "num_layers" transformer layers.
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(d_model, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

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
        assert (
            x_ids.shape[1] <= self.max_seq_len
        ), f"Input sequence is too long for the model! ({x_ids.shape[1]} > {self.max_seq_len})"

        # Embed the input tokens
        embedded_input = self.embedding(x_ids)

        # Compute positional embeddings
        positions = torch.arange(x_ids.shape[1], device=self.device)
        embedded_pos = self.pos_embedding(positions)

        # Add the two embeddings
        x = embedded_input + embedded_pos

        # Prepare the attention mask
        mask = get_mask(x_ids, self.pad_idx).to(x.device)

        # Apply the transformer layers
        for i, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(x, mask=mask)

        # Generate logits for the next token prediction
        logits = self.output_layer(x)

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
        input_ids = torch.LongTensor([[self.token2idx["<bos>"]]]).to(self.device)

        # Generate the sequence iteratively
        with torch.no_grad():
            for _ in range(self.max_seq_len - 1):
                # 1. Predict token probabilities (for the last position)
                logits = self(input_ids)
                final_logits = logits[:, -1, :]  # Keep only last token prediction
                probs = F.softmax(final_logits, dim=-1)

                # 2. Sample the next token (look into torch.multinomial())
                next_token = torch.multinomial(probs.squeeze(), 1)

                # 3. Concatenate the sampled token to the input sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

                # 4. Stop if <eos> has been sampled
                if next_token == self.token2idx["<eos>"]:
                    break

        # Decode the sequence of indices back into tokens
        input_ids = input_ids.squeeze(0).detach().cpu().numpy()
        sequence = [self.idx2token[i] for i in input_ids]
        return sequence
