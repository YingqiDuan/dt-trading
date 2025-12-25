import torch
from torch import nn


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        seq_len,
        d_model=128,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.state_emb = nn.Linear(state_dim, d_model)
        self.rtg_emb = nn.Linear(1, d_model)
        self.action_emb = nn.Embedding(act_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len * 3, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, act_dim)

    def forward(self, states, actions, rtg):
        # states: (B, T, state_dim), actions: (B, T), rtg: (B, T)
        bsz, tlen, _ = states.shape
        if tlen > self.seq_len:
            raise ValueError(f"sequence length {tlen} exceeds model max {self.seq_len}")

        rtg_tokens = self.rtg_emb(rtg.unsqueeze(-1))
        state_tokens = self.state_emb(states)
        action_tokens = self.action_emb(actions)

        tokens = torch.stack((rtg_tokens, state_tokens, action_tokens), dim=2)
        tokens = tokens.reshape(bsz, tlen * 3, self.d_model)
        tokens = tokens + self.pos_emb[:, : tlen * 3, :]

        mask = torch.triu(
            torch.ones(tlen * 3, tlen * 3, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(tokens, mask)
        action_positions = torch.arange(2, tlen * 3, 3, device=tokens.device)
        action_hidden = hidden[:, action_positions, :]
        logits = self.action_head(action_hidden)
        return logits
