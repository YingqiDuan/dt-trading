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
        action_mode="discrete",
        use_value_head=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.action_mode = action_mode

        self.state_emb = nn.Linear(state_dim, d_model)
        self.rtg_emb = nn.Linear(1, d_model)
        if action_mode == "continuous":
            self.action_emb = nn.Linear(act_dim, d_model)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.action_emb = nn.Embedding(act_dim, d_model)
            self.log_std = None

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
        self.value_head = nn.Linear(d_model, 1) if use_value_head else None

    def forward(self, states, actions, rtg, return_values=False):
        B, T, _ = states.shape
        if T > self.seq_len:
            raise ValueError(f"sequence length {T} exceeds model max {self.seq_len}")

        # Embeddings
        rtg_emb = self.rtg_emb(rtg.unsqueeze(-1))
        state_emb = self.state_emb(states)

        if self.action_mode == "continuous":
            if actions.dim() == 2:
                actions = actions.unsqueeze(-1)
            act_emb = self.action_emb(actions.float())
        else:
            act_emb = self.action_emb(actions.long())

        # Stack & Reshape: (B, T, 3, D) -> (B, T*3, D)
        x = torch.stack((rtg_emb, state_emb, act_emb), dim=2).reshape(
            B, T * 3, self.d_model
        )
        x = x + self.pos_emb[:, : T * 3]

        # Causal Mask & Transformer
        mask = torch.triu(
            torch.ones(T * 3, T * 3, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(x, mask)

        # Predict actions using the 3rd token of each step (indices 2, 5, 8...)
        h_act = h[:, 2::3]
        logits = self.action_head(h_act)

        if self.value_head is None or not return_values:
            return logits

        return logits, self.value_head(h_act).squeeze(-1)
