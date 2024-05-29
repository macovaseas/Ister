# Ister
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.Autoformer_EncDec import series_decomp


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(output_dim, bias=bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.mlp = MLP(configs.d_model, configs.d_model // 2, configs.d_model)

    def forward(self, season_enc, trend_enc):
        season_out, attns = self.encoder(season_enc, attn_mask=None)
        trend_out = trend_enc + self.mlp(trend_enc)
        return season_out, trend_out


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.season_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)
        self.trend_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        # Encoder
        self.backbone = Backbone(configs)
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.Linear_Seasonal = nn.Linear(configs.d_model, self.pred_len, bias=True)
            self.Linear_Trend = nn.Linear(configs.d_model, self.pred_len, bias=True)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([self.pred_len, configs.d_model]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([self.pred_len, configs.d_model]))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Decomposition
        seasonal_init, trend_init = self.decomp(x_enc)
        # Embedding
        season_out = self.season_embedding(seasonal_init, x_mark_enc)
        trend_out = self.trend_embedding(trend_init, x_mark_enc)
        # Encoder
        season_out, trend_out = self.backbone(season_out, trend_out)
        # Decoder
        season_out = self.Linear_Seasonal(season_out).permute(0, 2, 1)[:, :, :N]
        trend_out = self.Linear_Trend(trend_out).permute(0, 2, 1)[:, :, :N]
        dec_out = season_out + trend_out
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Only for LSTF tasks
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None
