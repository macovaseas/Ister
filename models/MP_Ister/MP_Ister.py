import torch
import torch.nn as nn
import torch.fft
from models.MP_Ister.layers.Embed import PositionEmbedding, TemporalEmbedding, Multiscale_InvertedEmbedding
from models.MP_Ister.layers.Ister_EncDec import Encoder
from utils.tools import series_decomp

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.top_k = configs.top_k
        self.peri_mid_const = Multiscale_InvertedEmbedding(configs)
        self.feature_flows_dim = configs.feature_flows_dim
        self.peri_midformer = Encoder(configs)
        self.position_embedding = PositionEmbedding(configs)
        self.temporal_embedding = TemporalEmbedding(configs)
        self.layer = configs.layers
        self.decompsition = series_decomp(configs.moving_avg)
        self.projection_trend = nn.Linear(self.seq_len, self.label_len + self.pred_len)
        self.dropout = nn.Dropout(p=configs.dropout)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(self.d_model * self.top_k, self.label_len + self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        seasonal_part, trend_part = self.decompsition(x_enc)
        enc_in = seasonal_part

        # temporal embedding
        if x_mark_enc is not None:
            enc_in = self.temporal_embedding(enc_in, x_mark_enc)

        # Mapping trend part to target length
        trend_part = trend_part.permute(0, 2, 1)
        trend_part = self.projection_trend(trend_part)
        trend_part = trend_part.permute(0, 2, 1)

        enc_in, components_per_level = self.peri_mid_const(enc_in)

        enc_in = self.position_embedding(enc_in)

        enc_out, attns = self.peri_midformer(enc_in, components_per_level, self.configs.task_name)

        enc_out_dim = enc_out.shape[-1]
        target_dim = self.d_model * self.top_k
        if enc_out_dim < target_dim:
            padding_size = target_dim - enc_out_dim
            padding = torch.zeros((enc_out.shape[:-1] + (padding_size,)), device=enc_out.device)
            enc_out = torch.cat((enc_out, padding), dim=-1)

        periodic_feature_flows = self.projection(enc_out)
        periodic_feature_flows_aggration = torch.mean(periodic_feature_flows, dim=-2)
        dec_out = periodic_feature_flows_aggration.permute(0, 2, 1)

        # add trend part
        dec_out = dec_out + trend_part

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.configs.label_len + self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.configs.label_len + self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out  # [B, L, C]