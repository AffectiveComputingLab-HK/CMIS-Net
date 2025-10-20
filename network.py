from torch import nn

def perturb_tensor_by_label_distance(
    x,
    labels,                 # shape (B,) or (B,1)
    mode=0,                 # 0~4
    noise_std=0.05,         # base noise
    drop_prob=0.1,          # base dropout
    noise_center=0.25,      # 想拟合的中心 label 值
    max_dyn_noise=0.1,
    min_drop=0.1,
    max_drop=0.5
):
    """
    Add perturbation based on how far labels are from a target center (default=0.25).

    The further the label from center, the stronger the noise and dropout.
    """
    if mode == 0:
        return x
    
    # with torch.no_grad():
    if labels.dim() == 1:
        labels = labels.unsqueeze(-1)

    # Compute |label - center|
    label_distance = torch.abs(labels - noise_center)  # shape (B, 1)

    # Normalize to [0, 1]
    norm_dist = label_distance / (label_distance.max() + 1e-6)

    # Expand to match x shape
    while norm_dist.dim() < x.dim():
        norm_dist = norm_dist.unsqueeze(-1)

    # --- Noise ---
    if mode in [1, 3]:  # 动态 noise
        dyn_std = torch.clamp(noise_std * norm_dist, max=max_dyn_noise)
        x = x + torch.randn_like(x) * dyn_std
        # print("dy std")
    elif mode in [2, 4]:  # 固定 noise
        x = x + torch.randn_like(x) * noise_std
        # print("fixed std")
    # mode 0 → no noise

    # --- Dropout ---
    if mode in [1, 4]:  # 固定 dropout
        mask = (torch.rand_like(x) > drop_prob).float()
        # print("fixed prob")
    elif mode in [2, 3]:  # 动态 dropout
        dyn_drop_prob = max_drop - norm_dist * (max_drop - min_drop)
        mask = (torch.rand_like(x) > dyn_drop_prob).float()
        # print("dy prob")

    x = x * mask

    return x


class GRU_Encoder(nn.Module):
    """GRU-based encoder"""
    def __init__(self, input_dim=620, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = torch.mean(out, dim=1)
        return out

class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim=620, hidden_dim=64, num_layers=2, num_heads=4, noise=False, is_dynoise=False, noise_std=0.05, drop_prob=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # 映射到 Transformer 维度
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.noise=noise
        self.is_dynoise=is_dynoise
        self.noise_std=noise_std
        self.drop_prob=drop_prob

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)  # 变成 (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)  # Transformer 处理
        x = x.permute(1, 0, 2)  # 变成 (batch_size, seq_len, hidden_dim)
        if self.training and self.noise:
            x = perturb_tensor_by_label_distance(x, y, self.is_dynoise, self.noise_std, self.drop_prob, 0.25)
        # x = torch.mean(x, dim=0)  # 取所有时间步的平均值
        return x


class Frame_Level_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise, is_dynoise=False, noise_std=0.05, drop_prob=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim//2)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim//2)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim//2)

        self.layernorm = nn.LayerNorm(hidden_dim//2)
        self.output_linear = nn.Linear(hidden_dim//2, hidden_dim)  # map回输入维度

        self.noise=noise
        self.is_dynoise=is_dynoise
        self.noise_std=noise_std
        self.drop_prob=drop_prob

    def forward(self, x,y=None):  # x: (B, T, input_dim)
        x = self.mlp(x)  # (B, T, hidden_dim)

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        b, t, c = Q.size()
        outputs = []
        for i in range(t):
            q = Q[:, i, :].unsqueeze(1)  # (B,1,C)
            k = K[:, i, :].unsqueeze(2)  # (B,C,1)
            v = V[:, i, :].unsqueeze(1)  # (B,1,C)
            attn_scores = torch.bmm(q, k)
            attn_weights = torch.sigmoid(attn_scores)
            context = attn_weights * v
            context = context.squeeze(1)
            outputs.append(context)

        out = torch.stack(outputs, dim=1)  # (B,T,C)
        out = self.layernorm(out)
        out = self.output_linear(out)  # map回 input_dim
        if self.training and self.noise:
            out = perturb_tensor_by_label_distance(out, y, self.is_dynoise, self.noise_std, self.drop_prob, 0.25)
        return out



class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, use_nonlinear=True, dropout=0.1, return_attn=False, ave_method='mean'):
        super(TemporalAttention, self).__init__()
        self.use_nonlinear = use_nonlinear
        self.return_attn = return_attn
        self.ave_method = ave_method

        self.attn_layer = nn.Linear(feature_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        single_input = False
        if x.dim() == 2:
            # reshape (T, D) -> (1, T, D)
            x = x.unsqueeze(0)
            single_input = True

        # (B, T, 1)
        attn_scores = self.attn_layer(x)

        if self.use_nonlinear:
            attn_scores = torch.tanh(attn_scores)

        # softmax along time dimension
        # attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = torch.sigmoid(attn_scores)
        attn_weights = self.dropout(attn_weights)

        # (B, T, D) * (B, T, 1) -> (B, T, D)
        weighted = x * attn_weights

        # (B, D)
        if self.ave_method == 'temp_attn_sum':
            out = weighted.sum(dim=1)
        elif self.ave_method == 'temp_attn_mean':
            out = weighted.mean(dim=1)

        return out.squeeze(0) if single_input else out

class GRU_Classifier(nn.Module):
    """GRU-based classifier"""
    def __init__(self, input_dim=64, hidden_dim=32, num_classes=1, ave_method='mean'):
        super().__init__()
        self.ave_method = ave_method
        self.ave_layer = TemporalAttention(input_dim, use_nonlinear=True, dropout=0.1, return_attn=False, ave_method=ave_method)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.SELU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if self.ave_method == 'mean':
            x = torch.mean(x, dim=1)
        elif self.ave_method == 'temp_attn_sum' or self.ave_method == 'temp_attn_mean':
            x = self.ave_layer(x)
        
        # x = torch.mean(x, dim=1)
        feature = self.relu(self.fc1(x))
        feature = self.dropout(feature)
        feature = self.fc2(feature)
        out = self.tanh(feature)
        return out, feature

class AttentionTransLayer(nn.Module):
    """Attention transformation layer"""
    def __init__(self, dinp, dout):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(dinp, dinp//2),
            nn.ReLU(),
            nn.Linear(dinp//2, dinp),
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.Linear(dinp, dout),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.attention(x) * x
        out += x
        out = self.fc(x)
        return out

class origin_Translator(nn.Module):
    """Original translator model"""
    def __init__(self, input_dim=64):
        super().__init__()
        self.layer1 = AttentionTransLayer(input_dim, input_dim)
        self.layer2 = AttentionTransLayer(input_dim, input_dim)
        self.layer3 = AttentionTransLayer(input_dim, input_dim)
        self.layer4 = AttentionTransLayer(input_dim, input_dim)

    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return out

class Attention(nn.Module):
    """Attention mechanism for encoder-decoder models"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value):
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Compute attention weights using dot product
        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.bmm(attention_weights, value)
        return context, attention_weights

import torch
import torch.nn as nn

class FrameAttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.q_linear = nn.Linear(input_dim, hidden_dim)
        self.k_linear = nn.Linear(input_dim, hidden_dim)
        self.v_linear = nn.Linear(input_dim, hidden_dim)

        self.output_linear = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (B, T, D) → 逐帧处理
        B, T, D = x.shape
        x_ = x.view(-1, D)  # (B*T, D)

        Q = self.q_linear(x_)
        K = self.k_linear(x_)
        V = self.v_linear(x_)

        attn_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2))  # (B*T, 1, 1)
        attn_weights = torch.sigmoid(attn_scores)                # scalar gate
        context = attn_weights * V.unsqueeze(1)                  # (B*T, 1, hidden_dim)
        context = context.squeeze(1)                             # (B*T, hidden_dim)

        out = self.output_linear(context)                        # (B*T, input_dim)
        out = self.norm(out + x_)                                # residual + norm
        return out.view(B, T, D)

class FrameLevelTranslator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            FrameAttentionBlock(input_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x):  # x: (B, T, D)
        for layer in self.layers:
            x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, translator_mode):
        super(Encoder, self).__init__()
        if translator_mode == 'ED-LSTM':
            self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif translator_mode == 'ED-GRU':
            self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        elif translator_mode == 'ED-Transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.linear = nn.Linear(input_size, hidden_size)  # 线性映射到隐藏维度
    
    def forward(self, x):
        if isinstance(self.encoder, nn.TransformerEncoder):
            x = self.linear(x)  # 变换输入维度
            x = x.permute(1, 0, 2)  # Transformer 需要 (seq_len, batch_size, hidden_size)
            output = self.encoder(x)
            return output.permute(1, 0, 2)  # 变回 (batch_size, seq_len, hidden_size)
        else:
            output, _ = self.encoder(x)
            return output

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, translator_mode):
        super(Decoder, self).__init__()
        self.attention = Attention(hidden_size)  # 使用注意力机制
        if translator_mode == 'ED-LSTM':
            self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif translator_mode == 'ED-GRU':
            self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif translator_mode == 'ED-Transformer':
            decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            self.linear = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_output, query):
        if isinstance(self.decoder, nn.TransformerDecoder):
            query = query.permute(1, 0, 2)
            encoder_output = encoder_output.permute(1, 0, 2)
            output = self.decoder(query, encoder_output)
            return self.linear(output.permute(1, 0, 2)), None
        else:
            context, attention_weights = self.attention(query, encoder_output, encoder_output)
            output, _ = self.decoder(context)
            output = self.fc(output)
            return output, attention_weights

class Translator(nn.Module):
    """Translator model combining encoder and decoder"""
    def __init__(self, input_size, hidden_size, output_size, translator_mode):
        super(Translator, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, translator_mode)
        self.decoder = Decoder(hidden_size, output_size, translator_mode)
    
    def forward(self, input_sequence):
        # Encode input sequence
        encoder_output = self.encoder(input_sequence)
        
        # Use encoder output as query for decoder
        decoder_output, _ = self.decoder(encoder_output, encoder_output)
        
        return decoder_output