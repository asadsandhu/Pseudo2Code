# gradio_app.py
import gradio as gr
import torch
import os
import math
import torch.nn as nn
import re
import sys
import asyncio

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
EMBED_DIM = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
FF_DIM = 512

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def tokenize_line(text: str):
    return re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)

def numericalize(text: str, stoi: dict):
    tokens = tokenize_line(text)
    return [stoi.get(tok, stoi[UNK_TOKEN]) for tok in tokens]

def pad_sequence(seq, max_len, pad_id):
    seq = seq[:max_len-1]
    seq = seq + [tgt_stoi[EOS_TOKEN]]
    if len(seq) < max_len:
        seq += [pad_id] * (max_len - len(seq))
    return seq

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    def forward(self, query, key, value, mask=None):
        B, Q_len, _ = query.size()
        B, K_len, _ = key.size()
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)
        Q = Q.view(B, Q_len, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(B, K_len, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(B, K_len, self.n_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1,2).contiguous().view(B, Q_len, self.d_model)
        return self.out_linear(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, src, src_mask=None):
        attn_out = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.ff(src)
        return self.norm2(src + self.dropout(ff_out))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.norm1(tgt + self.dropout(self.self_attn(tgt, tgt, tgt, mask=tgt_mask)))
        tgt = self.norm2(tgt + self.dropout(self.cross_attn(tgt, memory, memory, mask=memory_mask)))
        ff_out = self.ff(tgt)
        return self.norm3(tgt + self.dropout(ff_out))

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dim_feedforward) for _ in range(num_layers)])
    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dim_feedforward) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.fc_out(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads,
                 num_encoder_layers, num_decoder_layers, dim_feedforward):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, num_encoder_layers, dim_feedforward)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, num_decoder_layers, dim_feedforward)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        return self.decoder(tgt, memory, tgt_mask)

def generate_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask

def greedy_decode(model, src, src_stoi, tgt_stoi, tgt_itos, max_len=MAX_LEN):
    model.eval()
    src = torch.tensor(src, dtype=torch.long, device=DEVICE).unsqueeze(0)
    memory = model.encoder(src)
    ys = torch.tensor([tgt_stoi[SOS_TOKEN]], dtype=torch.long, device=DEVICE).unsqueeze(0)
    for i in range(max_len-1):
        tgt_mask = generate_subsequent_mask(ys.size(1)).to(DEVICE)
        out = model.decoder(ys, memory, tgt_mask)
        prob = out[:, -1, :]
        next_token = torch.argmax(prob, dim=1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], device=DEVICE)], dim=1)
        if next_token == tgt_stoi[EOS_TOKEN]:
            break
    out_tokens = ys.squeeze(0).tolist()[1:]
    if tgt_stoi[EOS_TOKEN] in out_tokens:
        out_tokens = out_tokens[:out_tokens.index(tgt_stoi[EOS_TOKEN])]
    return " ".join(tgt_itos[t] for t in out_tokens)

# Load model and vocabulary
if not os.path.exists("model.pth"):
    raise FileNotFoundError("Model file 'model.pth' not found. Please train first.")

checkpoint = torch.load("model.pth", map_location=DEVICE)
src_stoi = checkpoint['src_stoi']
src_itos = checkpoint['src_itos']
tgt_stoi = checkpoint['tgt_stoi']
tgt_itos = checkpoint['tgt_itos']

model = TransformerSeq2Seq(
    src_vocab_size=len(src_stoi),
    tgt_vocab_size=len(tgt_stoi),
    d_model=EMBED_DIM,
    n_heads=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=FF_DIM
).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def convert_pseudocode(text):
    lines = text.strip().split('\n')
    outputs = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            outputs.append("")
        elif line == "}":
            outputs.append("}")
        else:
            try:
                src_ids = numericalize(line, src_stoi)
                src_ids = pad_sequence(src_ids, MAX_LEN, src_stoi[PAD_TOKEN])
                output_line = greedy_decode(model, src_ids, src_stoi, tgt_stoi, tgt_itos)
                outputs.append(output_line)
            except Exception as e:
                outputs.append(f"// [Error in line {i+1}]: {e}")
    return "int main() {\n" + '\n'.join(outputs) + "\nreturn 0;\n}" 

iface = gr.Interface(
    fn=convert_pseudocode,
    inputs=gr.Textbox(label="Enter pseudocode (line-by-line)", lines=10),
    outputs=gr.Code(language="cpp", label="Generated C++ Code"),
    title="PseudoCode to C++ Converter (Transformer from Scratch)"
)

if __name__ == "__main__":
    iface.launch()
