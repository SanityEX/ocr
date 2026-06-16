import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d((2, 1))
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(256, 384),
            ResidualBlock(384, 384),
            nn.MaxPool2d((2, 1))
        )

        self.layer5 = nn.Sequential(
            ResidualBlock(384, 512),
            nn.Conv2d(
                512,
                512,
                kernel_size=(3, 3),
                padding=(0, 1),
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.squeeze(2)
        x = x.permute(2, 0, 1)

        return x

class EncoderRNN(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2
        )

    def forward(self, x):
        outputs, hidden = self.rnn(x)
        return outputs, hidden

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, embed_size=256):
        super().__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.enc_dim = hidden_size * 2

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.emb_dropout = nn.Dropout(0.3)

        self.attn = nn.Linear(
            self.enc_dim + hidden_size + embed_size,
            hidden_size
        )
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.LSTM(
            input_size=embed_size + self.enc_dim,
            hidden_size=hidden_size,
            num_layers=1
        )

        self.out = nn.Linear(hidden_size + self.enc_dim, vocab_size)

    def forward_step(self, input_token, hidden, encoder_outputs):
        T, B, _ = encoder_outputs.size()

        embedded = self.embedding(input_token)
        embedded = self.emb_dropout(embedded)

        h = hidden[0][-1]

        h_repeat = h.unsqueeze(0).repeat(T, 1, 1)
        emb_repeat = embedded.unsqueeze(0).repeat(T, 1, 1)

        attn_input = torch.cat(
            [encoder_outputs, h_repeat, emb_repeat],
            dim=2
        )

        energy = torch.tanh(self.attn(attn_input))
        scores = self.attn_score(energy).squeeze(2)
        attn_weights = F.softmax(scores, dim=0)

        context = torch.sum(
            encoder_outputs * attn_weights.unsqueeze(2),
            dim=0
        )

        rnn_input = torch.cat(
            [embedded, context],
            dim=1
        ).unsqueeze(0)

        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(0)

        logits = self.out(
            torch.cat([output, context], dim=1)
        )

        return logits, hidden, attn_weights

    def init_hidden_from_encoder(self, encoder_hidden):
        h, c = encoder_hidden

        h_dec = (h[-2] + h[-1]).unsqueeze(0)
        c_dec = (c[-2] + c[-1]).unsqueeze(0)

        return h_dec, c_dec

class AttentionOCR(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, embed_size=256):
        super().__init__()

        self.encoder_cnn = EncoderCNN()
        self.encoder_rnn = EncoderRNN(
            input_size=512,
            hidden_size=hidden_size
        )
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            embed_size=embed_size
        )

    def encode(self, images):
        feats = self.encoder_cnn(images)
        encoder_outputs, encoder_hidden = self.encoder_rnn(feats)
        decoder_hidden = self.decoder.init_hidden_from_encoder(encoder_hidden)
        return encoder_outputs, decoder_hidden

    def forward(self, images, target_input):
        encoder_outputs, hidden = self.encode(images)

        B, L = target_input.size()
        logits_all = []

        for t in range(L):
            input_token = target_input[:, t]
            logits, hidden, _ = self.decoder.forward_step(
                input_token,
                hidden,
                encoder_outputs
            )
            logits_all.append(logits.unsqueeze(1))

        logits_all = torch.cat(logits_all, dim=1)
        return logits_all

    @torch.no_grad()
    def predict(self, images, sos_idx, eos_idx, max_len=25):
        encoder_outputs, hidden = self.encode(images)

        B = images.size(0)

        input_token = torch.full(
            (B,),
            sos_idx,
            dtype=torch.long,
            device=images.device
        )

        results = []

        for _ in range(max_len):
            logits, hidden, _ = self.decoder.forward_step(
                input_token,
                hidden,
                encoder_outputs
            )

            pred = logits.argmax(dim=1)
            results.append(pred.unsqueeze(1))
            input_token = pred

        results = torch.cat(results, dim=1)
        return results

    @torch.no_grad()
    def predict_beam(
        self,
        images,
        sos_idx,
        eos_idx,
        beam_size=5,
        max_len=25,
        length_penalty=0.7
    ):
        self.eval()

        encoder_outputs, hidden = self.encode(images)

        B = images.size(0)
        if B != 1:
            raise ValueError("predict_beam only supports batch_size=1")

        beams = [
            {
                "tokens": [sos_idx],
                "score": 0.0,
                "hidden": hidden,
                "finished": False
            }
        ]

        for _ in range(max_len):
            new_beams = []

            for beam in beams:
                tokens = beam["tokens"]
                score = beam["score"]
                beam_hidden = beam["hidden"]
                finished = beam["finished"]

                if finished:
                    new_beams.append(beam)
                    continue

                input_token = torch.tensor(
                    [tokens[-1]],
                    dtype=torch.long,
                    device=images.device
                )

                logits, new_hidden, _ = self.decoder.forward_step(
                    input_token,
                    beam_hidden,
                    encoder_outputs
                )

                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_log_probs, top_indices = torch.topk(
                    log_probs,
                    beam_size
                )

                for log_p, idx in zip(top_log_probs, top_indices):
                    idx = idx.item()
                    log_p = log_p.item()

                    new_beams.append({
                        "tokens": tokens + [idx],
                        "score": score + log_p,
                        "hidden": new_hidden,
                        "finished": idx == eos_idx
                    })

            def normalized_score(b):
                length = max(1, len(b["tokens"]))
                return b["score"] / (length ** length_penalty)

            new_beams = sorted(
                new_beams,
                key=normalized_score,
                reverse=True
            )

            beams = new_beams[:beam_size]

            if all(b["finished"] for b in beams):
                break

        best = max(
            beams,
            key=lambda b: b["score"] / (
                max(1, len(b["tokens"])) ** length_penalty
            )
        )

        result = best["tokens"][1:]

        return torch.tensor(
            result,
            dtype=torch.long,
            device=images.device
        ).unsqueeze(0)
