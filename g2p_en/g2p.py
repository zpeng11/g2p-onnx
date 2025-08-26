# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p
'''
from nltk import pos_tag
from nltk.corpus import cmudict
import nltk
from nltk.tokenize import TweetTokenizer
word_tokenize = TweetTokenizer().tokenize
import numpy as np
import torch
import codecs
import re
import os
import unicodedata
from builtins import str as unicode
from .expand import normalize_numbers

# For ONNX export and runtime
import torch.nn as nn
import onnxruntime
import torch.onnx

try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict')

dirname = os.path.dirname(__file__)

def construct_homograph_dictionary():
    f = os.path.join(dirname,'homographs.en')
    homograph2features = dict()
    for line in codecs.open(f, 'r', 'utf8').read().splitlines():
        if line.startswith("#"): continue # comment
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features

# def segment(text):
#     '''
#     Splits text into `tokens`.
#     :param text: A string.
#     :return: A list of tokens (string).
#     '''
#     print(text)
#     text = re.sub('([.,?!]( |$))', r' \1', text)
#     print(text)
#     return text.split()

class G2p(object):
    def __init__(self):
        super().__init__()
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                             'EY2', 'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 'OW0', 'OW1',
                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 'UW',
                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.cmu = cmudict.dict()
        self.load_variables()
        self.homograph2features = construct_homograph_dictionary()

    def load_variables(self):
        self.variables = np.load(os.path.join(dirname,'checkpoint20.npz'))
        self.enc_emb = self.variables["enc_emb"]  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = self.variables["enc_w_ih"]  # (3*128, 64)
        self.enc_w_hh = self.variables["enc_w_hh"]  # (3*128, 128)
        self.enc_b_ih = self.variables["enc_b_ih"]  # (3*128,)
        self.enc_b_hh = self.variables["enc_b_hh"]  # (3*128,)

        self.dec_emb = self.variables["dec_emb"]  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = self.variables["dec_w_ih"]  # (3*128, 64)
        self.dec_w_hh = self.variables["dec_w_hh"]  # (3*128, 128)
        self.dec_b_ih = self.variables["dec_b_ih"]  # (3*128,)
        self.dec_b_hh = self.variables["dec_b_hh"]  # (3*128,)
        self.fc_w = self.variables["fc_w"]  # (74, 128)
        self.fc_b = self.variables["fc_b"]  # (74,)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
        rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0  # initial hidden state
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)

        return x

    def predict(self, word):
        # encoder
        enc = self.encode(word)
        enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh,
                       self.enc_b_ih, self.enc_b_hh, h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
        last_hidden = enc[:, -1, :]

        # decoder
        dec = np.take(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for i in range(20):
            h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)  # (b, h)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3: break  # 3: </s>
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else: # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]


class G2p_torch(object):
    def __init__(self):
        super().__init__()
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                             'EY2', 'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 'OW0', 'OW1',
                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 'UW',
                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.cmu = cmudict.dict()
        self.load_variables()
        self.homograph2features = construct_homograph_dictionary()

    def load_variables(self):
        variables = np.load(os.path.join(dirname, 'checkpoint20.npz'))
        self.enc_emb = torch.from_numpy(variables["enc_emb"]).float()
        self.enc_w_ih = torch.from_numpy(variables["enc_w_ih"]).float()
        self.enc_w_hh = torch.from_numpy(variables["enc_w_hh"]).float()
        self.enc_b_ih = torch.from_numpy(variables["enc_b_ih"]).float()
        self.enc_b_hh = torch.from_numpy(variables["enc_b_hh"]).float()

        self.dec_emb = torch.from_numpy(variables["dec_emb"]).float()
        self.dec_w_ih = torch.from_numpy(variables["dec_w_ih"]).float()
        self.dec_w_hh = torch.from_numpy(variables["dec_w_hh"]).float()
        self.dec_b_ih = torch.from_numpy(variables["dec_b_ih"]).float()
        self.dec_b_hh = torch.from_numpy(variables["dec_b_hh"]).float()
        self.fc_w = torch.from_numpy(variables["fc_w"]).float()
        self.fc_b = torch.from_numpy(variables["fc_b"]).float()

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = torch.matmul(x, w_ih.T) + b_ih
        rzn_hh = torch.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
        rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]

        rz = torch.sigmoid(rz_ih + rz_hh)
        r, z = torch.chunk(rz, 2, -1)

        n = torch.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = torch.zeros((x.shape[0], w_hh.shape[1]), dtype=torch.float32)
        h = h0  # initial hidden state
        outputs = torch.zeros((x.shape[0], steps, w_hh.shape[1]), dtype=torch.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = self.enc_emb[torch.LongTensor(x).unsqueeze(0)]

        return x

    def predict(self, word):
        # encoder
        enc = self.encode(word)
        enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh,
                       self.enc_b_ih, self.enc_b_hh, h0=torch.zeros((1, self.enc_w_hh.shape[-1]), dtype=torch.float32))
        last_hidden = enc[:, -1, :]

        # decoder
        dec = self.dec_emb[[2]]  # 2: <s>
        h = last_hidden

        preds = []
        for i in range(20):
            h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)  # (b, h)
            logits = torch.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            pred_idx = pred.item()
            if pred_idx == 3: break  # 3: </s>
            preds.append(pred_idx)
            dec = self.dec_emb[[pred_idx]]

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else: # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]


class G2pEncoder(nn.Module):
    def __init__(self, enc_emb, enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh):
        super().__init__()
        self.enc_emb = nn.Parameter(enc_emb, requires_grad=False)
        self.gru = nn.GRU(input_size=enc_emb.shape[1], hidden_size=enc_w_hh.shape[1], batch_first=True)
        
        # Set GRU parameters manually
        self.gru.weight_ih_l0.data = enc_w_ih
        self.gru.weight_hh_l0.data = enc_w_hh
        self.gru.bias_ih_l0.data = enc_b_ih
        self.gru.bias_hh_l0.data = enc_b_hh
    
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        x = self.enc_emb[input_ids]  # (batch_size, seq_len, emb_dim)
        _, hidden = self.gru(x)  # hidden: (1, batch_size, hidden_dim)
        return hidden.squeeze(0)  # (batch_size, hidden_dim)


class G2pDecoderStep(nn.Module):
    def __init__(self, dec_emb, dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh, fc_w, fc_b):
        super().__init__()
        self.dec_emb = nn.Parameter(dec_emb, requires_grad=False)
        self.gru_cell = nn.GRUCell(input_size=dec_emb.shape[1], hidden_size=dec_w_hh.shape[1])
        self.fc = nn.Linear(dec_w_hh.shape[1], fc_w.shape[0])
        
        # Set parameters manually
        self.gru_cell.weight_ih.data = dec_w_ih
        self.gru_cell.weight_hh.data = dec_w_hh
        self.gru_cell.bias_ih.data = dec_b_ih
        self.gru_cell.bias_hh.data = dec_b_hh
        self.fc.weight.data = fc_w
        self.fc.bias.data = fc_b
    
    def forward(self, input_id, hidden):
        # input_id: (batch_size,) - single token
        # hidden: (batch_size, hidden_dim)
        x = self.dec_emb[input_id]  # (batch_size, emb_dim)
        new_hidden = self.gru_cell(x, hidden)  # (batch_size, hidden_dim)
        logits = self.fc(new_hidden)  # (batch_size, vocab_size)
        return new_hidden, logits


def export_onnx_models(g2p_torch_instance, export_dir=dirname):
    import os
    os.makedirs(export_dir, exist_ok=True)
    
    # Export encoder
    encoder = G2pEncoder(
        g2p_torch_instance.enc_emb,
        g2p_torch_instance.enc_w_ih,
        g2p_torch_instance.enc_w_hh,
        g2p_torch_instance.enc_b_ih,
        g2p_torch_instance.enc_b_hh
    )
    
    # Example input for encoder (max length word)
    dummy_input = torch.randint(0, len(g2p_torch_instance.graphemes), (1, 30))  # batch_size=1, max_len=30
    
    torch.onnx.export(
        encoder,
        dummy_input,
        f"{export_dir}/g2p_encoder.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['hidden_state'],
        dynamic_axes={'input_ids': {1: 'seq_len'}}  # Allow variable sequence length
    )
    
    # Export decoder step
    decoder_step = G2pDecoderStep(
        g2p_torch_instance.dec_emb,
        g2p_torch_instance.dec_w_ih,
        g2p_torch_instance.dec_w_hh,
        g2p_torch_instance.dec_b_ih,
        g2p_torch_instance.dec_b_hh,
        g2p_torch_instance.fc_w,
        g2p_torch_instance.fc_b
    )
    
    # Example inputs for decoder step
    dummy_input_id = torch.randint(0, len(g2p_torch_instance.phonemes), (1,))  # batch_size=1
    dummy_hidden = torch.randn(1, g2p_torch_instance.dec_w_hh.shape[1])  # batch_size=1, hidden_dim
    
    torch.onnx.export(
        decoder_step,
        (dummy_input_id, dummy_hidden),
        f"{export_dir}/g2p_decoder_step.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_id', 'hidden_state'],
        output_names=['new_hidden_state', 'logits']
    )
    
    print(f"ONNX models exported to {export_dir}/")


class G2pOnnx(object):
    def __init__(self, onnx_dir=dirname):
        super().__init__()
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                             'EY2', 'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 'OW0', 'OW1',
                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 'UW',
                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}
        
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        
        self.cmu = cmudict.dict()
        self.homograph2features = construct_homograph_dictionary()
        
        # Load ONNX models
        self.encoder_session = onnxruntime.InferenceSession(
            os.path.join(onnx_dir, 'g2p_encoder.onnx'),
            providers=['CPUExecutionProvider']
        )
        self.decoder_session = onnxruntime.InferenceSession(
            os.path.join(onnx_dir, 'g2p_decoder_step.onnx'),
            providers=['CPUExecutionProvider']
        )
    
    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        return np.array([x], dtype=np.int64)  # batch_size=1
    
    def predict(self, word):
        # encoder
        input_ids = self.encode(word)
        hidden_state = self.encoder_session.run(['hidden_state'], {'input_ids': input_ids})[0]
        
        # decoder
        current_token = np.array([2], dtype=np.int64)  # Start with <s> token
        hidden = hidden_state.astype(np.float32)
        
        preds = []
        for _ in range(20):  # max length
            new_hidden, logits = self.decoder_session.run(
                ['new_hidden_state', 'logits'],
                {'input_id': current_token, 'hidden_state': hidden}
            )
            
            pred_idx = np.argmax(logits, axis=1)[0]
            if pred_idx == 3:  # </s> token
                break
                
            preds.append(pred_idx)
            current_token = np.array([pred_idx], dtype=np.int64)
            hidden = new_hidden
        
        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds
    
    def __call__(self, text):
        # Same preprocessing as G2p_torch
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else: # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]


if __name__ == '__main__':
    texts = ["I have $250 in my pocket.", # number -> spell-out
             "popular pets, e.g. cats and dogs", # e.g. -> for example
             "I refuse to collect the refuse around here.", # homograph
             "I'm an activationist."] # newly coined word
    
    print("=== Testing original G2p (numpy) ===")
    g2p = G2p()
    for text in texts:
        out = g2p(text)
        print(f"Input: {text}")
        print(f"Output: {out}\n")
    
    print("=== Testing G2p_torch ===")
    g2p_torch = G2p_torch()
    for text in texts:
        out = g2p_torch(text)
        print(f"Input: {text}")
        print(f"Output: {out}\n")
    
    print("=== Exporting ONNX models ===")
    export_onnx_models(g2p_torch)
    
    print("=== Testing G2pOnnx ===")
    try:
        g2p_onnx = G2pOnnx()
        for text in texts:
            out = g2p_onnx(text)
            print(f"Input: {text}")
            print(f"Output: {out}\n")
        
        # Test specific OOV word prediction comparison
        test_word = "activationist"
        torch_pred = g2p_torch.predict(test_word)
        onnx_pred = g2p_onnx.predict(test_word)
        print(f"=== Comparing predictions for '{test_word}' ===")
        print(f"Torch: {torch_pred}")
        print(f"ONNX:  {onnx_pred}")
        print(f"Match: {torch_pred == onnx_pred}")
        
    except Exception as e:
        print(f"ONNX test failed: {e}")
        print("Make sure you have onnxruntime installed: pip install onnxruntime")

