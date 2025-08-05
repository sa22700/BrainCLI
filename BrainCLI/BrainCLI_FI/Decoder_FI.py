'''
Copyright [2025] [Pirkka Toivakka]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
# This project uses model weights licensed under CC BY 4.0 (see /Models/LICENSE)

import math
import random
from BrainCLI.BrainCLI_FI.Tokenizer_FI import tokens

token2id = {tok: i for i, tok in enumerate(tokens())}
id2token = {i: tok for tok, i in token2id.items()}
vocab_size = len(tokens())
hidden_size = 32

def rand_mat(rows, cols, scale=0.1):
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

def rand_vec(size, scale=0.1):
    return [random.uniform(-scale, scale) for _ in range(size)]

# Painot (pysyvät session ajan; generatiivinen demo)
W_embed = rand_mat(vocab_size, hidden_size)
W_hh = rand_mat(hidden_size, hidden_size)
W_in = rand_mat(hidden_size, hidden_size)
b_h = [0.0] * hidden_size
W_out = rand_mat(hidden_size, vocab_size)
b_out = [0.0] * vocab_size

def matvec_dot(mat, vec):
    return [sum(a*b for a, b in zip(row, vec)) for row in mat]

def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def vec_tanh(v):
    return [math.tanh(x) for x in v]

def softmax(logits):
    maxlog = max(logits)
    expv = [math.exp(x - maxlog) for x in logits]
    s = sum(expv)
    return [v/s for v in expv]

def argmax(probs):
    maxv = max(probs)
    for i, p in enumerate(probs):
        if p == maxv:
            return i
    return 0

def decode(vektori, max_len=10):
    h = list(vektori)
    token_id = token2id["<START>"]
    output = []
    for _ in range(max_len):
        x = W_embed[token_id]
        h = vec_tanh(vec_add(matvec_dot(W_in, x), vec_add(matvec_dot(W_hh, h), b_h)))
        logits = vec_add(matvec_dot(W_out, h), b_out)
        probs = softmax(logits)
        token_id = argmax(probs)
        word = id2token[token_id]
        if word == "<END>":
            break
        output.append(word)
    return " ".join(output)
