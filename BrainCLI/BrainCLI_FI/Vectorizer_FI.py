"""
Copyright [2025] [Pirkka Toivakka]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# This project uses model weights licensed under CC BY 4.0 (see /Models/LICENSE)


import math
from array import array
from random import Random
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

class BrainVectorizer:
    def __init__(self, vector_size: int = 300):
        self.vector_size = vector_size
        self.word_vectors: dict[str, array] = {}

    def _generate_vector(self, word: str) -> array:
        seed = sum(ord(c) for c in word)
        rnd = Random(seed)
        return array('d', (rnd.uniform(-1.0, 1.0) for _ in range(self.vector_size)))

    def vectorize_text(self, text: str) -> list[float]:
        try:
            words = text.lower().split()
            wv = self.word_vectors
            vecs: list[array] = []
            for w in words:
                if w in wv:
                    vecs.append(wv[w])
                else:
                    v = self._generate_vector(w)
                    wv[w] = v
                    vecs.append(v)
            if not vecs:
                return [0.0] * self.vector_size
            size = self.vector_size
            sum_vec = array('d', [0.0] * size)
            for vec in vecs:
                for i in range(size):
                    sum_vec[i] += vec[i]
            inv_n = 1.0 / len(vecs)
            for i in range(size):
                sum_vec[i] *= inv_n
            norm = math.sqrt(sum(x * x for x in sum_vec))
            if norm > 0:
                sum_vec = array('d', [x / norm for x in sum_vec])
            return sum_vec.tolist()

        except Exception as e:
            print(f"Virhe tekstin vektoroinnissa: {e}")
            log_error(f"Virhe tekstin vektoroinnissa: {e}")
            return [0.0] * self.vector_size
