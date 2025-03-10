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

from BrainCLI.BrainCLI_FI.Degug_Log_FI import log_error

class BrainVectorizer:
    def __init__(self, vector_size=300):
        self.vector_size = vector_size
        self.word_vectors = {}

    @staticmethod
    def _custom_random(seed, min_val=-1.0, max_val=1.0):
        seed = (seed * 9301 + 49297) % 233280
        return min_val + (seed / 233280.0) * (max_val - min_val)

    def _generate_vector(self, word):
        seed = sum(ord(char) for char in word)
        return [self._custom_random(seed + i)
            for i in range(self.vector_size)]

    def vectorize_text(self, text):
        try:
            words = text.lower().split()
            vectors = [self.word_vectors.setdefault(word, self._generate_vector(word))
            for word in words]

            if vectors:
                avg_vector = [sum(vals) / len(vectors)
                for vals in zip(*vectors)]

            else:
                avg_vector = [0.0] * self.vector_size
            return avg_vector

        except Exception as e:
            print(f"Virhe tekstin vektoroinnissa: {e}")
            log_error(f"Virhe tekstin vektoroinnissa: {e}")
            return [0.0] * self.vector_size
