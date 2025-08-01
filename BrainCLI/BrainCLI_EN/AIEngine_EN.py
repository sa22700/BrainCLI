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

from BrainCLI.BrainCLI_EN.DataManager_EN import SaveToFile
from BrainCLI.BrainCLI_EN.FuzzySearcher_EN import FuzzySearch
from BrainCLI.BrainCLI_EN.Vectorizer_EN import BrainVectorizer
from BrainCLI.BrainCLI_EN.MatrixArray_EN import BrainNetwork, BrainLayer
from BrainCLI.BrainCLI_EN.MarkovsChain_EN import build_markov_chain_from_data, generate_text
from BrainCLI.BrainCLI_EN.Utils_EN import normalize_text, preprocess_text, select_start_word


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a ** 2 for a in v1) ** 0.5
    mag2 = sum(b ** 2 for b in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

class AIEngine:
    def __init__(self, data_path):
        self.data_manager = SaveToFile(data_path)
        self.data = self.data_manager.load_pickle()
        self.vectorizer = BrainVectorizer()
        self.fuzzy_search = FuzzySearch(self)
        self.chain = build_markov_chain_from_data(data_path)
        self.nn = BrainNetwork([BrainLayer(300, 5), BrainLayer(5, 1)])
        self.vocabulary = None
        self.context = []

    def is_contextual(self, question, context):
        v1 = self.vectorizer.vectorize_text(question)
        v2 = self.vectorizer.vectorize_text(context)
        return cosine_similarity(v1, v2) > 0.85

    def get_response(self, user_input, context=None):
        cleaned_input = preprocess_text(user_input)
        user_input_norm = normalize_text(user_input)
        questions_norm = [normalize_text(q) for q in self.data["questions"]]

        if context and self.is_contextual(user_input, context):
            cleaned_input = f"{context} {cleaned_input}"
            user_input_norm = normalize_text(cleaned_input)

        if cleaned_input in questions_norm:
            index = questions_norm.index(user_input_norm)
            return self.data["answers"][index]

        best_match = self.fuzzy_search.performfuzzysearch(user_input_norm, questions_norm)
        if best_match:
            index = questions_norm.index(best_match)
            return self.data["answers"][index]

        vector = self.vectorizer.vectorize_text(cleaned_input)
        prediction = self.nn.array_predict([vector])

        outputs = [self.nn.array_predict([self.vectorizer.vectorize_text(q)]) for q in self.data["questions"]]

        def output_similarity(a, b):
            from math import sqrt
            a = a[0] if isinstance(a[0], list) else a
            b = b[0] if isinstance(b[0], list) else b
            dot = sum(i * j for i, j in zip(a, b))
            norm_a = sqrt(sum(i ** 2 for i in a))
            norm_b = sqrt(sum(i ** 2 for i in b))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        similarities = [output_similarity(prediction, out) for out in outputs]
        if similarities:
            best_idx = similarities.index(max(similarities))
            return self.data["answers"][best_idx]

        first_word = select_start_word(user_input_norm, self.chain)
        return generate_text(self.chain, start_word=first_word, length=10)

    def update_knowledge(self, question, answer):
        self.data_manager.save_to_pickle(question, answer)
        self.data = self.data_manager.load_pickle()
        self.chain = build_markov_chain_from_data(self.data_manager.pickle_file)