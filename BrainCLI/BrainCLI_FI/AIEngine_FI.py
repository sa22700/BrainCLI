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


import difflib
import time
from BrainCLI.BrainCLI_FI.Utils_FI import normalize_text
from BrainCLI.BrainCLI_FI.DataManager_FI import SaveToFile
from BrainCLI.BrainCLI_FI.Vectorizer_FI import BrainVectorizer
from BrainCLI.BrainCLI_FI.MatrixArray_FI import BrainNetwork, BrainLayer
from BrainCLI.BrainCLI_FI.Decoder_FI import decode
from BrainCLI.BrainCLI_FI.FuzzySearcher_FI import FuzzySearch
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a ** 2 for a in v1) ** 0.5
    mag2 = sum(b ** 2 for b in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

def find_close_match(input_norm, questions_norm, cutoff=0.8):
    matches = difflib.get_close_matches(input_norm, questions_norm, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    return None

class AIEngine:
    def __init__(self, data_path):
        self.data_manager = SaveToFile(data_path)
        self.data = self.data_manager.load_pickle()
        self.vectorizer = BrainVectorizer()
        self.nn = BrainNetwork([BrainLayer(300, 128), BrainLayer(128, 300)])
        self.fuzzy_search = FuzzySearch(self)
        self.vocabulary = None
        self.context = []

    def get_response(self, user_input, threshold=0.95):
        try:
            user_input_norm = normalize_text(user_input)
            questions_norm = [normalize_text(q) for q in self.data["questions"]]
            if user_input_norm in questions_norm:
                idx = questions_norm.index(user_input_norm)
                return self.data["answers"][idx]
            best_match = self.fuzzy_search.performfuzzysearch(user_input_norm, self.data["questions"])
            if best_match:
                idx = self.data["questions"].index(best_match)
                return self.data["answers"][idx]
            input_vec = self.vectorizer.vectorize_text(user_input)
            output_vec = self.nn.array_predict([input_vec])[0]
            best_idx, best_score = None, -1
            for idx, answer in enumerate(self.data["answers"]):
                a_vec = self.vectorizer.vectorize_text(answer)
                score = cosine_similarity(output_vec, a_vec)
                if score > best_score:
                    best_idx, best_score = idx, score
            if best_score > threshold:
                return self.data["answers"][best_idx]
            else:
                return f"{decode(output_vec)}"

        except Exception as e:
            print(f"Virhe vastaamisessa: {e}")
            log_error(f"Virhe vastaamisessa: {e}")
            return "Virhe vastaamisessa"

    def train_network(self, epochs=1, learning_rate=0.00001):
        questions = self.data["questions"]
        answers = self.data["answers"]
        if not questions or not answers:
            print("Ei koulutettavaa dataa!")
            return
        for epoch in range(epochs):
            total_loss = 0
            N = len(questions)
            start_time = time.time()
            for i, (q, a) in enumerate(zip(questions, answers)):
                try:
                    q_vec = self.vectorizer.vectorize_text(q)
                    a_vec = self.vectorizer.vectorize_text(a)
                    input_matrix = [q_vec]
                    target_matrix = [a_vec]
                    self.nn.train(input_matrix, target_matrix, learning_rate=learning_rate)
                    prediction = self.nn.array_predict(input_matrix)
                    pred_vec = prediction[0] if isinstance(prediction[0], list) else prediction
                    loss = sum((pi - ai) ** 2 for pi, ai in zip(pred_vec, a_vec)) / len(a_vec)
                    total_loss += loss

                except Exception as e:
                    print(f"Virhe koulutuksessa: {e}")
                    log_error(f"Virhe koulutuksessa: {e}")
                    continue

                if (i % (N // 100 + 1) == 0) or (i == N - 1):
                    percent = int(100 * (i + 1) / N)
                    elapsed = time.time() - start_time
                    if i > 0:
                        eta = (elapsed / (i + 1)) * (N - i - 1)
                        mins, secs = divmod(int(eta), 60)
                        eta_str = f"{mins:02d}:{secs:02d}"
                    else:
                        eta_str = "--:--"
                    print(f"\rEpoch {epoch + 1}/{epochs}: {percent} % (ETA: {eta_str})", end="")
                print()
            print(f"Epoch {epoch + 1}/{epochs}, loss: {total_loss / N}")

    def update_knowledge(self, question, answer):
        try:
            self.data_manager.save_to_pickle(question, answer)
            self.data = self.data_manager.load_pickle()

        except Exception as e:
            print(f"Virhe tallennuksessa: {e}")
            log_error(f"Virhe tallennuksessa: {e}")