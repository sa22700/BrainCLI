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

import pickle
import os
import difflib
from BrainCLI.BrainCLI_FI.MatrixArray_FI import BrainNetwork, BrainLayer, BrainMatrix
from BrainCLI.BrainCLI_FI.Vectorizer_FI import BrainVectorizer
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

class BrainPredictor:
    def __init__(self):
        self.vectorizer = BrainVectorizer()
        self.categories = ["Tuntematon"]
        self.model = BrainNetwork([BrainLayer(input_size=300, output_size=1)])
        self.data = self.load_data()
        # Cache original and normalized questions
        self.questions = list(self.data.keys())
        self.norm_questions = [q.strip().lower() for q in self.questions]

    @staticmethod
    def load_data():
        try:
            path = os.path.join(os.path.dirname(__file__), "../Models/braindata.fi.pkl")
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            log_error("Pickle-tiedostoa ei löytynyt")
            return {}
        except Exception as e:
            print(f"Virhe ladattaessa dataa: {e}")
            log_error(e)
            return {}

    def fuzzy_match(self, user_input):
        # Strip & lower once
        query = user_input.strip().lower()
        if not self.questions:
            return None
        # Fast C-backed fuzzy match
        match = difflib.get_close_matches(query, self.norm_questions, n=1, cutoff=0.0)
        if match:
            idx = self.norm_questions.index(match[0])
            return self.questions[idx]
        return None

    def predict(self, question):
        try:
            vector = self.vectorizer.vectorize_text(question)
            prediction = self.model.array_predict([vector])
            # Ensure list-of-lists of floats
            if isinstance(prediction, BrainMatrix):
                prediction = prediction.to_list()
            elif isinstance(prediction, (int, float)):
                prediction = [[float(prediction)]]
            elif isinstance(prediction, list) and all(isinstance(x, (int, float)) for x in prediction):
                prediction = [[float(x)] for x in prediction]
            elif not isinstance(prediction, list):
                raise TypeError(f"Virhe array_predict: odotettiin lista, saatiin {type(prediction)}")

            best = self.fuzzy_match(question)
            if best:
                response = self.data.get(best, "En löytänyt vastausta.")
            else:
                response = "En osaa sanoa."

            return {"prediction": prediction, "match": best, "response": response}
        except Exception as e:
            log_error(e)
            return {"error": f"Virhe ennustuksessa: {e}"}
