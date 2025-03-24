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


from BrainCLI.BrainCLI_FI.DataManager_FI import SaveToFile
from BrainCLI.BrainCLI_FI.FuzzySearcher_FI import FuzzySearch
from BrainCLI.BrainCLI_FI.Vectorizer_FI import BrainVectorizer
from BrainCLI.BrainCLI_FI.MatrixArray_FI import BrainNetwork, BrainLayer
from BrainCLI.BrainCLI_FI.MarkovsChain_FI import build_markov_chain_from_data, generate_text
from BrainCLI.BrainCLI_FI.Utils_FI import normalize_text, delete_stop_marks

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

    def get_response(self, user_input):
        user_input_norm = normalize_text(user_input)
        questions_norm = [normalize_text(q) for q in self.data["questions"]]
        if user_input_norm in questions_norm:
            index = questions_norm.index(user_input_norm)
            return self.data["answers"][index]

        best_match = self.fuzzy_search.performfuzzysearch(user_input_norm, questions_norm)
        if best_match:
            index = questions_norm.index(best_match)
            return self.data["answers"][index]

        del_user_input = delete_stop_marks(user_input_norm)

        vector = self.vectorizer.vectorize_text(user_input_norm)
        prediction = self.nn.array_predict([vector])

        if prediction[0][0] > 0.5:
            return "En osaa vastata varmasti, mutta neuroverkkoni arvioi asian olevan todennäköinen."

        first_word = del_user_input.split()[0] if user_input.split()[0] in self.chain else next(iter(self.chain))
        return generate_text(self.chain, start_word=first_word, length=10)

    def update_knowledge(self, question, answer):
        self.data_manager.save_to_pickle(question, answer)
        self.data = self.data_manager.load_pickle()
        self.chain = build_markov_chain_from_data(self.data_manager.pickle_file)
