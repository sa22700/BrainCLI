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
from BrainCLI.BrainCLI_EN.Utils_EN import normalize_text
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error

class SaveToFile:
    def __init__(self, pickle_file=os.path.join(os.path.dirname(__file__), "../Models/braindata.en.pkl")):
        try:
            self.pickle_file = pickle_file
            self.initialize_files()

        except Exception as e:
            print(f"Error with SaveToFile-class: {e}")

    def initialize_files(self):
        try:
            if not os.path.exists(self.pickle_file):
                with open(self.pickle_file, "wb") as f:
                    pickle.dump({"questions": [], "answers": []}, f)
                print(f"Created new Pickle-file: {self.pickle_file}")

        except Exception as e:
            print(f"Error initializing files: {e}")
            log_error(e)

    def load_pickle(self):
        try:
            if not os.path.exists(self.pickle_file):
                return {"questions": [], "answers": []}
            with open(self.pickle_file, "rb") as f:
                data = pickle.load(f)
            if "questions" not in data or "answers" not in data:
                return {"questions": [], "answers": []}
            return data
        except Exception as e:
            print(f"Error loading Pickle-file: {e}")
            log_error(e)
            return {"questions": [], "answers": []}

    def save_to_pickle(self, question, answer):
        try:
            normalized_question = normalize_text(question)
            existing_data = self.load_pickle()
            if normalized_question in [normalize_text(q) for q in existing_data["questions"]]:
                print(f"Question '{question}' is already saved.")
                return
            existing_data["questions"].append(normalized_question)
            existing_data["answers"].append(answer)
            with open(self.pickle_file, "wb") as f:
                pickle.dump(existing_data, f)

        except Exception as e:
            print(f"Error saving to Pickle-file: {e}")
            log_error(e)

    def save_weights(self, brainnetwork, path):
        with open(path, "wb") as f:
            pickle.dump(brainnetwork.get_weights(), f)

    def load_weights(self, brainnetwork, path):
        with open(path, "rb") as f:
            brainnetwork.set_weights(pickle.load(f))
