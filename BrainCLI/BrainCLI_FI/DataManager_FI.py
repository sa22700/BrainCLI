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


import pickle
import os
import stat
from BrainCLI.BrainCLI_FI.Utils_FI import normalize_text
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

class SaveToFile:
    def __init__(self, pickle_file=os.path.join(os.path.dirname(__file__), "../Models/braindata.fi.pkl")):
        try:
            self.pickle_file = pickle_file
            self.initialize_files()

        except Exception as e:
            print(f"Virhe SaveToFile-luokan alustuksessa: {e}")
            log_error(e)

    @staticmethod
    def _assert_safe_file(path: str) -> None:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            raise ValueError("Symlinkkiä ei ladattu")
        if (st.st_mode & stat.S_IWOTH) != 0:
            raise ValueError("Tiedostoa ei ladattu")

    def initialize_files(self):
        try:
            if not os.path.exists(self.pickle_file):
                with open(self.pickle_file, "wb") as f:
                    pickle.dump({"questions": [], "answers": []}, f)
                print(f"Luotiin uusi Pickle-tiedosto: {self.pickle_file}")

        except Exception as e:
            print(f"Virhe Pickle-tiedoston luonnissa: {e}")
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
            print(f"Virhe ladattaessa Pickle-tiedostoa: {e}")
            log_error(e)
            return {"questions": [], "answers": []}

    def save_to_pickle(self, question, answer):
        try:
            normalized_question = normalize_text(question)
            existing_data = self.load_pickle()
            if normalized_question in [normalize_text(q) for q in existing_data["questions"]]:
                print(f"Kysymys '{question}' on jo tallennettu.")
                return
            existing_data["questions"].append(normalized_question)
            existing_data["answers"].append(answer)
            with open(self.pickle_file, "wb") as f:
                pickle.dump(existing_data, f)

        except Exception as e:
            print(f"Virhe tallennettaessa Pickle-tiedostoon: {e}")
            log_error(e)

    @staticmethod
    def save_weights(brainnetwork, path):
        try:
            with open(path, "wb") as f:
                pickle.dump(brainnetwork.get_weights(), f)

        except Exception as e:
            print(f"Virhe tallennettaessa modelle: {e}")
            log_error(f"Virhe tallennettaessa modelle: {e}")

    @staticmethod
    def load_weights(brainnetwork, path):
        try:
            with open(path, "rb") as f:
                brainnetwork.set_weights(pickle.load(f))

        except Exception as e:
            print(f"Virhe ladattaessa modelle: {e}")
            log_error(f"Virhe ladattaessa modelle: {e}")