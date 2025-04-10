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
from BrainCLI.BrainCLI_FI.Utils_FI import normalize_text
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

class FuzzySearch:
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine

    def performfuzzysearch(self, query, questions):
        try:
            if not questions:
                return None

            query_norm = normalize_text(query)
            normalized_questions = [normalize_text(q) for q in questions if q]
            matches = difflib.get_close_matches(query_norm, normalized_questions, n=1, cutoff=0.8)
            if matches:
                best_match_norm = matches[0]
                index = normalized_questions.index(best_match_norm)
                return questions[index]
            return None

        except Exception as e:
            print(f"Virhe fuzzy-haussa: {e}")
            log_error(f"Virhe fuzzy-haussa: {e}")
            return None