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

from Utils import normalize_text


class FuzzySearch:
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine

    @staticmethod
    def levenshteindistance(source, target):
        try:
            if not source or not target:
                return float('inf')

            source = normalize_text(source)
            target = normalize_text(target)
            m, n = len(source), len(target)

            if m < n:
                source, target = target, source
                m, n = n, m
            prev_row = list(range(n + 1))

            for i in range(1, m + 1):
                curr_row = [i] + [0] * n
                for j in range(1, n + 1):
                    cost = 0 \
                        if source[i - 1] == target[j - 1] \
                        else 1
                    curr_row[j] = min(
                        prev_row[j] + 1,
                        curr_row[j - 1] + 1,
                        prev_row[j - 1] + cost)
                prev_row = curr_row
            return prev_row[-1]

        except Exception as e:
            print(f"Virhe Levenshtein-etäisyyden laskennassa: {e}")
            return float('inf')

    def performfuzzysearch(self, query, questions):
        try:
            if not questions:
                return None

            query = normalize_text(query)
            distances = [
                (question, self.levenshteindistance(query, normalize_text(question)))
                for question in questions
                if question]

            if not distances:
                return None

            best_match, best_distance = min(distances, key=lambda x: x[1])
            max_distance_threshold = max(2, int(len(query) * 0.2))

            if best_distance > max_distance_threshold:
                return None
            return best_match

        except Exception as e:
            print(f"Virhe fuzzy-haussa: {e}")
            return None