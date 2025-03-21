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

import pickle
import os
import random
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error

def load_brain_data(file_path = os.path.join(os.path.dirname(__file__), "../Models/braindata.en.pkl")):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

    except Exception as e:
        print("Error loading braindata.en.pkl:", e)
        log_error(f"Error loading braindata.en.pkl: {e}")
        return {"questions": [], "answers": []}

def build_markov_chain(text):
    words = text.split()
    markov_chain = {}
    for i in range(len(words) - 1):
        key = words[i]
        next_word = words[i + 1]
        if key in markov_chain:
            markov_chain[key].append(next_word)
        else:
            markov_chain[key] = [next_word]
    return markov_chain

def build_markov_chain_from_data(file_path):
    data = load_brain_data(file_path)
    answers = data.get("answers", [])
    combined_text = " ".join(answers)
    return build_markov_chain(combined_text)

def update_brain_data(file_path, question, answer):
    data = load_brain_data(file_path)
    questions = data.get("questions", [])
    answers = data.get("answers", [])
    normalized_q = question.lower().strip()
    found = False
    for i, q in enumerate(questions):
        if normalized_q == q.lower().strip():
            answers[i] = answer
            found = True
            break
    if not found:
        questions.append(question)
        answers.append(answer)
    data["questions"] = questions
    data["answers"] = answers

    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print("Update successful!")

    except Exception as e:
        print("Error updating pickle file:", e)
        log_error(f"Error updating pickle file: {e}")

def generate_text(markov_chain, start_word, length=10):
    current_word = start_word
    result = [current_word]
    for _ in range(length - 1):
        next_words = markov_chain.get(current_word, None)
        if not next_words:
            break

        current_word = random.choice(next_words)
        result.append(current_word)
    return " ".join(result)
