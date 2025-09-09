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


import os
import random
import pickle
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

def load_facts(filename=os.path.join(os.path.dirname(__file__), "../Models/braindata.fi.pkl")):
    try:
        with open(filename, "rb") as file:
            facts = pickle.load(file)
        return facts
    except Exception as e:
        print(f"Virhe faktojen lukemisessa: {e}")
        log_error(f"Virhe faktojen lukemisessa: {e}")
        return []

def get_random_fact(facts):
    if isinstance(facts, dict) and "answers" in facts:
        answers = facts["answers"]
        if isinstance(answers, list) and answers:
            return random.choice(answers)
    elif isinstance(facts, list) and facts:
        return random.choice(facts)
    return "Fakta tietoa ei löytynyt."

def command_random_fact():
    facts = load_facts(os.path.join(os.path.dirname(__file__), "../Models/braindata.fi.pkl"))
    fact = get_random_fact(facts)
    return f"Fakta: {fact}"