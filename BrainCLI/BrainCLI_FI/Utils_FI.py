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


import re
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

def normalize_text(text):
    try:
        return text.strip().lower()

    except Exception as e:
        print(f"Virhe tekstin normalisoinnissa: {e}")
        log_error(e)
        return text

def delete_stop_marks(text):
    try:
        return re.sub(r"[?!,.:;\"']", "", text).strip().lower()

    except Exception as e:
        print(f"Virhe stop merkkien poistossa: {e}")
        log_error(e)
        return text

def delete_stop_words(text):
    stop_words = [
        'ja', 'että', 'mutta', 'vaan', 'sekä', 'sillä', 'joten', 'koska', 'kun', 'jos', 'vaikka',
        'se', 'tämä', 'tuo', 'ne', 'nämä', 'nuo',
        'minä', 'sinä', 'hän', 'me', 'te', 'he',
        'joka', 'mikä', 'kuka', 'kenen', 'ketä', 'kenet', 'jotka',
        'niin', 'siis', 'kai', 'varmaan', 'ehkä', 'juuri', 'edes', 'ikinä', 'enää', 'vielä',
        'myös', 'aina', 'kuitenkin', 'jo', 'nyt', 'taas', 'tosin', 'ihan', 'muuten',
        'olla', 'ei', 'ole', 'oli', 'ovat', 'on', 'olisin', 'olit', 'ollut',
        'mitä', 'mikä', 'missä', 'milloin', 'kuinka', 'kuinka paljon', 'kumpi', 'kenen', 'miksi', 'kuinka', 'kuka'
    ]
    try:
        words = text.strip().lower().split()
        filtered = [w for w in words if w not in stop_words]
        return " ".join(filtered)

    except Exception as e:
        print(f"Virhe stop sanojen poistossa: {e}")
        log_error(e)
        return text

def preprocess_text(text, remove_stop_marks=True, remove_stop_words=True):
    try:
        if remove_stop_marks:
            text = delete_stop_marks(text)
        if remove_stop_words:
            text = delete_stop_words(text)
        return normalize_text(text)

    except Exception as e:
        print(f"Virhe tekstin esikäsittelyssä: {e}")
        log_error(e)
        return text

def select_start_word(user_input, markov_chain):
    try:
        cleaned_text = delete_stop_words(user_input)
        if not cleaned_text:
            cleaned_text = user_input
        words = cleaned_text.split()
        for word in words:
            if word in markov_chain:
                return word

    except Exception as e:
        print(f"Virhe tekstin käsittelyssä: {e}")
        log_error(e)

    return next(iter(markov_chain))