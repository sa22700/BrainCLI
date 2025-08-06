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
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error

def normalize_text(text):
    try:
        return text.strip().lower()

    except Exception as e:
        print(f"Error normalizing text: {e}")
        log_error(e)
        return text

def delete_stop_marks(text):
    try:
        return re.sub(r"[?!,.:;\"]", "", text).strip().lower()

    except Exception as e:
        print(f"Error deleting stop marks: {e}")
        log_error(e)
        return text

def delete_stop_words(text):
    stop_words = [
        'a', 'an', 'the',
        'and', 'or', 'but', 'if', 'because', 'as',
        'of', 'at', 'by', 'for', 'with', 'about', 'between',
        'to', 'from', 'in', 'on', 'out', 'over', 'under',
        'again', 'then', 'once',
        'when', 'where', 'why', 'how',
        'all', 'any', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'can', 'will', 'just',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had',
        'do', 'does', 'did',
        'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their',
        'what', 'which', 'who', 'whom'
    ]
    try:
        words = text.strip().lower().split()
        filtered = [w for w in words if w not in stop_words]
        return " ".join(filtered)

    except Exception as e:
        print(f"Error removing stop words: {e}")
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
        print(f"Error during preprocessing: {e}")
        log_error(e)
        return text

def select_start_word(user_input):
    try:
        cleaned_text = delete_stop_words(user_input)
        if not cleaned_text:
            cleaned_text = user_input
        words = cleaned_text.split()
        for word in words:
            if word:
                return word

    except Exception as e:
        print(f"Error selecting start word: {e}")
        log_error(e)

    return None