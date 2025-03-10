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

import unicodedata
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error

def normalize_text(text):
    try:
        text = text.strip().lower()
        normalized = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn')
        return unicodedata.normalize('NFC', normalized)

    except Exception as e:
        print(f"Error normalizing text: {e}")
        log_error(e)
        return text
