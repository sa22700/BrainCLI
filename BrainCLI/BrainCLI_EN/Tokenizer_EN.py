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


from collections import Counter
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error
import pickle
import os

def gen_tokens(data):
    all_text = " ".join(data).lower()
    words = all_text.split()
    most_common = [w for w, _ in Counter(words).most_common(1000)]
    special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>", "<SEP>", "<CLS>"]
    return special_tokens + most_common

try:
    data_path = os.path.join(os.path.dirname(__file__), '../Models/braindata.en.pkl')
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    all_data = data["questions"] + data["answers"]
    tokens = gen_tokens(all_data)

except Exception as e:
    print(f"Error generating tokens: {e}")
    log_error(e)
    tokens = ["<PAD>", "<START>", "<END>", "<UNK>", "<SEP>", "<CLS>"]
