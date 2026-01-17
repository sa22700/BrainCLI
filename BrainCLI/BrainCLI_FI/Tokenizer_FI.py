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


from collections import Counter
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error
import pickle
import os
import stat

def _assert_safe_file(path: str) -> None:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("Symlinkkiä ei ladattu")
    if (st.st_mode & stat.S_IWOTH) != 0:
        raise ValueError("Tiedostoa ei ladattu")

def gen_tokens(datasheet):
    all_text = " ".join(datasheet).lower()
    words = all_text.split()
    most_common = [w for w, _ in Counter(words).most_common(1000)]
    special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>", "<SEP>", "<CLS>"]
    return special_tokens + most_common

def load_tokens(_data_path: str) -> list[str]:
    _assert_safe_file(_data_path)
    with open(_data_path, "rb") as f:
        data = pickle.load(f)
    all_data = data.get("questions", []) + data.get("answers", [])
    return gen_tokens(all_data)

try:
    data_path = os.path.join(os.path.dirname(__file__), '../Models/braindata.fi.pkl')
    tokens = load_tokens(data_path)

except Exception as e:
    print(f"Virhe generoidessa tokeneja: {e}")
    log_error(e)
    tokens = ["<PAD>", "<START>", "<END>", "<UNK>", "<SEP>", "<CLS>"]
