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

def tokens():
    return [
        "<PAD>", "<START>", "<END>", "<UNK>", "<SEP>", "<CLS>",

        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",

        "i", "you", "he", "she", "we", "they",
        "am", "are", "is", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "not", "and", "but", "or", "if", "when", "then", "so", "also",
        "the", "a", "an", "this", "that", "these", "those", "what", "why", "where", "because", "how",
        "yes", "no", "know", "can", "want", "part", "say", "make", "go", "went", "come", "came",
        "do", "does", "doing", "make", "made", "making", "get", "good", "bad", "big", "small", "old", "young",
        "man", "woman", "child", "cat", "dog", "car", "house", "school", "book", "work", "friend", "family",
        "day", "week", "year", "month", "time", "hour", "minute", "second", "now", "then", "always", "sometimes",
        "before", "after", "here", "there", "home", "city", "country", "finland", "english", "speak", "write",
        "read", "listen", "watch", "know", "understand", "remember", "forget", "ask", "answer",

        ".", ",", "!", "?", ":", ";", "'", '"', "-", "_", "(", ")", "[", "]", "{", "}", "/", "\\", "@", "#", "$", "%",
        "&", "*", "+", "=", "<", ">", "|", "^", "~", "`", " ", "\n", "\t"
    ]