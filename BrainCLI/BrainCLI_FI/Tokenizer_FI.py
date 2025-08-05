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
    # 1. Sanasto (täydennä tarpeesi mukaan)
    return [
        # Erikoistokenit
        "<PAD>", "<START>", "<END>", "<UNK>", "<SEP>", "<CLS>",

        # Numerot
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",

        # Suomi: yleisimmät sanat (100 kpl)
        "minä", "sinä", "hän", "me", "te", "he",
        "on", "ei", "ja", "mutta", "tai", "jos", "kun", "että", "niin", "myös", "kuten",
        "se", "ne", "tämä", "tuo", "nämä", "nuo", "mikä", "miksi", "missä", "koska", "kuinka",
        "kyllä", "en", "tiedä", "olen", "oot", "voin", "haluan", "osa", "sanoa", "tehdä",
        "voit", "tulee", "mennä", "mennyt", "tule", "tulen", "tekee", "tehdään", "saada",
        "hyvä", "huono", "suuri", "pieni", "vanha", "nuori", "mies", "nainen", "lapsi",
        "kissa", "koira", "auto", "talo", "koulu", "kirja", "työ", "ystävä", "perhe", "päivä",
        "viikko", "vuosi", "kuukausi", "aika", "tunti", "hetki", "minuutti", "sekunti",
        "nyt", "sitten", "aina", "joskus", "ennen", "jälkeen", "täällä", "tuolla", "kotona",
        "kaupungissa", "maalla", "suomi", "englanti", "puhua", "kirjoittaa", "lukea", "kuunnella",
        "katsoa", "tietää", "ymmärtää", "muistaa", "unohtaa", "kysyä", "vastata",

        # Perusmerkit ja välimerkit
        ".", ",", "!", "?", ":", ";", "'", '"', "-", "_", "(", ")", "[", "]", "{", "}", "/", "\\", "@", "#", "$", "%",
        "&", "*", "+", "=", "<", ">", "|", "^", "~", "`", " ", "\n", "\t"
    ]