** 🧠 BrainCLI

BrainCLI on komentorivikäyttöliittymä tekoälymoottorille, joka osaa hakea vastauksia, oppia uusia kysymyksiä ja suorittaa ennustuksia.

** 🚀 Ominaisuudet
- 🔍 **Tunnistaa ja tallentaa kysymyksiä ja vastauksia.
- 🤖 **Käyttää epätäsmällistä hakua (fuzzy search) oikeiden vastausten löytämiseen.
- 🧮 **Neuroverkkoennustukset matriisilaskennan avulla.
- 📝 **Tekstin vektorointi parantaa tietojen käsittelyä.

---

** 📥 Asennus ja käyttö

** 1️⃣ Kloonaa repository:

git clone https://github.com/sa22700/BrainCLI.git

cd BrainCLI

---

** ⚙ PowerShellin enkoodauksen muuttaminen UTF-8:ksi (PyPy-käyttäjille)
Jos käytät PyPy:tä, PowerShellin oletusmerkistö voi aiheuttaa virheitä ääkkösillä (ä, ö, å).

Korjaa tämä muuttamalla PowerShell käyttämään UTF-8-koodausta pysyvästi.

Avaa PowerShellin profiilitiedosto.

notepad $PROFILE

Jos tiedostoa ei ole, PowerShell luo sen automaattisesti.

Lisää rivit:

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Tallenna tiedosto ja sulje Muistio.

käynnistä powershell uudestaan.

Nyt PowerShell käyttää aina UTF-8-koodausta, ja ääkköset pitäisi toimia oikein myös PyPy:llä.

---

** Aja komennolla:

python Main.py

pypy Main.py

Jos käytät PyPy:tä, varmista että olet asettanut PowerShellin UTF-8-koodauksen kuten yllä.

---

** Tämä projekti on lisensoitu Apache License 2.0 -ehdoilla.
Katso lisätiedot tiedostosta LICENSE.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

** 🧠 BrainCLI

BrainCLI is a command-line interface for an AI engine that can retrieve answers, learn new questions, and make predictions.

** 🚀 Features
- 🔍 **Recognizes and stores questions and answers.
- 🤖 **Uses fuzzy search to find the correct answers.
- 🧮 **Performs neural network predictions using matrix calculations.
- 📝 **Text vectorization improves data processing.

---

** 📥 Installation & Usage

** 1️⃣ Clone the repository:

git clone https://github.com/sa22700/BrainCLI.git

cd BrainCLI

--

** ⚙ Fix UTF-8 Issues in PowerShell (For PyPy Users)
If you are using PyPy, PowerShell’s default encoding may cause issues with special characters (ä, ö, å).

Fix this by forcing PowerShell to use UTF-8 permanently.

Open PowerShell profile file

notepad $PROFILE

If the file does not exist, PowerShell will create it automatically.

Add these lines.

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Save the file and close Notepad.

Restart PowerShell.

Now PowerShell will always use UTF-8, and special characters should work correctly in PyPy.

---

** Run the command:

python Main.py

pypy Main.py

---

** This project is licensed under the Apache License 2.0.
See the LICENSE file for more details.
