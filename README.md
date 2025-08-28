# Surprisal Calculator with GPT-2

This project uses the GPT-2 language model to compute **word-level surprisal** (negative log-probability) scores for text data. It takes as input a CSV file with words and sentence identifiers, reconstructs full sentences, processes them through GPT-2, and outputs surprisal values per word.

---

## 🔧 Features

* Uses **GPT-2** from HuggingFace to get token-level log probabilities.
* Reconstructs full sentences from a CSV format.
* Computes **surprisal values** for each word by summing log-probabilities of its sub-tokens.
* Saves the output to new CSV files with added surprisal values.

---

## 📁 Directory Structure

```
project/
│
├── input_files/
│   └── <your_input>.csv
│
├── output_files/
│   └── <your_input>_processed.csv
│
├── config.json
├── main.py
└── README.md
```

---

## 📥 Input Format

Input CSV files (placed in `input_files/`) must contain the following columns:

| Passage | WordWithPunctuation |
| ------- | ------------------- |
| 1       | The                 |
| 1       | dog                 |
| 1       | ran.                |
| 2       | It                  |
| 2       | barked.             |

* `Passage`: An integer ID to identify sentence boundaries.
* `WordWithPunctuation`: The actual words in the sentence (with punctuation as needed).

---

## 📤 Output

Each processed file will be saved in `output_files/` with a new column:

| Passage | WordWithPunctuation | surprisal |
| ------- | ------------------- | --------- |
| 1       | The                 | 2.32      |
| 1       | dog                 | 4.01      |
| 1       | ran.                | 3.21      |
| 2       | It                  | 1.87      |
| 2       | barked.             | 3.74      |

* **surprisal**: The summed negative log-probabilities of sub-tokens composing the word.

---

## ⚙️ Configuration

The `config.json` file specifies the list of input file names to process:

```json
{
  "INPUT_FILE_NAMES": ["example1", "example2"]
}
```

Each entry should match a CSV filename in the `input_files/` directory (without the `.csv` extension).

---

## 🚀 Running the Script

Make sure dependencies are installed:

```bash
pip install torch transformers pandas
```

Then run:

```bash
python main.py
```

---

## 📌 Notes

* GPT-2 uses subword tokenization, so words may be split internally — surprisal values are computed by summing log-probabilities of subword parts.
* Ensure your inputs are cleanly formatted, with consistent sentence IDs.
* Results will differ slightly across model runs due to potential non-determinism unless fixed with seeds (not implemented here).