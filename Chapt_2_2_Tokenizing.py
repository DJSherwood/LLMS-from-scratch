## Block 1
import urllib.request
import tiktoken
from SimpleTokenizer import SimpleTokenizerV2

# url = ("https://raw.githubusercontent.com/rasbt/"
#        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#        "the-verdict.txt")
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

## Block 2
import re
preprocessed = re.split(r'([,.:?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

## Block 3
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

## Block 4
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

tokenizer = SimpleTokenizerV2(vocab)
text = """It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

# Block 5
tokenizer