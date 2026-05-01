from tokenizer import SimpleTokenizer
from dataset import load_data

data = load_data("training_data/llm_tutor/tutor_train.jsonl")

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(data)

sample = data[0]

encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Original:", sample[:100])
print("Encoded:", encoded[:20])
print("Decoded:", decoded[:100])