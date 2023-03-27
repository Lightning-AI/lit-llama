from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from tokenizer import Tokenizer
import torch
from tqdm import tqdm

tok = SentencePieceProcessor("checkpoints/llama/tokenizer.model")


print(tok.encode("Hello, my name is"))
print(tok.encode("Hello"))
print(tok.decode([15043]))

#print(tok.encode("Hello, my name is"))
#print(tok2.encode("Hello, my name is"))

words = []
for i in tqdm(range(tok.vocab_size()), total=tok.vocab_size()):
    if i == 15043:
        print(tok.decode([i]))
    words.append(tok.decode([i]))
    # print(words[-1])

# with open("data/vocab.txt", "wb") as f:
#     for w in words:
#         f.write((w + "\n").encode())
torch.save(words, "data/vocab.pt")


tok2 = Tokenizer("data/vocab.pt")
print(tok2.vocab[15043])
print(tok2.encode("Hello, my name is"))
print(tok.decode([10994]))

# spm = SentencePieceProcessor(model_file='checkpoints/llama/tokenizer.model')
# SentencePieceProcessor.LoadVocabulary




# trained = SentencePieceTrainer.train(input='vocab.txt', model_prefix='tok', vocab_size=tok.vocab_size)
