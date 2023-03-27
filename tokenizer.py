import torch



class Tokenizer:
    """Tokenizer for LLaMA."""

    def __init__(self, vocab_file: str, bos_id: int = 1, eos_id: int = 2):
        self.bos_id = bos_id
        self.eos_id = eos_id
        # with open(vocab_file, "rb") as file:
        #     # exactly one token per line
        #     self.vocab = file.readlines()
        self.vocab = torch.load(vocab_file)
        self.max_tok_length = max(len(t) for t in self.vocab)
        print(self.vocab[15043], self.vocab[29892], self.vocab[590], self.vocab[1024], self.vocab[338])

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, string: str, bos: bool = True, eos: bool = False) -> torch.Tensor:
        tokens = []
        s = 0
        while s < len(string):
            for t in reversed(range(s + 1, s + self.max_tok_length)):
                
                maybe_token = string[s:t]
                # print("trying", s, t, maybe_token)

                try:
                    idx = self.vocab.index(maybe_token)
                    tokens.append(idx)
                    s += len(maybe_token)
                    break
                except ValueError:
                    continue

        # if bos:
        #     tokens = [self.bos_id] + tokens
        # if eos:
        #     tokens = tokens + [self.eos_id]
        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join([self.vocab[idx] for idx in tokens.tolist()])
