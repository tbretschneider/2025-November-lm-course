import torch

from ttlm.tokenizer.base import Tokenizer

class BPETokenizer(Tokenizer):
    """A simple BPE tokenizer."""

    @property
    def bos_token_id(self) -> int:
        return 128

    @property
    def bos_token(self) -> str:
        return "<BOS>"

    @property
    def eos_token_id(self) -> int:
        return 129

    @property
    def eos_token(self) -> str:
        return "<EOS>"

    @property
    def pad_token_id(self) -> int:
        return 130

    @property
    def pad_token(self) -> str:
        return "<PAD>"

    @property
    def unk_token_id(self) -> int:
        return 131

    @property
    def unk_token(self) -> str:
        return "<UNK>"

    @property
    def vocab_size(self) -> int:
        # 128 BPE + 4 special tokens
        return 132
    
    def train(self, texts: list[str]) -> None:
        """We do need to train the tokenizer for BPE."""
        num_merges = 500
        text = " ".join(texts)
        vocab = set(text)
        tokens = list(text)

        for _ in range(num_merges):
            pairs = {}
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] = pairs.get(pair,0) + 1

            if not pairs: break

            best_pair = max(pairs, key=pairs.get)
            new_tokens = []
            i=0
            while i < len(tokens):
                if i < len(tokens)-1 and \
                    (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(
                        tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            vocab.add(best_pair[0] + best_pair[1])
        self.tokens = list(vocab)
        self.vocab = vocab

    def encode(
        self, strings: list[str], bos: bool = True, eos: bool = True
    ) -> list[torch.LongTensor]:
        """Encodes a batch of strings to their BPE values."""
        encoded = []
        for s in strings:
            max_len = max((len(t) for t in self.vocab), default = 0)
            indices = [i for i in range(len(s))] 
            encodedtokens = [None for i in range(len(s))] 
            for i in range(len(s)):
                for j in range(max_len,0):
                    if s[i:i+j] in self.vocab:
                        s = s[:i] + s[i+j:]
                        encodedtokens[i] = self.tokens.index(s[i:i+j])
                        indices = indices[:i]+indices[i+j:]

            tokens = [x for x in encodedtokens if x is not None]
            if bos:
                tokens = [self.bos_token_id] + tokens
            if eos:
                tokens.append(self.eos_token_id)
            encoded.append(torch.tensor(tokens, dtype=torch.long))
        return encoded

    def decode(
        self, tokens: list[list[int]], special_tokens: bool = False
    ) -> list[str]:
        """Decodes a batch of BPE values to strings."""
        decoded = []
        special_tokens_to_remove = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        for token_list in tokens:
            if not special_tokens:
                token_list = [t for t in token_list if t not in special_tokens_to_remove]
            chars = []
            for token in token_list:
                if token < 1000:
                    chars.append(self.tokens[token])
                elif token == self.bos_token_id:
                    chars.append("<BOS>")
                elif token == self.eos_token_id:
                    chars.append("<EOS>")
                elif token == self.unk_token_id:
                    chars.append("<UNK>")
                elif token == self.pad_token_id:
                    chars.append("<PAD>")
            decoded.append("".join(chars))
        return decoded
