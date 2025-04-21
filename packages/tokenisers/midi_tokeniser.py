# midi_tokenizer.py
import miditoolkit
import os

class MidiTokenizer:
    def __init__(self, time_division=10):
        self.time_division = time_division  # in ms
        self.vocab = {
            "PAD": 0, "START": 1, "END": 2,
        }
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.build_vocab()

    def build_vocab(self):
        idx = len(self.vocab)
        for i in range(128):
            self.vocab[f"NOTE_ON_{i}"] = idx
            self.rev_vocab[idx] = f"NOTE_ON_{i}"
            idx += 1
        for i in range(128):
            self.vocab[f"VELOCITY_{i}"] = idx
            self.rev_vocab[idx] = f"VELOCITY_{i}"
            idx += 1
        for i in range(1, 500):  # 0.01s to 5s
            self.vocab[f"TIME_SHIFT_{i}"] = idx
            self.rev_vocab[idx] = f"TIME_SHIFT_{i}"
            idx += 1

    def encode(self, midi_path):
        midi_obj = miditoolkit.MidiFile(midi_path)
        events = []
        for note in sorted(midi_obj.instruments[0].notes, key=lambda x: x.start):
            time_shift = int(note.start * 1000 // self.time_division)
            events.append(f"TIME_SHIFT_{time_shift}")
            events.append(f"NOTE_ON_{note.pitch}")
            events.append(f"VELOCITY_{note.velocity}")
        tokens = [self.vocab["START"]] + [self.vocab.get(e, self.vocab["PAD"]) for e in events] + [self.vocab["END"]]
        return tokens

    def decode(self, tokens):
        # Not implemented here — just token to string
        return [self.rev_vocab.get(t, "UNK") for t in tokens]

    @property
    def vocab_size(self):
        return len(self.vocab)
