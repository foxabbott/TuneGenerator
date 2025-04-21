# data_loader.py
import os
from torch.utils.data import Dataset, DataLoader
import torch
from packages.tokenisers.midi_tokeniser import MidiTokenizer
import numpy as np

class MaestroMIDIDataset(Dataset):
    def __init__(self, midi_dir, years, tokenizer, max_len=512, min_len=32):
        self.files = []
        for year in years:
            print(f"Processing year: {year}")
            year_dir = os.path.join(midi_dir, year)
            if os.path.exists(year_dir):
                self.files.extend([
                    os.path.join(year_dir, f) 
                    for f in os.listdir(year_dir) 
                    if f.endswith('.midi')
                ])
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.min_len = min_len
        
        # Pre-process files to ensure they're all valid
        print(f"Found {len(self.files)} MIDI files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            tokens = self.tokenizer.encode(path)
            
            # Ensure minimum length by padding if needed
            if len(tokens) < self.min_len:
                pad_len = self.min_len - len(tokens)
                tokens = tokens + [self.tokenizer.vocab["PAD"]] * pad_len
            
            # Random crop if sequence is too long
            if len(tokens) > self.max_len:
                start = np.random.randint(0, len(tokens) - self.max_len)
                tokens = tokens[start:start + self.max_len]
            
            # Pad if sequence is shorter than max_len
            if len(tokens) < self.max_len:
                pad_len = self.max_len - len(tokens)
                tokens = tokens + [self.tokenizer.vocab["PAD"]] * pad_len
            
            return torch.tensor(tokens)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a valid sequence of padding tokens
            return torch.full((self.max_len,), self.tokenizer.vocab["PAD"])

def get_dataloader(midi_dir, years, tokenizer, batch_size=8, num_workers=4, shuffle=True):
    print(f"Getting dataloader")
    dataset = MaestroMIDIDataset(midi_dir, years, tokenizer)
    
    print(f"Creating dataloader")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
