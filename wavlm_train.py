from s3prl.hub import wavlm_base
import torch
import torch.nn as nn
import json
import torchaudio
from torch.utils.data import Dataset, DataLoader
import loralib as lora
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
import sys


class PhonemeRecognitionModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(PhonemeRecognitionModel, self).__init__()
        self.base_model = base_model
        self.output_layer = nn.Linear(768, num_classes)  # 768 -> len(lexicon)

    def forward(self, x):
        embeddings = self.base_model(x)
        last_hidden_state = embeddings['last_hidden_state']
        
        output = self.output_layer(last_hidden_state)
        
        return output
    
class PhonemeDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav_file = self.data[idx]["wav_file"]
        transcription = self.data[idx]["transcription"]
        audio, freq = torchaudio.load(wav_file)
        audio = torchaudio.transforms.Resample(orig_freq=freq, new_freq=16000)(audio)
        
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)  # Merge channels
        audio = audio.squeeze(0)  # Shape becomes (T, )
        
        # Process labels
        labels = transcription.split()
        label_indices = [lexicon.index(label) if label in lexicon else lexicon.index('<unk>') for label in labels]
        return audio, torch.tensor(label_indices)

def collate_fn(batch):
    audios, labels = zip(*batch)
    
    # Pad audio to the same length (shape is (B, max_T))
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0)
    
    # Pad labels (using index tensors returned by the dataset)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    
    return audios_padded, labels_padded

# ctc
def train(model, train_loader, device, epochs=10, lr=1e-4, early_stop_threshold=0.05):
    model.train()
    model.to(device)
    loss_values = []

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,    
        patience=2,      
        verbose=True,    
        min_lr=1e-6      
    )
    
    criterion = nn.CTCLoss(blank=lexicon.index('<unk>'), zero_infinity=True)
    
    max_grad_norm = 5.0
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_samples = 0
        early_stop = False
        
        for i, (audio, labels) in enumerate(train_loader):
            audio = audio.to(device)
            labels = labels.to(device)
            
            embeddings = model(audio)
            last_hidden_state = embeddings
            output = torch.log_softmax(last_hidden_state.transpose(0, 1), dim=2)
            
            input_lengths = torch.full((audio.size(0),), last_hidden_state.size(1), dtype=torch.long, device=device)
            target_lengths = torch.sum(labels != -1, dim=1)
            loss = criterion(output, labels, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss * audio.size(0)
            total_samples += audio.size(0)
            
            if i % 10 == 0:
                avg_loss = epoch_loss / total_samples
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {current_loss:.4f}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
            
            if current_loss < early_stop_threshold:
                early_stop = True
                break
        
        avg_epoch_loss = epoch_loss / total_samples
        loss_values.append(avg_epoch_loss)
        
        scheduler.step(avg_epoch_loss)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with loss {best_loss:.4f}")
        
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    torch.save(model.state_dict(), "wavlmctc-pr.pth")
    print("Training completed. Final model saved.")

# load parameters of training to yaml file
import yaml
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    lexicon = config["lexicon"]
    lr = float(config["lr"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    
print("Initial learning rate: ", lr)
# Load the model
# model = wavlm_base()
model = PhonemeRecognitionModel(wavlm_base(), len(lexicon))
train_dataset = PhonemeDataset(json_file='Data/train/train_meta_data.json')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# Train the model
train(model, train_loader, device='cuda:0', epochs=epochs, lr=lr)