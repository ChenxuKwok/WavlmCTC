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

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=lexicon.index('<unk>'), zero_infinity=True)  # CTC loss with blank index
    early_stop = False
    for epoch in range(epochs):
        # empty cache
        torch.cuda.empty_cache()
        for i, (audio, labels) in enumerate(train_loader):
            audio = audio.to(device)
            labels = labels.to(device)  # Directly use the padded tensor
            
            # Extract features
            embeddings = model(audio)
            last_hidden_state = embeddings # Assume output is (B, T, 768)
            
            # CTC requires the output to be (B, T, num_classes)
            # output = last_hidden_state.transpose(0, 1)  # (T, B, 768) --> (T, B, num_classes)
            output = torch.log_softmax(last_hidden_state.transpose(0, 1), dim=2)
            # print(output.shape)  # Should be (T, B, len(lexicon))
            
            
            # Compute CTC loss
            input_lengths = torch.full(size=(audio.size(0),), fill_value=last_hidden_state.size(1), dtype=torch.long)
            target_lengths = torch.sum(labels != -1, dim=1)  # Exclude padding (-1) when computing target length
            
            loss = criterion(output, labels, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}")
                loss_values.append(loss.item())
            
            # avg loss
            if i % 100 == 0:
                avg_loss = sum(loss_values[-100:]) / 100
                print(f"Average loss: {avg_loss}")
                if np.abs(avg_loss) < 1e-4:
                    early_stop = True
                    break
            
            current_loss = loss.item()
            if current_loss < early_stop_threshold:
                early_stop = True
                break
            
        if early_stop:
            break
            

    torch.save(model, "wavlm_ctc_model_linear.pth")
    with open("loss_values.txt", "w") as f:
        for loss_value in loss_values:
            f.write(f"{loss_value}\n")

# load parameters of training to yaml file
import yaml
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    lexicon = config["lexicon"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

# Load the model
# model = wavlm_base()
model = PhonemeRecognitionModel(wavlm_base(), len(lexicon))
train_dataset = PhonemeDataset(json_file='Data/train/train_meta_data.json')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# Train the model
train(model, train_loader, device='cuda:0', epochs=epochs, lr=lr)