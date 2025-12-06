import torch
import torch.nn as nn
import numpy as np
import os
from typing import Tuple, List, Optional

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, latent_dim=32):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (h_n, _) = self.encoder(x)
        # h_n shape: (1, batch_size, hidden_dim)
        
        # Latent space
        latent = self.latent(h_n[-1]) # (batch_size, latent_dim)
        
        # Decode
        decoder_in = self.decoder_input(latent) # (batch_size, hidden_dim)
        decoder_in = decoder_in.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_dim)
        
        decoded, _ = self.decoder(decoder_in)
        reconstructed = self.output(decoded) # (batch_size, seq_len, input_dim)
        
        return reconstructed

class LSTMPredictor:
    def __init__(self, model_path: Optional[str] = None, input_dim=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 ML Engine initializing on {self.device}")
        
        self.model = LSTMAutoencoder(input_dim=input_dim).to(self.device)
        self.criterion = nn.MSELoss(reduction='none')
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded ML model from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
        else:
            print("🆕 Initialized new LSTM model (untrained)")
            
        self.model.eval()

    def predict(self, features: np.ndarray) -> float:
        """
        Predict anomaly score (reconstruction error)
        features: (seq_len, input_dim)
        """
        if len(features.shape) == 1:
            # Handle single vector by repeating it to form a sequence
            # This is a simplification for single-packet inference
            features = np.tile(features, (10, 1))
            
        # Ensure input is (batch, seq, dim)
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(x)
            
            # Calculate MSE loss per time step, then mean over sequence
            loss = self.criterion(reconstructed, x)
            mse = loss.mean().item()
            
            return mse

    def train(self, features_list: List[np.ndarray], epochs=1):
        """Simple online training simulation"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Convert list of arrays to tensor batch
        # Assuming all arrays have same shape (seq_len, input_dim)
        # If not, we'd need padding, but for now assume consistent input
        try:
            batch = np.array(features_list)
            x = torch.FloatTensor(batch).to(self.device)
            
            for _ in range(epochs):
                optimizer.zero_grad()
                recon = self.model(x)
                loss = self.criterion(recon, x).mean()
                loss.backward()
                optimizer.step()
                
            self.model.eval()
            return loss.item()
        except Exception as e:
            print(f"⚠️ Training error: {e}")
            return 0.0
