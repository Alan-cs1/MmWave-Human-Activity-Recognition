import torch.nn as nn

class IMUCNN_BiLSTM(nn.Module):
    def __init__(self, config):
        super(IMUCNN_BiLSTM, self).__init__()

        input_dim = config.get("input_dim")
        feature_dim = config.get("transformer_dim")
        window_size = config.get("window_size")
        lstm_hidden_dim = config.get("lstm_hidden_dim", feature_dim)  # Default to feature_dim if not provided
        num_layers = config.get("lstm_layers", 1)  # Default to 1 layer if not provided

        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, feature_dim, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feature_dim, feature_dim, kernel_size=1), nn.ReLU())

        self.dropout = nn.Dropout(config.get("baseline_dropout"))
        self.maxpool = nn.MaxPool1d(2)  # Collapse T time steps to T/2
        
        # Update BiLSTM input size to feature_dim (not divided)
        self.bilstm = nn.LSTM(feature_dim, lstm_hidden_dim, num_layers=num_layers, 
                              batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_dim * 2, feature_dim)  # *2 for bidirection
        self.fc2 = nn.Linear(feature_dim, config.get("num_classes"))
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        """
        Forward pass
        :param data: dict containing 'imu' tensor with shape B x M x T
        :return: B x N weights for each mode per sample
        """
        x = data.get('imu').transpose(1, 2)  # Shape: B x T x M
        x = self.conv1(x)                     # Shape: B x feature_dim x T
        x = self.conv2(x)                     # Shape: B x feature_dim x T
        x = self.dropout(x)
        x = self.maxpool(x)                   # Shape: B x feature_dim x (T/2)

        # Prepare for BiLSTM: Shape (B, T', feature_dim)
        x = x.transpose(1, 2)                 # Shape: B x (T/2) x feature_dim

        # BiLSTM expects input of shape (batch, seq_len, input_size)
        bilstm_out, _ = self.bilstm(x)        # Shape: B x (T/2) x (lstm_hidden_dim * 2)
        
        # Use the last output of BiLSTM for classification
        x = bilstm_out[:, -1, :]               # Take the last time step's output: Shape: B x (lstm_hidden_dim * 2)

        x = self.fc1(x)                        # Shape: B x feature_dim
        x = self.log_softmax(self.fc2(x))      # Shape: B x num_classes
        return x  # B x N
