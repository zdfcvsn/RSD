from models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.cnn import CNN
from models.attention import NoQueryAttention

class ATAE_LSTM(BaseModel):
    def __init__(self, filed = 80):
        super(ATAE_LSTM, self).__init__()
        self.filed = filed
        self.cnn_l = CNN(filed=self.filed)
        self.rnn_l = nn.LSTM(
            input_size=80,
            hidden_size=64,
            num_layers=4,
            batch_first=True)

        self.attention = NoQueryAttention(128, score_function='bi_linear')

        self.linear = nn.Sequential(
                nn.Linear(64, 64),
                nn.Linear(64, 2),
        )


    def forward(self, x_l):

        x_l = x_l.view(-1, x_l.size(-2), x_l.size(-1))
        c_out_l = self.cnn_l(x_l)
        batch, cheight, cwidth =c_out_l.size()
        r_in_l = c_out_l.clone()
        for i in range(cheight):
            r_in_l[:,i,:] = c_out_l[:,-1,:]
        # print(r_in_l.shape)
        r_in_l = torch.cat((c_out_l,r_in_l), dim=2)
        h_n_l, (_, _) = self.rnn_l(r_in_l)
        batch, cheight, cwidth = h_n_l.size()
        h_n = h_n_l.clone()
        for i in range(cheight):
            h_n[:,i,:] = h_n_l[:,-1,:]
        h_n = torch.cat((h_n_l, h_n), dim=2)
        # print(h_n.shape)
        _, score = self.attention(h_n)
        # print(score.shape)
        output = torch.squeeze(torch.bmm(score, h_n_l), dim=1)
        # print(output.shape)
        out = self.linear(output)
        return f.log_softmax(out, dim=1)

