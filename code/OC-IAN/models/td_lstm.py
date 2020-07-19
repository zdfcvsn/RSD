from models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.cnn import CNN


class TD_LSTM(BaseModel):
    def __init__(self, filed = 80):
        super(TD_LSTM, self).__init__()
        self.filed = filed
        self.cnn_l = CNN(self.filed)
        self.rnn_l = nn.LSTM(
            input_size=40,
            hidden_size=64,
            num_layers=4,
            batch_first=True)

        self.cnn_r = CNN(self.filed)
        self.rnn_r = nn.LSTM(
            input_size=40,
            hidden_size=64,
            num_layers=4,
            batch_first=True)

        self.linear = nn.Sequential(
                nn.Linear(128, 64),
                nn.Linear(64, 2),
        )

    def forward(self, x_l, x_r):

        x_l = x_l.view(-1, x_l.size(-2), x_l.size(-1))
        c_out_l = self.cnn_l(x_l)
        batch, _, cwidth =c_out_l.size()
        r_in_l = c_out_l #.view(-1, batch, cwidth)
        _, (h_n_l, _) = self.rnn_l(r_in_l)

        x_r = x_r.view(-1, x_r.size(-2), x_r.size(-1))
        c_out_r = self.cnn_r(x_r)
        batch, _, cwidth =c_out_r.size()
        r_in_r = c_out_r #.view(-1, batch, cwidth)
        _, (h_n_r, _) = self.rnn_r(r_in_r)

        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.linear(h_n)
        return f.log_softmax(out, dim=1)



class TC_LSTM(BaseModel):
    def __init__(self, filed = 80):
        super(TC_LSTM, self).__init__()
        self.filed = filed
        self.cnn_l = CNN(filed = self.filed)
        self.cnn_r = CNN(filed = self.filed)
        self.rnn_l = nn.LSTM(
            input_size=80,
            hidden_size=64,
            num_layers=4,
            batch_first=True)


        self.rnn_r = nn.LSTM(
            input_size=80,
            hidden_size=64,
            num_layers=4,
            batch_first=True)

        self.linear = nn.Sequential(
                nn.Linear(128, 64),
                nn.Linear(64, 2),
        )

    def forward(self, x_l, x_r):

        x_l = x_l.view(-1, x_l.size(-2), x_l.size(-1))
        c_out_l = self.cnn_l(x_l)
        batch, cheight, cwidth =c_out_l.size()
        r_in_l = c_out_l.clone()
        for i in range(cheight):
            r_in_l[:,i,:] = c_out_l[:,-1,:]
        # print(r_in_l.shape)
        r_in_l = torch.cat((c_out_l,r_in_l), dim=2)
        # print(r_in_l.shape)
        _, (h_n_l, _) = self.rnn_l(r_in_l)

        x_r = x_r.view(-1, x_r.size(-2), x_r.size(-1))
        c_out_r = self.cnn_r(x_r)
        batch, cheight, cwidth =c_out_r.size()
        r_in_r = c_out_r.clone()
        for i in range(cheight):
            r_in_r[:,i,:] = c_out_r[:,-1,:]
        r_in_r = torch.cat((c_out_r,r_in_r), dim=2)
        _, (h_n_r, _) = self.rnn_r(r_in_r)

        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.linear(h_n)
        return f.log_softmax(out, dim=1)
