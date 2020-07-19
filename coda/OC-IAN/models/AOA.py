from models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.cnn import CNN
from models.attention import Attention

class AOANet(BaseModel):
    def __init__(self, filed = 80):
        super(AOANet, self).__init__()
        self.filed = filed
        self.cnn_l = CNN(filed=self.filed)
        self.rnn_l = nn.LSTM(
            input_size=40,
            hidden_size=64,
            num_layers=4,
            batch_first=True)


        self.linear = nn.Sequential(
                nn.Linear(64, 64),
                nn.Linear(64, 2),
        )


    def forward(self, x_l):
        x_l = x_l.view(-1, x_l.size(-2), x_l.size(-1))
        c_out_l = self.cnn_l(x_l)
        batch, _, cwidth =c_out_l.size()
        r_in_l = c_out_l #.view(-1, batch, cwidth)
        h_n_l, (_, _) = self.rnn_l(r_in_l)
        # print(h_n_l.shape)
        h_n_l_t = h_n_l[:,-1,:].view(h_n_l.size(0),-1,h_n_l.size(2))
        h_n_l_c = h_n_l[:,:-1,:]

        interaction_mat = torch.matmul(h_n_l_c, torch.transpose(h_n_l_t, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = f.softmax(interaction_mat, dim=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = f.softmax(interaction_mat, dim=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True) # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(h_n_l_c, 1, 2), gamma).squeeze(-1) # batch_size x 2*hidden_dim
        # print(weighted_sum.size())
        out = self.linear(weighted_sum)
        return f.log_softmax(out, dim=1)

