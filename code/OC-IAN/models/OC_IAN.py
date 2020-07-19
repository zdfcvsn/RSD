from models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.cnn import CNN
from models.attention import Attention

class OC_IAN(BaseModel):
    def __init__(self, filed = 80):
        super(OC_IAN, self).__init__()
        self.filed = filed
        self.cnn_l = CNN(filed=self.filed)
        self.rnn_l = nn.LSTM(
            input_size=40,
            hidden_size=64,
            num_layers=4,
            batch_first=True)

        self.attention_aspect = Attention(64, score_function='bi_linear')
        self.attention_context = Attention(64, score_function='bi_linear')


        self.linear = nn.Sequential(
                nn.Linear(128, 64),
                nn.Linear(64, 2),
        )


    def forward(self, x_l):

        x_l = x_l.view(-1, x_l.size(-2), x_l.size(-1))
        c_out_l = self.cnn_l(x_l)
        r_in_l = c_out_l
        h_n_l, (_, _) = self.rnn_l(r_in_l)
        batch, cheight, cwidth = h_n_l.size()
        h_n_l_t = h_n_l[:,-1,:].view(h_n_l.size(0),-1,h_n_l.size(2))
        h_n_l_c = h_n_l[:,:-1,:]


        pool_t = h_n_l_t
        pool_c = torch.mean(h_n_l_c, dim=1)

        t_final, _ = self.attention_aspect(h_n_l_t, pool_c)
        t_final = t_final.squeeze(dim=1)
        c_final, _ = self.attention_context(h_n_l_c, pool_t)
        c_final = c_final.squeeze(dim=1)

        x = torch.cat((t_final, c_final), dim=-1)
        # print(x.size())
        out = self.linear(x)
        return f.log_softmax(out, dim=1)

