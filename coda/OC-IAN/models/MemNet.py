from models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.cnn import CNN
from models.attention import Attention

class MemNet(BaseModel):
    def __init__(self, filed = 80):
        super(MemNet, self).__init__()
        self.filed = filed
        self.cnn_l = CNN(filed=self.filed)

        self.attention = Attention(40, score_function='mlp')

        self.x_linear = nn.Sequential(
                nn.Linear(40, 40),
        )
        self.linear = nn.Sequential(
                nn.Linear(40, 64),
                nn.Linear(64, 2),
        )


    def forward(self, x_l):
        x_l = x_l.view(-1, x_l.size(-2), x_l.size(-1))
        c_out_l = self.cnn_l(x_l)
        c_out_l_t = c_out_l[:,-1,:].view(c_out_l.size(0),-1,c_out_l.size(2))
        c_out_l_c = c_out_l[:,:-1,:]

        for _ in range(self.filed):
            x_linear = self.x_linear(c_out_l_t)
            out_at, _ = self.attention(c_out_l_c, c_out_l_t)
            c_out_l_t = out_at + x_linear

        c_out_l_t = c_out_l_t.view(c_out_l_t.size(0), -1)
        out = self.linear(c_out_l_t)
        return f.log_softmax(out, dim=1)

