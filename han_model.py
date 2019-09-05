import torch
from torch import nn
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable
import numpy as np


class sentenceLevelRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_num):
        super(sentenceLevelRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight.requires_grad = True
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=embed_size,
            hidden_size=hidden_num,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )


    def forward(self, x, sentence_lens):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x_embedding = self.embed(x)
        embed_input_x_packed = nn.utils.rnn.pack_padded_sequence(x_embedding, sentence_lens, batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, (h_n, h_c) = self.rnn(embed_input_x_packed, None)   # None represents zero initial hidden state
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs_packed, batch_first=True)
        # choose r_out at the last time step

        return encoder_outputs, x_embedding


class HAN(nn.Module):

    def __init__(self, vocab_size, embed_size, sen_hidden_num, con_hidden_num, class_num, embed_y_size):
        super(HAN, self).__init__()
        self.senRNN = sentenceLevelRNN(vocab_size, embed_size, sen_hidden_num)
        # self.conRNN = nn.LSTM(         # if use nn.RNN(), it hardly learns
        #     input_size=hidden_num,
        #     hidden_size=hidden_num,         # rnn hidden unit
        #     num_layers=1,           # number of rnn layer
        #     batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        # )
        self.conRNN = nn.LSTMCell(sen_hidden_num, con_hidden_num)
        self.hx = torch.randn(1, con_hidden_num)
        self.cx = torch.randn(1, con_hidden_num)
        self.out = nn.Linear(con_hidden_num, class_num)
        # attention parameter
        self.w_in = nn.Linear(embed_size, embed_y_size)
        self.w_co = nn.Linear(con_hidden_num, embed_y_size)
        self.b = Variable(torch.FloatTensor(np.zeros(embed_y_size)), requires_grad=True)
        self.u = nn.Linear(embed_y_size, 1)
        self.embed_y = nn.Embedding(class_num, embed_y_size)

    def forward(self, x, sentence_lens):
        sen_vec, x_embedding = self.senRNN(x, sentence_lens)
        # sen_vec_view = sen_vec.unsqueeze(0)
        # r_out, (h_n, h_c) = self.conRNN(sen_vec_view)
        r_out = list()
        for i in range(x_embedding.shape[0]):
            if i >= 1:
                att = self.__attention_layer(x_embedding[i, :, :], hx, eyt_1).double().view(1, -1)
            else:
                att = torch.from_numpy(np.full((1, sen_vec.shape[1]), 1.0/sen_vec.shape[1], "double"))
            sen_vec_1 = torch.mm(att, sen_vec[i, :, :].double())
            hx, cx = self.conRNN(sen_vec_1.float(), (self.hx, self.cx))
            out = F.log_softmax(self.out(hx), dim=1)
            pre = torch.max(out, 1)[1].data.numpy()[0]
            eyt_1 = self.embed_y(torch.LongTensor([pre]))
            r_out.append(hx)
        r_out_cat = r_out[0]
        for i in range(1, len(r_out)):
            r_out_cat = torch.cat((r_out_cat, r_out[i]), 0)
        out = self.out(r_out_cat)
        tag_score = F.log_softmax(out, dim=1)
        return tag_score

    def __attention_layer(self, xt, gt_1, eyt_1):
        t1 = self.w_in(xt)
        t2 = self.w_co(gt_1)
        t2_total = t2
        for i in range(t1.shape[0]-1):
            t2_total = torch.cat((t2_total, t2))
        t2_total = t2_total.view(t1.shape[0], -1)

        eyt_1_total = eyt_1
        for i in range(t1.shape[0]-1):
            eyt_1_total = torch.cat((eyt_1_total, eyt_1))
        eyt_1_total = eyt_1_total.view(t1.shape[0], -1)

        b_total = self.b.data
        for i in range(t1.shape[0] - 1):
            b_total = torch.cat((b_total, self.b.data))
        b_total = b_total.view(t1.shape[0], -1)

        att = t1 + t2_total + eyt_1_total + b_total
        st = torch.tanh(self.u(att))
        m = nn.Softmax(dim=0)
        return m(st)



