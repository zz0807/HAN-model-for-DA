from preprocess import BuildDateset
from util import compute_sen_len_in_con
from keras.preprocessing import text, sequence
from han_model import HAN
import torch
from torch import nn
import numpy


def get_train_max_sen(train_x):
    max_len = 0
    for con in train_x:
        for sen in con:
            if len(sen) > max_len:
                max_len = len(sen)
    return max_len


def get_train_max_con_len(train_y):
    max_len = 0
    for con in train_y:
        if len(con) > max_len:
            max_len = len(con)
    return max_len


def get_dataset_acc(model, x, y):
    total_pre = numpy.array([])
    total_y = numpy.array([])
    for x1, y1 in zip(x, y):
        x1_sen_len = compute_sen_len_in_con(x1)
        x1 = sequence.pad_sequences(x1, maxlen=None, dtype='int32', padding='post', value=0)
        tag_scores = model(torch.from_numpy(x1).long(), x1_sen_len)
        y1 = torch.from_numpy(numpy.array(y1)).long()
        pred_y = torch.max(tag_scores, 1)[1].data.numpy()
        y1 = y1.data.numpy()
        if total_y.size == 0 and total_pre.size == 0:
            total_pre = pred_y
            total_y = y1
        else:
            total_pre = numpy.append(total_pre, pred_y)
            total_y = numpy.append(total_y, y1)
    accuracy = float((total_pre == total_y).astype(int).sum()) / float(total_y.size)
    return accuracy

dataset = BuildDateset()
vocab_size = len(dataset.text_tokenizer.word_index) + 1
class_num = len(dataset.tag_dict)
embed_size = 250
sen_hidden_num = 160
con_hidden_num = 250
lr = 0.01
embed_y_size = 180
train = dataset.get_train()
dev = dataset.get_val()
test = dataset.get_test()
train_x = train[0]
train_y = train[1]
dev_x = dev[0]
dev_y = dev[1]
test_x = test[0]
test_y = test[1]

max_con_len = get_train_max_con_len(train_y)
max_sen_len = get_train_max_sen(train_x)
model = HAN(vocab_size, embed_size, sen_hidden_num, con_hidden_num, class_num, embed_y_size)
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # optimize all cnn parameters

for epoch in range(100):
    for conv1_x, conv1_y in zip(train_x, train_y):
        conv1_sen_len = compute_sen_len_in_con(conv1_x)
        conv1_x = sequence.pad_sequences(conv1_x, maxlen=None, dtype='int32', padding='post', value=0)
        tag_scores = model(torch.from_numpy(conv1_x).long(), conv1_sen_len)
        conv1_y = torch.from_numpy(numpy.array(conv1_y)).long()
        loss = loss_function(tag_scores, conv1_y)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        pred_y = torch.max(tag_scores, 1)[1].data.numpy()
        conv1_y = conv1_y.data.numpy()
        accuracy = float((pred_y == conv1_y).astype(int).sum()) / float(conv1_y.size)
        dev_acc = get_dataset_acc(model, dev_x, dev_y)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| conversation train accuracy: %.2f' % accuracy, '| dev accuracy: %.2f' % dev_acc)
    test_acc = get_dataset_acc(model, test_x, test_y)
    print('test acc: ', test_acc)

