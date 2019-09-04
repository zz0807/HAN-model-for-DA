# 这种形式的数据
# model.target_y: [
#     [1, 1, 0],
#     [2, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
# ]


def compute_acc(info, pre_data, true_data):
    total = 0
    correct = 0
    for paragraph_pre_data, paragraph_true_data in zip(pre_data, true_data):
        for utter_pre_label, utter_true_label in zip(paragraph_pre_data, paragraph_true_data):
            if utter_true_label != 0:
                total += 1
                if utter_pre_label == utter_true_label:
                    correct += 1
    acc = (float(correct)/total)*100
    print(info + str(acc) + "%")
    return acc

# 计算句子最大长度:句子里面的单词个数
def compute_max_sentence(data_x):
    max = len(data_x[0][0])
    for par in data_x:
        for sen in par:
            if len(sen) > max:
                max = len(sen)
    return max


# 计算每段对话里最多有多少个句子
def compute_max_para(data_y):
    max = len(data_y[0])
    for par in data_y:
        if len(par) > max:
            max = len(par)
    return max


# 计算对话中的每个句子的长度
def compute_sen_len_in_con(con):
    sens_len_list = list()
    for sen in con:
        sens_len_list.append(len(sen))
    return sens_len_list





