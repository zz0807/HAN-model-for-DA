import pandas as pd
from keras.preprocessing import text, sequence
import numpy as np

class BuildDateset:

    def __init__(self):
        train_data = pd.read_csv("maptask_train.csv")
        self.tag_dict = self.__build_tag_dict(train_data['label'])
        self.text_tokenizer = text.Tokenizer(oov_token='unk')
        self.text_tokenizer.fit_on_texts(train_data['utterances'])

    def data_x2seq(self, data_x):
        des_data = list()
        for para in data_x:
            para = sequence.pad_sequences(para, maxlen=110, padding="post")
            zero_a = np.zeros((700 - para.shape[0], 110))
            para = np.concatenate((para, zero_a), axis=0)
            des_data.append(para.tolist())
        return des_data

    def get_train(self):
        data = self.__data_load("maptask_train.csv")
        return data

    def get_val(self):
        data = self.__data_load("maptask_val.csv")
        return data

    def get_test(self):
        data = self.__data_load("maptask_test.csv")
        return data

    def __build_tag_dict(self, tag_data):
        tag_dict = dict()
        for i in tag_data:
            if i not in tag_dict:
                tag_dict[i] = len(tag_dict)
        return tag_dict

    def __data_load(self, filename):
        data = pd.read_csv(filename, encoding='utf-8')
        # data[['id','utterances']]
        ids = data['id']
        x_data = self.text_tokenizer.texts_to_sequences(list(data['utterances']))
        a_conversation_x_list = list()
        a_conversation_y_list = list()
        conversation_x_list = list()
        conversation_y_list = list()
        cur_id = ids[0]
        for id, x, y in zip(ids, x_data, data['label']):
            if cur_id == id:
                a_conversation_x_list.append(x)
                a_conversation_y_list.append(self.tag_dict[y])
            else:
                cur_id = id
                a_conversation_x_list = list()
                a_conversation_y_list = list()
                a_conversation_x_list.append(x)
                a_conversation_y_list.append(self.tag_dict[y])
                conversation_x_list.append(a_conversation_x_list)
                conversation_y_list.append(a_conversation_y_list)
        return conversation_x_list, conversation_y_list


