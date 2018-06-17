# -*- coding: utf-8 -*-
import re
import yaml
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
from sklearn.cross_validation import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import csv
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml

np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd
import pickle

sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 15
input_length = 100
cpu_count = multiprocessing.cpu_count()


# 加载训练文件
def loadfile():
    # neg=pd.read_excel('../data/neg.xls',header=None,index=None)
    # pos=pd.read_excel('../data/pos.xls',header=None,index=None)

    sens = pd.read_excel('../data/敏感数据1.xlsx', header=None, index=None)
    notsens = pd.read_excel('../data/不敏感数据1.xlsx', header=None, index=None)
    combined = np.concatenate((notsens[0], sens[0]))
    y = np.concatenate((np.ones(len(notsens), dtype=int), np.zeros(len(sens), dtype=int)))
    # y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))
    return combined, y


# 对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')

# 创建词语字典，并返回word2vec模型中词语的索引，词向量

def create_dictionaries1(p_model, combined):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: p_model[word] for word in w2indx.keys()}  # 词语的词向量

    def parse_dataset(combined):
        data = []
        for sentence in combined:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            data.append(new_txt)
        return data

    combined = parse_dataset(combined)
    combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
    return w2indx, w2vec, combined

# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    '''
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations,total_examples=model.corpus_count, epochs=model.iter)


    model.build_vocab(combined)
    model.train(combined)
    '''
    model = Word2Vec(combined, size=100, min_count=5, window=5)  # 窗口大小
    # model.build_vocab(combined)

    model.save('../new_data_lstm4/Word2vec_model.pkl')
    # print(model.wv['物理学'])
    # index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    index_dict, word_vectors, com = create_dictionaries1(model, combined)

    # 存储为pkl文件
    # pkl_name = raw_input(u"请输入保存的pkl文件名...\n").decode("utf-8")
    output = open('../new_data_lstm4/wordvec.pkl', 'wb')
    pickle.dump(index_dict, output)  # 索引字典
    pickle.dump(word_vectors, output)  # 词向量字典
    output.close()
    print('OK')
    return index_dict, word_vectors, combined


def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)

    return np.array(new_sentences)


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)

    # 读取大语料文本
    f = open('../new_data_lstm4/wordvec.pkl', 'rb')  # 预先训练好的
    index_dict = pickle.load(f)  # 索引字典，{单词: 索引数字}
    word_vectors = pickle.load(f)  # 词向量, {单词: 词向量(100维长的数组)}

    new_dic = index_dict
    n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
    embedding_weights = np.zeros((n_symbols, 100))  # 创建一个n_symbols * 100的0矩阵
    for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
    wx_train = text_to_index_array(new_dic, x_train)
    wx_test = text_to_index_array(new_dic, x_test)
    wy_train = np.array(wx_train)  # 转numpy数组
    wy_test = np.array(wx_test)
    wx_train = sequence.pad_sequences(wx_train, maxlen=maxlen)
    wx_test = sequence.pad_sequences(wx_test, maxlen=maxlen)
    # print(x_train)
    return n_symbols, embedding_weights, wx_train, y_train, wx_test, y_test


##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch)
    # with open('../lstm1_data/lstm.yml', 'r') as f:
    #     yaml_string = yaml.load(f)
    # model = model_from_yaml(yaml_string)
    # model.load_weights('../lstm1_data/lstm.h5')
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    score, acc = model.evaluate(x_train, y_train, batch_size=batch_size)
    print('Train score:', score)
    print('Train accuracy:', acc)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    '''
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test),show_accuracy=True)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    '''
    yaml_string = model.to_yaml()
    with open('../new_data_lstm4/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('../new_data_lstm4/lstm.h5')
    print('Test score:', score)


# 训练模型，并保存
def train():
    print('Loading Data...')
    combined, y = loadfile()
    print(len(combined), len(y))
    print('loadfile此处的combind', combined)
    print('因变量', y)
    print('Tokenising...')
    combined = tokenizer(combined)
    print('经过tokenizer的combind', combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def get_word2vec():
    f = open('../new_data_lstm4/wordvec.pkl', 'rb')  # 预先训练好的
    index_dict = pickle.load(f)  # 索引字典，{单词: 索引数字}
    word_vectors = pickle.load(f)  # 词向量, {单词: 词向量(100维长的数组)}
    return index_dict, word_vectors


def input_transform(string):
    # words=jieba.lcut(string)
    words = jieba.lcut(string)
    print(words)
    print(np.array(string))
    words = np.array(words).reshape(1, -1)
    index_dic, word_vectors = get_word2vec()
    test = text_to_index_array(index_dic, words)
    test = np.array(test)
    test = sequence.pad_sequences(test, maxlen=maxlen)
    print(test)
    return test

def lstm_predict(string):
    print('loading model......')

    with open('../new_data_lstm4/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../new_data_lstm4/lstm.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # while True:
    # string=input('输入一句话:')
    string = re_str(string)
    data = input_transform(string)
    result = model.predict_classes(data)
    if result[0][0] == 1:
        print(string, ' 不敏感')
        result = '不敏感'
    else:
        print(string, ' 敏感')
        result = '敏感'
    yield string, result


# def main_pre_juzi():


def again():
    print('Loading Data...')
    combined, y = loadfile()
    print(len(combined), len(y))
    print('loadfile此处的combind', combined)
    print('因变量', y)
    print('Tokenising...')
    combined = tokenizer(combined)
    print('经过tokenizer的combind', combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print('loading model......')
    with open('../new_data_lstm4/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../new_data_lstm4/lstm.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    loss1, acc1 = model.evaluate(x_train, y_train, batch_size=batch_size)

    print('loss%f,acc%f' % (loss, acc))
    print('loss1%f,acc1%f' % (loss1, acc1))


def sentencs():
    try:
        datas = pd.read_excel('E:\我的项目\舆情项目\Sentiment\data\sbumingan.xlsx', header=None)
        for i in datas[0]:
            try:
                yield i
            except:
                pass
    except:
        with open('E:\我的项目\舆情项目\Sentiment\data\sbumingan.xlsx', 'r') as f:
            reader = csv.reader(f)
            for i in reader:
                try:
                    yield i[0]
                except:
                    pass


def lstm_predict1(string=None):
    print('loading model......')

    with open('../lstm3_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../lstm3_data/lstm.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    while True:
        string = input('输入一句话:')
        string = re_str(string)
        data = input_transform(string)
        result = model.predict_classes(data)
        if result[0][0] == 1:
            print(string, ' 不敏感')
            result = '不敏感'
        else:
            print(string, ' 敏感')
            result = '敏感'
        yield string, result


def lstm_predict2(string=None):
    print('loading model......')

    with open('../new_data_lstm4/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../new_data_lstm4/lstm.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    my_num = 0
    my_sum = 0
    for string in sentencs():
        my_sum += 1
        try:
            data = input_transform(string)
            result = model.predict_classes(data)
            if result[0][0] == 1:
                print('不敏感')
                print(string, ' 不敏感')
                my_sentencs = string
                result = '不敏感'

            else:
                my_num += 1
                print('敏感')
                print(string, ' 敏感')
                my_sentencs = string
                result = '敏感'
        except:
            pass
    acc = my_num / my_sum
    print('准确度是%f' % acc)


def re_str(x):
    my_list = re.findall(r'[\u4e00-\u9fa5，,。]', x)
    my_sente = ''
    for i in my_list:
        my_sente += i
    return my_sente


if __name__ == '__main__':
    again()
    # train()
    # x,y=get_word2vec()
    # print(x,y)
    # train()
    # train()
    # string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    # string='酒店的环境非常好，价格也便宜，值得推荐'
    # string='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    # string='我是傻逼'
    # string='你是傻逼'
    # string='屏幕较差，拍照也很粗糙。'
    # string='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    # string='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
    #     string='''现有成都市公安局成华区分局彭学军，身份证号码：51010319710317**** ，身为人民警察不但不为民办事，知法犯法，态度恶劣，利用职务之便到处借钱，无法无天，在中国共产党领导下的公职人员处处与人民唱反调，视党纪国法为无物，恶意欺骗，严重损害人民警察的形象，其行为严重损害人民群众合法权益，我们曾多次致电其单位，让单位督促并监督其自身行为，单位以鲜洋林为首拒不配合并涉嫌包庇此人，让群众合法权益无法得以维护，让党和政府的蛀虫逍遥法外，敢问群众利益何在？
    # '''
    #     string='''"乘坐神州专车的奇葩经历
    # 4月11日凌晨，从双流机场T2回家，车牌川AYK800的颜师傅接单后，打电话告知我需要我提供真实手机号码，我纳闷，未给！继而拨打神州专车客服电话问询是否这样做合理，客服说，已经跟实际取得联系，这样做是因为司机手机没有电了，我感觉此事蹊跷，所以上车后，从头到尾摄制视频，结果真的乱收费现象出现。"
    # '''
    #     string='''蛇溜进宿舍被学生剥皮炖汤食用 校方:已教育处理
    # '''
    # lstm_predict2()
    # lstm_predict()
    # lstm_predict1()