import re
import yaml
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
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
from keras.layers.core import Dense, Dropout,Activation
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

def loadfile():
    # neg=pd.read_excel('../data/neg.xls',header=None,index=None)
    # pos=pd.read_excel('../data/pos.xls',header=None,index=None)

    sens = pd.read_excel('../data/smingan.xlsx', header=None, index=None)

    notsens = pd.read_excel('../data/sbumingan.xlsx', header=None, index=None)

    combined = np.concatenate((notsens[0], sens[0]))
    y = np.concatenate((np.ones(len(notsens), dtype=int), np.zeros(len(sens), dtype=int)))
    # y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))
    return combined, y

def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

if __name__=='__main__':
    comlobind,y=loadfile()
    comlobind=tokenizer(comlobind)
    for i in comlobind:
        print(i)

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

        model.save('../lstm3_data/Word2vec_model.pkl')
        # print(model.wv['物理学'])
        # index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
        index_dict, word_vectors, com = create_dictionaries1(model, combined)

        # 存储为pkl文件
        # pkl_name = raw_input(u"请输入保存的pkl文件名...\n").decode("utf-8")
        output = open('../lstm3_data/wordvec.pkl', 'wb')
        pickle.dump(index_dict, output)  # 索引字典
        pickle.dump(word_vectors, output)  # 词向量字典
        output.close()
        print('OK')
        return index_dict, word_vectors, combined

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

