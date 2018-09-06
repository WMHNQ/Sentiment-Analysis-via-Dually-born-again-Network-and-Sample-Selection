import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    # Split by words
    x_text = positive_examples
    x_text = [clean_str(sent) for sent in x_text]
    #move lable
    x_text = [(sent[2:]) for sent in x_text]
    # Generate labels
    lable=[]
    for i in positive_examples:
        if i[0:1]=='0':
            lable.append([1,0,0,0,0])
        if i[0:1]=='1':
            lable.append([0,1,0,0,0])
        if i[0:1]=='2':
            lable.append([0,0,1,0,0])
        if i[0:1]=='3':
            lable.append([0,0,0,1,0])
        if i[0:1]=='4':
            lable.append([0,0,0,0,1])

    y = np.array(lable)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print ('a bad word embedding: {}'.format(line[0]))
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print (np.shape(w2v))
    word_dict['UNK'] = cnt + 1
    print(word_dict['UNK'], len(w2v))
    return word_dict, w2v


def word2id(input_file, word_id_file, sentence_len, encoding='utf8'):
    word_to_id = word_id_file
    print ('load word-to-id done!')
    sen_id,  sen_len = [], []
    for i in input_file:
        words = i.split(" ")
        sen = len(words)
        sen_len.append(sen)
        words_id = []
        for word in words:
            try:
                words_id.append(word_to_id[word])
            except:
                words_id.append(word_to_id['UNK'])
        sen_id.append(words_id + [0] * (sentence_len - len(words)))

    return np.asarray(sen_id), np.asarray(sen_len)

