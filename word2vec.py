#from gensim.models.keyedvectors import KeyedVectors
#
# model = KeyedVectors.load_word2vec_format('wordembedding/rob04.wv.cbow.d300.w10.n10.m10.i10.W.bin', binary=True)
#
# # 转换成 txt 格式
# model.save_word2vec_format('wordembedding/rob04.wv.cbow.d300.w10.n10.m10.i10.W.txt', binary=False)
#
# # 另存为 bin 文件
# model.save_word2vec_format('wordembedding/rob04.wv.cbow.d300.w10.n10.m10.i10.W22.bin', binary=True)

from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
import string
import os
from gensim.models import Word2Vec


def remove_punc(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def get_embeddings(data_type, qa_file, doc_file):
    corpus = []

    with open(qa_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line != "" and i % 2 == 0:
                line = remove_punc(line)
                words = word_tokenize(line)
                corpus.append(words)

    with open(doc_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line != "":
                line = remove_punc(line)
                words = word_tokenize(line)
                corpus.append(words)


    data_type = data_type.lower()
    min_count = 1
    size = 200
    window = 10
    negative = 10
    sg = 0

    w2v_model = Word2Vec(corpus,
                         min_count=min_count,
                         size=size,
                         window=window,
                         negative=negative,
                         sg = sg)
    to_file = "models/%s.wv.cbow.d%d.w%d.n%d.bin" % (data_type, size, window,negative)
    w2v_model.wv.save_word2vec_format(to_file, binary=True)
    print("saved to:", to_file)


if __name__ == '__main__':
    data_type = "adwords"
    doc_file = data_type + "/Doc_list.txt"
    qa_file = data_type + "/QA_list.txt"
    get_embeddings(data_type, qa_file, doc_file)



