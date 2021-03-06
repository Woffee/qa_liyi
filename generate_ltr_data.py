"""


@Time    : 10/30/20
@Author  : Wenbo


        dnn_model = load_model(dnn_model_path)
        print("loading dnn model:", dnn_model_path)
        new_dnn_model = Model(inputs=dnn_model.input, outputs=dnn_model.get_layer('hidden_layer').output)
        res = new_dnn_model.predict([doc_train, que_train, fea_train])

"""
import argparse
import os
import numpy as np

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import adam
from keras.layers.recurrent import GRU
from keras.layers.core import Lambda
from keras.layers import Dot, add, Bidirectional, Dropout, Reshape
from keras.models import Input, Model, load_model
from keras import backend as K
from adding_weight import adding_weight

import keras

import time
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
now_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=BASE_DIR + '/' + now_time + '.log')
logger = logging.getLogger(__name__)

from word2vec import remove_punc
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
nltk.download('punkt')
from Model import loss_c, get_randoms, sentence2vec, negative_samples
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

import random
random.seed(9)

def get_train_data(data_type, w2v_model,  qa_file, doc_file, to_file_path, args):
    logger.info("preprocessing...")
    ns_amount = args.ns_amount

    questions = []
    answers = []

    # 计算每个question的向量
    input_length = 0
    with open(qa_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line != "" and i % 2 == 0:
                words = word_tokenize(remove_punc(line))
                input_length = max(len(words), input_length)
                questions.append(words)
            elif line != "" and i % 2 == 1:
                arr = line.strip().split(" ")
                ans = []
                for a in arr:
                    if a != "":
                        ans.append(int(a) - 1) # 因为原始数据从1开始计数，这里减去1。改为从0开始。
                answers.append(ans)

    question_vecs = []
    for q_words in questions:
        question_vecs.append(sentence2vec(w2v_model, q_words, input_length))
    print("len(question_vecs)", len(question_vecs))


    # 计算每个document的向量
    docs = []
    output_length = 0
    with open(doc_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line != "":
                words = word_tokenize(remove_punc(line))
                output_length = max(len(words), output_length)
                docs.append(words)
    doc_vecs = []
    output_length = args.output_length
    for d_words in docs:
        doc_vecs.append(sentence2vec(w2v_model, d_words, output_length))
    print("len(doc_vecs)",len(doc_vecs))
    logger.info("input_length:%d, output_length:%d" % (input_length, output_length))

    # 计算每个doc出现的频率
    doc_count = {}
    for ans in answers:
        for a in ans:
            if a in doc_count.keys():
                doc_count[a] += 1
            else:
                doc_count[a] = 1

    # 计算每个doc的weight
    doc_weight = {}
    t_max = 0
    for k in doc_count.keys():
        t_max = max(t_max, doc_count[k])
    for k in doc_count.keys():
        doc_weight[k] = doc_count[k] / t_max


    total = len(question_vecs)
    train_num = int(total * 0.9)
    logger.info("train_num:%d, total:%d" % (train_num, total))

    # 打乱数据
    qa_index = list( range(total) )
    random.shuffle(qa_index)


    step = 0
    while step * 200 <= train_num:
        # [q_encoder_input, r_decoder_input, w_decoder_input, weight_data_r, weight_data_w]
        q_encoder_input = []
        r_decoder_input = []
        w_decoder_input = []
        weight_data_r = []
        weight_data_w = []
        y_data = []

        qid_list = []
        label_list = []
        aid_list = []

        logger.info("step: %d" % step)

        end = min(train_num, (step + 1) * 200)
        for ss in range( step * 200, end):
            i = qa_index[ss]
            logger.info("question: %d" % i)
            qid_list.append(i)
            label_list.append(1)

            y = [1] + [0] * ns_amount
            y_data.append(y)
            # question
            q_encoder_input.append( question_vecs[i] )
            # 每个question一个正确答案
            aid = answers[i][0]
            aid_list.append(aid)
            r_decoder_input.append( doc_vecs[ aid ])
            weight_data_r.append(doc_weight[ aid ])
            # 10个un-related答案
            aids = get_randoms(list(doc_weight.keys()), [aid], 10)
            w_decoder = []
            w_weight = []
            for aid in aids:
                w_decoder.append( doc_vecs[aid] )
                w_weight.append( doc_weight[ aid ])

            w_decoder = np.array(w_decoder).reshape(output_length, args.input_dim, ns_amount)
            w_weight = np.array(w_weight).reshape((1, ns_amount))
            w_decoder_input.append(w_decoder)
            weight_data_w.append(w_weight)


            for aaid in aids:
                qid_list.append(i)
                label_list.append(0)
                aid_list.append(aaid)

                # 这些答案都是unrelated
                y = [0] * (1+ns_amount)
                y_data.append(y)
                # question
                q_encoder_input.append(question_vecs[i])

                r_decoder_input.append(doc_vecs[aaid])
                weight_data_r.append(doc_weight[aaid])
                # 10个un-related答案
                aids = get_randoms(list(doc_weight.keys()), [aid], 10)
                w_decoder = []
                w_weight = []
                for aid in aids:
                    w_decoder.append(doc_vecs[aid])
                    w_weight.append(doc_weight[aid])

                w_decoder = np.array(w_decoder).reshape(output_length, args.input_dim, ns_amount)
                w_weight = np.array(w_weight).reshape((1, ns_amount))
                w_decoder_input.append(w_decoder)
                weight_data_w.append(w_weight)


        logger.info("loading weights: ckpt/nn_weights_%s.h5" % data_type)
        model = negative_samples(input_length=input_length,
                                 input_dim=args.input_dim,
                                 output_length=output_length,
                                 output_dim=args.output_dim,
                                 hidden_dim=args.hidden_dim,
                                 ns_amount=ns_amount,
                                 learning_rate=args.learning_rate,
                                 drop_rate=args.drop_rate)
        model.load_weights("ckpt/nn_weights_%s.h5" % data_type)
        new_dnn_model = Model(inputs=model.input, outputs=model.get_layer('dropout_con').output)


        logger.info("predicting...")
        res = new_dnn_model.predict([q_encoder_input, r_decoder_input, w_decoder_input, weight_data_r, weight_data_w])
        # print(res)


        with open(to_file_path, "a") as f:
            for i in range(len(res)):
                row = res[i]
                feature_str = ''
                for j in range(len(row)):
                    feature_str = feature_str + (" %d:%.9f" % (j + 1, row[j]))
                label = label_list[i]
                id = qid_list[i]
                doc_id = aid_list[i]

                line = "%d qid:%d%s # doc-%d \n" % (label, id, feature_str,doc_id)
                f.write(line)
        print("saved to:", to_file_path)
        logger.info("step:%d added" % step)
        step += 1

    logger.info("saved to: %s" % to_file_path)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--data_type', help='data_type',  required=True)

    parser.add_argument('--input_dim', help='input_dim', type=int, default=200)
    parser.add_argument('--output_dim', help='output_dim', type=int, default=200)
    parser.add_argument('--hidden_dim', help='hidden_dim', type=int, default=64)
    parser.add_argument('--ns_amount', help='ns_amount', type=int, default=10)

    parser.add_argument('--learning_rate', help='learning_rate', type=float, default=0.001)
    parser.add_argument('--drop_rate', help='drop_rate', type=float, default=0.01)

    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--epochs', help='epochs', type=int, default=20)

    parser.add_argument('--output_length', help='output_length', type=int, default=1000)
    args = parser.parse_args()
    logger.info("training parameters %s", args)

    data_type = args.data_type

    logger.info("running generate_ltr_data.py, data_type:%s" % data_type)


    to_file_path = "for_ltr/ltr_%s_train.txt" % (data_type)

    if os.path.exists(to_file_path):
        logger.info("file exists: ", to_file_path)
        exit(0)


    model_path = "models/nn_%s.bin" % data_type

    w2v_path = "models/%s.wv.cbow.d%d.w10.n10.bin" % (data_type, args.input_dim)
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    qa_path = "%s/QA_list.txt" % data_type
    doc_path = "%s/Doc_list.txt" % data_type

    get_train_data(data_type, w2v_model, qa_path, doc_path, to_file_path, args)
