"""


@Time    : 10/30/20
@Author  : Wenbo
"""

import os
import time
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    ranklib_path = '/Users/woffee/www/gis_qa/ltrdemo2wenbo/utils/bin/RankLib.jar'

    # /Users/woffee/www/gis_qa/ltrdemo2wenbo/evaluation.py
    # trec_eval_path= '/Users/woffee/www/trec_eval-9.0.7/trec_eval'

    m_dict = {
        'RankNet': 1,
        'lambdaMART': 6
    }

    train_metric = 'MAP'
    train_model = 'RankNet'

    data_type = "ebay"
    data_train_path = "for_ltr/ltr_%s_train.txt" % data_type
    data_test_path = "for_ltr/ltr_%s_test.txt" % data_type

    save_model_path = BASE_DIR + '/ltr/models/' + data_type + "_" + train_model + '_' + train_metric + ".txt"
    data_pred_path = BASE_DIR + '/ltr/predicts/' + data_type + "_" + train_model + '_' + train_metric + "_pred.txt"

    # train
    ranker = m_dict[train_model]
    train_cmd = "java -jar %s -train %s -ranker %d -tvs 0.8 -metric2t MAP -save %s" % (ranklib_path, data_train_path, ranker, save_model_path)

    # pred
    pred_cmd = "java -jar %s -rank %s -load %s -indri %s" % (ranklib_path, data_test_path, save_model_path, data_pred_path)

    os.system(train_cmd)
    os.system(pred_cmd)

    print("done")





