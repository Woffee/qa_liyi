

data_type="twitter"

python word2vec.py $data_type
python Model.py $data_type
python generate_ltr_data.py $data_type
python generate_ltr_testdata.py $data_type

