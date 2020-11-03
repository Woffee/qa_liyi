now_time=`date +"%Y-%m-%d-%H-%M"`
log_file="${now_time}.log"

data_type="twitter"
input_dim=300
output_dim=300
hidden_dim=64
ns_amount=10
learning_rate=0.001
drop_rate=0.01
batch_size=32
epochs=20
output_length=1000

args="--data_type ${data_type} --input_dim ${input_dim} --output_dim ${output_dim} --hidden_dim ${hidden_dim} --ns_amount ${ns_amount} --learning_rate ${learning_rate} --drop_rate ${drop_rate} --batch_size ${batch_size} --epochs ${epochs} --output_length ${output_length}"
echo $args

python word2vec.py --data_type $data_type --input_dim $input_dim
python Model.py $args
python generate_ltr_data.py $args
python generate_ltr_testdata.py $args

