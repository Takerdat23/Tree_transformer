# Tree_transformer

# Pretraining

python main.py -train -seq_length 100 -batch_size 64 -model_dir ./Model2 -train_path ./data/demo-full.txt -num_step 60000


tree base PretrainBERT PretrainBART PhoBert
# Training 
python main.py -train -strategy PretrainBERT -seq_length 128 -batch_size 10 -model_dir ./Model -train_path ./data/VLSP-2018/Hotel/Hotel_train.csv -valid_path ./data/VLSP-2018/Hotel/Hotel_dev.csv -epoch 5 -wandb_api [your wandb key]


# Train tokenizer
python test.py -train_path ./data/train.txt -model_dir ./Model

# Testing

python main.py -load -test -seq_length 512 -batch_size 32 -model_dir ./Model/VLSP2018/Restaurant/No_Segment_bert -train_path ./data/VLSP-2018/Restaurant/Restaurant_train.csv -valid_path ./data/VLSP-2018/Restaurant/Restaurant_dev.csv -test_path ./data/VLSP-2018/Restaurant/Restaurant_test.csv