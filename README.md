# Tree_transformer

# Pretraining

python main.py -train -seq_length 100 -batch_size 64 -model_dir ./Model2 -train_path ./data/demo-full.txt -num_step 60000

tree base PretrainBERT
# Training 
python main.py -train -strategy PretrainBERT -model_name 'vinai/phobert-base' -config '{"M_Constituent": 1 , "d_model": 768, "d_ff": 3072, "heads": 12, "dropout" :0.1}' -seq_length 128 -batch_size 10 -model_dir ./Model/ABSA -train_path ./data/UIT-VSF/Train.csv -valid_path ./data/UIT-VSF/Dev.csv -epoch 5 -wandb_api [your wandb key]


# Train tokenizer
python test.py -train_path ./data/train.txt -model_dir ./Model

# Testing

python main.py -load -test -seq_length 512 -batch_size 32 -model_dir ./Model/VSF/No_segment_bert_base/ -train_path .\data\UIT-VSF\Train.csv -valid_path .\data\UIT-VSF\Dev.csv -test_path .\data\UIT-VSF\Test.csv

