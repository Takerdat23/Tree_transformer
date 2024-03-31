# Tree_transformer

# Pretraining

python main.py -train -seq_length 100 -batch_size 64 -model_dir ./Model2 -train_path ./data/demo-full.txt -num_step 60000

# Training 
python main.py -train -seq_length 128 -batch_size 10 -model_dir ./Model -train_path ./data/UIT-VSF/Train.csv -valid_path ./data/UIT-VSF/Dev.csv -epoch 5 -wandb_api [your wandb key]


# Train tokenizer
python test.py -train_path ./data/train.txt -model_dir ./Model

# Testing

python main.py -tree -load -test -seq_length 512 -batch_size 32 -model_dir ./Model/VSF/Original_transformer_tree -train_path ./data/UIT-VSF/Train.csv -valid_path ./data/UIT-VSF/Dev.csv -test_path ./data/UIT-VSF/Test.csv

segment

python main.py -load -test -seq_length 512 -batch_size 32 -model_dir ./Model/VSF/Original_Transformer_segment -train_path ./data/UIT-VSF/SegTrain.csv -valid_path ./data/UIT-VSF/SegDev.csv -test_path ./data/UIT-VSF/SegTest.csv