# Tree_transformer

# Pretraining

python main.py -train -seq_length 100 -batch_size 64 -model_dir ./Model2 -train_path ./data/demo-full.txt -num_step 60000

# Training 
python main.py -train -tree -seq_length 128 -batch_size 10 -model_dir ./Model/ViHOS -train_path ./data/ViHOS/train.csv -valid_path ./data/ViHOS/dev.csv -epoch 5 -wandb_api [your wandb key]


# Train tokenizer
python test.py -train_path ./data/train.txt -model_dir ./Model

# Testing

python main.py -load -test -seq_length 512 -batch_size 32 -model_dir ./Model/ViSFD/No_segment_AIAYN/ -train_path ./data/UIT-ViSFD/Train.csv -valid_path ./data/UIT-ViSFD/Dev.csv -test_path ./data/UIT-ViSFD/Test.csv

