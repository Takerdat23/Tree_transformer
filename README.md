# Tree_transformer

# Choice
tree base PretrainBERT PretrainBART PhoBert

# Training 
python main.py -train -strategy tree -seq_length 128 -batch_size 10 -model_dir ./Model/UIT-ViCTSD -train_path ./data/UIT-ViCTSD/Train.csv -valid_path ./data/UIT-ViCTSD/Dev.csv -test_path ./data/UIT-ViCTSD/Test.csv -epoch 5 -wandb_api [your wandb key]


lstm

python main.py -train -strategy lstm -seq_length 128 -batch_size 10 -model_dir ./Model/UIT-ViCTSD -train_path ./data/UIT-ViCTSD/Train.csv -valid_path ./data/UIT-ViCTSD/Dev.csv -test_path ./data/UIT-ViCTSD/Test.csv -epoch 5 -wandb_api [your wandb key]


# Train tokenizer
python test.py -train_path ./data/train.txt -model_dir ./Model

# Testing

python main.py -tree -load -test -seq_length 512 -batch_size 32 -model_dir ./Model/ViSFD/Tree_original/ -train_path ./data/UIT-ViSFD/Train.csv -valid_path ./data/UIT-ViSFD/Dev.csv -test_path ./data/UIT-ViSFD/Test.csv

