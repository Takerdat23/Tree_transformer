# Tree_transformer


# Choice
tree base PretrainBERT PretrainBART PhoBert

# Training 
python main.py -train -strategy tree -seq_length 128 -batch_size 10 -model_dir ./Model/UIT-ViHSD -train_path ./data/ViHSD/train.csv -valid_path ./data/ViHSD/dev.csv -test_path ./data/ViHSD/test.csv -epoch 5 -wandb_api [your wandb key]


# Train tokenizer
python test.py -train_path ./data/train.txt -model_dir ./Model

# Testing

python main.py -tree -load -test -seq_length 512 -batch_size 32 -model_dir ./Model/UIT-ViHSD -train_path ./data/ViHSD/train.csv -valid_path ./data/ViHSD/dev.csv -test_path ./data/ViHSD/test.csv

