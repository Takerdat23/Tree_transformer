# Tree_transformer

# Pretraining

python main.py -train -seq_length 100 -batch_size 64 -model_dir ./Model2 -train_path ./data/demo-full.txt -num_step 60000

# Training 
python main.py -train -seq_length 128 -batch_size 32 -model_dir ./Model -train_path ./data/UIT-ViSFD/Train.csv -valid_path ./data/UIT-ViSFD/Dev.csv -epoch 5 -wandb_api [your wandb key]

# Testing

python main.py -load -test -seq_length 128 -batch_size 32 -model_dir ./Model/ABSA/model_epoch_50_step_243.pth -train_path ./data/UIT-ViSFD/Train.csv -valid_path ./data/UIT-ViSFD/Dev.csv -test_path ./data/UIT-ViSFD/Test.csv