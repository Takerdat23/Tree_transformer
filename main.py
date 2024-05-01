import argparse
from solver import Solver
import json

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-no_cuda', action='store_true', help="Don't use GPUs.")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-seq_length', type=int, default=50, help='sequence length')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_step', type=int, default=100000, help='sequence length')
    parser.add_argument('-epoch', type=int, default=10, help='sequence length')
    parser.add_argument('-data_dir',default='data_dir',help='data dir')
    parser.add_argument('-load',action='store_true',help='load pretrained model')
    parser.add_argument('-tree',action='store_true',help='load pretrained model')
    parser.add_argument('-strategy',default=None,help='Model type')
    parser.add_argument('-segment',action='store_true',help='segment or not')
    parser.add_argument('-config', default=None, type=str, help="Model config" )
    parser.add_argument('-train', action='store_true',help='whether train the model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-valid_path',default='data/valid.txt',help='validation data path')
    parser.add_argument('-train_path',default='data/train.txt',help='training data path')
    parser.add_argument('-test_path',default='data/test.txt',help='testing data path')
    parser.add_argument('-wandb_api',default='',help='wandb api key')
    parser.add_argument('-wandb_Project',default='default',help='wandb project name')
    parser.add_argument('-wandb_RunName',default='Run',help='wandb run name')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    print(solver.ModelSummary())
    
    if args.train:
        solver.train()
    elif args.test:
        print("VALID SET")
        solver.evaluate()
        print("TEST SET")
        solver.test()
