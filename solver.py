import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from parse import *
import random
from bert_optimizer import BertAdam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import wandb

class Solver():
    def __init__(self, args):
        self.args = args

        self.model_dir = make_save_dir(args.model_dir)
        self.no_cuda = args.no_cuda

        self.data_util = data_utils(args)
        self.vocab_size = 64000

        if args.config: 
            

            modelConfig = json.loads(args.config)
   
        else: 


            modelConfig =  read_json("./model_config.json")
   
    
        if args.wandb_api != "": 

            wandb.login(key=args.wandb_api)



        if args.tree: 
            self.model = Tree_transfomer(  vocab_size= self.vocab_size, N = modelConfig['N_layer'], 
                                          d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'],   
                                          dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        else: 

            self.model = Transfomer( vocab_size= self.vocab_size, N = modelConfig['N_layer'], 
                                          d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'] ,  
                                          dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        
        
       

        if self.args.load: 
            self.LoadPretrain()
            print(self.model)
        

     




    def ModelSummary(self): 
        print(self.model)
    
    def LoadPretrain(self): 
        path = os.path.join(self.args.model_dir, "model_epoch_50.pth")
        return self.model.load_state_dict(torch.load(path)['model_state_dict'])
    

    
    def evaluate(self):
        if self.args.no_cuda == False:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        span_preds = []
        span_targets = []

        with torch.no_grad():
            for step, batch in enumerate(self.data_util.val_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                spans = batch['spans'].float().to(device)

                with torch.no_grad():
                    span_logits = self.model(inputs,mask)

                span_preds.append(span_logits.squeeze().cpu().numpy().flatten())
                span_targets.append(spans.cpu().numpy().flatten())

        span_preds = np.concatenate(span_preds)
        span_targets = np.concatenate(span_targets)
        span_preds = (span_preds > 0.5).astype(int)

        precision = precision_score(span_targets, span_preds, average='weighted')
        recall = recall_score(span_targets, span_preds, average='weighted')
        span_f1 = f1_score(span_targets, span_preds, average='weighted')

       



        result = {
            "Val Span Precision": precision,
            "Val Span Recall": recall,
            "Val Span F1 Score": span_f1
        }

        print(result)

        return precision, recall,  span_f1
    
    def test(self):
        if self.args.no_cuda == False:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        span_preds = []
        span_targets = []

        with torch.no_grad():
            for step, batch in enumerate(self.data_util.test_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                spans = batch['spans'].float().to(device)

                with torch.no_grad():
                    span_logits = self.model(inputs,mask)

                span_preds.append(span_logits.squeeze().cpu().numpy().flatten())
                span_targets.append(spans.cpu().numpy().flatten())

        span_preds = np.concatenate(span_preds)
        span_targets = np.concatenate(span_targets)
        span_preds = (span_preds > 0.5).astype(int)

        precision = precision_score(span_targets, span_preds, average='weighted')
        recall = recall_score(span_targets, span_preds, average='weighted')
        span_f1 = f1_score(span_targets, span_preds, average='weighted')


        return precision, recall,  span_f1
    
    def save_model(self, model, optimizer, epoch, step, model_dir):
        model_name = f'model_epoch_{epoch}.pth'
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(model_dir, model_name))


    def train(self):
        if self.args.no_cuda == False: 
            device = "cuda"
        else: 
            device = "cpu"
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])
        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #print(name)
                ttt = 1
                for  s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:',tt)
        if (self.args.wandb_api != ""):
            wandb.init(project=self.args.wandb_Project, name=self.args.wandb_RunName)


        self.model.to(device)
     
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        #optim = BertAdam(self.model.parameters(), lr=1e-4)
        criterion_span = nn.BCELoss()
     
        
        total_loss = []
        start = time.time()
     

        self.model.train()
        total_loss = []
        start = time.time()
        
        for epoch in tqdm(range(self.args.epoch)):
            epoch_progress = tqdm(total=len(self.data_util.train_loader), desc=f'Epoch {epoch+1}/{self.args.epoch}', position=0)

            for step, batch in enumerate(self.data_util.train_loader):
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].to(device)
                spans = batch['spans'].float().to(device)

                optim.zero_grad()
                span_logits = self.model(input_ids, attention_mask)
                loss_span = criterion_span(span_logits.squeeze(), spans)

                loss = loss_span
            

         
           
                total_loss.append(loss.item())


                # Backpropagation
                loss.backward()
                optim.step()
                if (self.args.wandb_api != ""):
                    wandb.log({"Loss": loss.item()})
                epoch_progress.update(1)
                epoch_progress.set_postfix({'Loss': loss.item()})

                if (step + 1) % 100 == 0:
                    elapsed = time.time() - start
                    print(f'Epoch [{epoch + 1}/{self.args.epoch}], Step [{step + 1}/{len(self.data_util.train_loader)}], '
                        f'Loss: {loss.item():.4f}, Total Time: {elapsed:.2f} sec')
                    # Score = self.evaluate()
               
                    # print(f"Epoch {epoch} Validation accuracy: ", Score)
                    # print(f"Epoch {epoch} Validation accuracy (Sentiment): ", sentiment)
            epoch_progress.close()
            #Valid stage 
            precision, recall,  span_f1 = self.evaluate()
               
            print(f"Epoch {epoch} Validation accuracy: ",  precision)

            
            if (self.args.wandb_api != ""):
                
              
                wandb.log({"Validation Accuracy": precision})
                wandb.log({"Validation Recall": recall})
                wandb.log({"Validation F1_score": span_f1})

            
            #testing
            
            precision, recall,  span_f1= self.test()

            if (self.args.wandb_api != ""):
              
                wandb.log({"Test Precision": precision})
                wandb.log({"Test Recall": recall})
                wandb.log({"Test F1-score": span_f1})

           
                    
                
                 
        self.save_model(self.model, optim, self.args.epoch, step, self.model_dir)