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

        preds = []
        targets = []

        with torch.no_grad():
            for step, batch in enumerate(self.data_util.val_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].float().to(device)

                with torch.no_grad():
                    pred_logits = self.model(inputs,mask)

                preds.append(pred_logits.squeeze().cpu().numpy().flatten())
                targets.append(labels.cpu().numpy().flatten())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        preds = (preds > 0.5).astype(int)

        precision = precision_score(targets , preds, average='macro')
        recall = recall_score(targets , preds, average='macro')
        f1 = f1_score(targets , preds, average='macro')

       



        result = {
            "Val Precision": precision,
            "Val Recall": recall,
            "Val F1 Score": f1
        }

        print(result)

        return precision, recall,  f1


    def test(self):
        if self.args.no_cuda == False:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        preds = []
        targets = []

        with torch.no_grad():
            for step, batch in enumerate(self.data_util.test_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].float().to(device)

                with torch.no_grad():
                    pred_logits = self.model(inputs,mask)

                preds.append(pred_logits.squeeze().cpu().numpy().flatten())
                targets.append(labels.cpu().numpy().flatten())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        preds = (preds > 0.5).astype(int)

        precision = precision_score(targets , preds, average='macro')
        recall = recall_score(targets , preds, average='macro')
        f1 = f1_score(targets , preds, average='macro')

       



        result = {
            "test Precision": precision,
            "test Recall": recall,
            "test F1 Score": f1
        }

        print(result)

        return precision, recall,  f1

   
    
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
     
        
        total_loss = []
        start = time.time()
     

        self.model.train()
        total_loss = []
        start = time.time()


        best_accuracy = 0
        best_epoch = 0
        best_model_state = None

        try:
        
            for epoch in tqdm(range(self.args.epoch)):
                epoch_progress = tqdm(total=len(self.data_util.train_loader), desc=f'Epoch {epoch+1}/{self.args.epoch}', position=0)

                for step, batch in enumerate(self.data_util.train_loader):
                    inputs = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    optim.zero_grad()
                    output = self.model.forward(inputs, mask)  # Assuming categories are not used for now
                
                    # Calculate loss
                
                    output = torch.sigmoid(output)
                    output = output.float()
                    labels = labels.float()


                    loss = F.binary_cross_entropy(output, labels)
                    
                    # loss = self.model.masked_lm_loss(output, labels)
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
                        # precision, recall,  f1 = self.evaluate()
                
                        
                epoch_progress.close()
                #Valid stage 
                precision , recall , f1 = self.evaluate()
                
                print(f"Epoch {epoch} Validation accuracy (Aspect): ", precision)
                

                
                if (self.args.wandb_api != ""):
                
                    wandb.log({"Validation Accuracy": precision}) 
            
                if precision > best_accuracy:
                        best_accuracy = precision
                        best_epoch = epoch
                        best_model_state = self.model.state_dict()
        
        except KeyboardInterrupt:
            if best_model_state != None: 
                print("Training interrupted. Saving the best model...")
                self.model.load_state_dict(best_model_state)
                self.save_model(self.model, optim, best_epoch, step, self.model_dir)
                print("Best model saved.")
            else: 
                print("Training interrupted. Saving model...")
                self.save_model(self.model, optim, best_epoch, step, self.model_dir)
                print("Model saved.")

            raise 

        self.model.load_state_dict(best_model_state)

        #Save the best model
        self.save_model(self.model, optim, self.args.epoch, step, self.model_dir)
        

        precision, recall , f1 = self.test()
        
        if (self.args.wandb_api != ""): 
            wandb.log({"test precision": precision})
            wandb.log({"test recall": recall })
            wandb.log({"test f1": f1})  
           
           
                
                 
        self.save_model(self.model, optim, self.args.epoch, step, self.model_dir)