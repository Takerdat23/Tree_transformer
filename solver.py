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
from sklearn.metrics import classification_report
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



        if args.strategy == "tree": 
            self.model = Tree_transfomer(  vocab_size= self.vocab_size, N = modelConfig['N_layer'], d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'],   dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        elif args.strategy == 'base' : 

            self.model = Transfomer( vocab_size= self.vocab_size, N = modelConfig['N_layer'], d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'] ,  dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        elif args.strategy == 'PretrainBERT' : 

            self.model = Constituent_Pretrained_transformer(  vocab_size= self.vocab_size, model = self.args.model_name, M = modelConfig['M_Constituent'] , d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'],   dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        elif args.strategy == 'PhoBert' : 

            self.model = Pretrained_transformer(model = self.args.model_name,  d_model= modelConfig['d_model'], dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        
        
         
       

        if self.args.load: 
            self.LoadPretrain()
            print(self.model)
        

     




    def ModelSummary(self): 
        print(self.model)
    
    def LoadPretrain(self): 
        path = os.path.join(self.args.model_dir, "model_epoch_50.pth")
        return self.model.load_state_dict(torch.load(path)['model_state_dict'])
    
  

    
    def evaluate(self):
        if not self.args.no_cuda:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        all_toxic_predictions = []
        all_toxic_ground_truths = []
        all_constructive_predictions = []
        all_constructive_ground_truths = []

        with torch.no_grad():
            epoch_progress = tqdm(total=len(self.data_util.val_loader), position=0)
            for step, batch in enumerate(self.data_util.val_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                toxic_labels = batch['toxicity'].to(device)
                construct_labels = batch['constructive'].to(device)

                toxic, construct  = self.model(inputs, mask)
                toxic = toxic.cpu()
                construct   = construct.cpu()

                # Extracting the index of the maximum score from each prediction
                toxic_predictions_indices = np.argmax(toxic, axis=-1)
                constructive_predictions_indices = np.argmax(construct, axis=-1)

                toxic_ground_truth_indices = np.argmax(toxic_labels.cpu().numpy(), axis=-1)
                constructive_ground_truth_indices = np.argmax(construct_labels.cpu().numpy(), axis=-1)

                toxic_label = ['No constructive', 'Constructive']
                constructive_label = ['No constructive', 'Constructive']


  

                all_toxic_predictions.extend(toxic_predictions_indices)
                all_toxic_ground_truths.extend(toxic_ground_truth_indices)

                all_constructive_predictions.extend(constructive_predictions_indices)
                all_constructive_ground_truths.extend(constructive_ground_truth_indices)
                epoch_progress.update(1)
        epoch_progress.close()

        # Generate classification reports
        report_toxic = classification_report(all_toxic_ground_truths, all_toxic_predictions, target_names=toxic_label)
        print("Aspect" , report_toxic)
        toxic_f1 = f1_score(all_toxic_ground_truths, all_toxic_predictions, average='weighted')

        report_constructive = classification_report(all_constructive_ground_truths, all_constructive_predictions, target_names=constructive_label)
        print("Sentiment" , report_constructive )
        constructive_f1 = f1_score(all_constructive_ground_truths, all_constructive_predictions, average='weighted')

        return toxic_f1  , constructive_f1


    def test(self):
        if not self.args.no_cuda:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        all_toxic_predictions = []
        all_toxic_ground_truths = []
        all_constructive_predictions = []
        all_constructive_ground_truths = []

        with torch.no_grad():
            epoch_progress = tqdm(total=len(self.data_util.val_loader), position=0)
            for step, batch in enumerate(self.data_util.test_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                toxic_labels = batch['toxicity'].to(device)
                construct_labels = batch['constructive'].to(device)

                toxic, construct  = self.model(inputs, mask)
                toxic = toxic.cpu()
                construct   = construct.cpu()

                # Extracting the index of the maximum score from each prediction
                toxic_predictions_indices = np.argmax(toxic, axis=-1)
                constructive_predictions_indices = np.argmax(construct, axis=-1)

                toxic_ground_truth_indices = np.argmax(toxic_labels.cpu().numpy(), axis=-1)
                constructive_ground_truth_indices = np.argmax(construct_labels.cpu().numpy(), axis=-1)

                toxic_label = ['No constructive', 'Constructive']
                constructive_label = ['No constructive', 'Constructive']


  

                all_toxic_predictions.extend(toxic_predictions_indices)
                all_toxic_ground_truths.extend(toxic_ground_truth_indices)

                all_constructive_predictions.extend(constructive_predictions_indices)
                all_constructive_ground_truths.extend(constructive_ground_truth_indices)
                epoch_progress.update(1)
        epoch_progress.close()

        # Generate classification reports
        report_toxic = classification_report(all_toxic_ground_truths, all_toxic_predictions, target_names=toxic_label)
        print("Toxic" , report_toxic )
        toxic_precision = precision_score(all_toxic_ground_truths, all_toxic_predictions, average='weighted')
        toxic_recall = recall_score(all_toxic_ground_truths, all_toxic_predictions, average='weighted')
        toxic_f1 = f1_score(all_toxic_ground_truths, all_toxic_predictions, average='weighted')

        report_constructive = classification_report(all_constructive_ground_truths, all_constructive_predictions, target_names=constructive_label)
        print("Constructive" , report_constructive  )
        
        constructive_precision = precision_score(all_constructive_ground_truths, all_constructive_predictions, average='weighted')
        constructive_recall = recall_score(all_constructive_ground_truths, all_constructive_predictions, average='weighted')
        constructive_f1 = f1_score(all_constructive_ground_truths, all_constructive_predictions, average='weighted')

        return toxic_precision, toxic_recall, toxic_f1, constructive_precision,  constructive_recall,  constructive_f1

   
    
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

        loss_fn = torch.nn.CrossEntropyLoss()
     

        self.model.train()
        total_loss = 0.0
        start = time.time()

        best_combined_accuracy = 0
        best_epoch = 0
        best_model_state = None

        try:
            
            for epoch in tqdm(range(self.args.epoch)):
                epoch_progress = tqdm(total=len(self.data_util.train_loader), desc=f'Epoch {epoch+1}/{self.args.epoch}', position=0)

                for step, batch in enumerate(self.data_util.train_loader):
                    inputs = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    toxic_labels = batch['toxicity'].to(device)
                    construct_labels = batch['constructive'].to(device)

                    optim.zero_grad()
                    toxic, construct = self.model.forward(inputs, mask)  

              

                    toxic_loss = loss_fn(toxic, toxic_labels)


                    construct_labels = loss_fn(construct, construct_labels)
                
            

                    loss = toxic_loss + construct_labels

                
              
                    total_loss += loss.item()
                    loss.backward()
                    optim.step()




                    if (self.args.wandb_api != ""):
                        wandb.log({"Loss": loss.item()})
                    epoch_progress.update(1)
                    epoch_progress.set_postfix({'Loss': loss.item()})

                    if (step + 1) % 100 == 0:
                        # toxic_f1  , constructive_f1= self.evaluate()
                        elapsed = time.time() - start
                        print(f'Epoch [{epoch + 1}/{self.args.epoch}], Step [{step + 1}/{len(self.data_util.train_loader)}], '
                            f'Loss: {loss.item():.4f}, Total Time: {elapsed:.2f} sec')
                      
                epoch_progress.close()
                #Valid stage 
                toxic_f1  , constructive_f1= self.evaluate()
                
                print(f"Epoch {epoch} Validation accuracy (Aspect): ",toxic_f1)
                print(f"Epoch {epoch} Validation accuracy (Sentiment): ", constructive_f1)

                combined_accuracy = (toxic_f1 + constructive_f1) / 2
                if (self.args.wandb_api != ""):
                
                    wandb.log({"Validation Accuracy": combined_accuracy})
                
                if combined_accuracy > best_combined_accuracy:
                    best_combined_accuracy = combined_accuracy
                    best_epoch = epoch
                    best_model_state = self.model.state_dict()
                
                # self.save_model(self.model, optim, epoch, step, self.model_dir)
                

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

        aspect_precision, aspect_recall, aspect_f1, constructive_precision,  constructive_recall,  constructive_f1 = self.test()
       
        if (self.args.wandb_api != ""):
              
                wandb.log({"Test toxic_precision": aspect_precision})
                wandb.log({"Test toxic_recall":aspect_recall})
                wandb.log({"Test toxic_f1": aspect_f1})
                wandb.log({"Test constructive_precision": constructive_precision})
                wandb.log({"Test constructive_recall": constructive_recall})
                wandb.log({"Test constructive_f1":  constructive_f1})
           
                    
                 
                 
        