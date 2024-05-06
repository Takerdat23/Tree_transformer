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



        if args.strategy == "tree": 
            self.model = ABSA_Tree_transfomer(  vocab_size= self.vocab_size, N = modelConfig['N_layer'], No_consti =  modelConfig['Consti'], d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'],   dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        elif args.strategy == 'base' : 

            self.model = ABSA_transfomer( vocab_size= self.vocab_size, N = modelConfig['N_layer'], d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'] ,  dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        elif args.strategy == 'PretrainBERT' : 

            self.model = Constituent_Pretrained_transformer(  vocab_size= self.vocab_size, model = self.args.model_name, M = modelConfig['M_Constituent'] , d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'],   dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        elif args.strategy == 'PhoBert' : 

            self.model = Pretrained_transformer(model = self.args.model_name,  d_model= modelConfig['d_model'], dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        
        
         
        
        # elif args.strategy == 'PretrainBART' : 

        #     self.model = Constituent_Pretrained_BART( vocab_size= self.vocab_size, model = modelConfig["Model_name"] , M = modelConfig['M_Constituent'] , d_model= modelConfig['d_model'], 
        #                                   d_ff= modelConfig['d_ff'], h= modelConfig['heads'],num_categories = len(self.data_util.categories) ,  dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        
        
       
        
       

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

        all_topic_predictions = []
        all_sentiment_predictions = []
        all_topic_ground_truth = []
        all_sentiment_ground_truth = []

        with torch.no_grad():
            progress = tqdm(total=len(self.data_util.val_loader), position=0)
            for step, batch in tqdm(enumerate(self.data_util.val_loader)):
                
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                topics  = batch['topic'].to(device)
                sentiments = batch['sentiment'].to(device)

      
                topic_output, sentiment_output = self.model(inputs, mask)

               
               
            
                # Calculate loss

                topic_probs = torch.sigmoid(topic_output)
              
               
                sentiment_probs = torch.sigmoid(sentiment_output)

                _, topic_predictions = torch.max(topic_probs , dim = 1)
                _ , sentiment_predictions = torch.max(sentiment_probs , dim = 1)

                _, topic_groundtruth = torch.max(topics , dim = 1)
                _, sentiment_groundtruth = torch.max(sentiments , dim = 1)




                all_topic_predictions.append(topic_predictions.cpu().numpy())
                all_sentiment_predictions.append(sentiment_predictions.cpu().numpy())
                all_topic_ground_truth.append(topic_groundtruth.cpu().numpy())
                all_sentiment_ground_truth.append(sentiment_groundtruth.cpu().numpy())

                progress.update(1)

        all_topic_predictions = np.concatenate(all_topic_predictions)
        all_sentiment_predictions = np.concatenate(all_sentiment_predictions)
        all_topic_ground_truth = np.concatenate(all_topic_ground_truth)
        all_sentiment_ground_truth = np.concatenate(all_sentiment_ground_truth)

        
        print(all_topic_ground_truth.shape)
        print(all_topic_predictions.shape)


        aspect_precision = precision_score(all_topic_ground_truth.flatten(), all_topic_predictions.flatten(), average='weighted')
        aspect_recall = recall_score(all_topic_ground_truth.flatten(), all_topic_predictions.flatten(), average='weighted')
        topic_f1 = f1_score(all_topic_ground_truth.flatten(), all_topic_predictions.flatten(), average='weighted')

        sentiment_precision = precision_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')
        sentiment_recall = recall_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')
        sentiment_f1 = f1_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')

        print("Aspect Precision:", aspect_precision)
        print("Aspect Recall:", aspect_recall)
        print("Aspect F1 Score:", topic_f1)

        print("Sentiment Precision:", sentiment_precision)
        print("Sentiment Recall:", sentiment_recall)
        print("Sentiment F1 Score:", sentiment_f1)


        

        return aspect_precision, aspect_recall, topic_f1, sentiment_precision, sentiment_recall, sentiment_f1
    
    def test(self):
        if self.args.no_cuda == False:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        all_topic_predictions = []
        all_sentiment_predictions = []
        all_topic_ground_truth = []
        all_sentiment_ground_truth = []

        with torch.no_grad():
            progress = tqdm(total=len(self.data_util.test_loader), position=0)
            for step, batch in tqdm(enumerate(self.data_util.test_loader)):
                
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                topics  = batch['topic'].to(device)
                sentiments = batch['sentiment'].to(device)

      
                topic_output, sentiment_output = self.model(inputs, mask)

               
               
            
                # Calculate loss

                topic_probs = torch.sigmoid(topic_output)
              
               
                sentiment_probs = torch.sigmoid(sentiment_output)

                _, topic_predictions = torch.max(topic_probs , dim = 1)
                _ , sentiment_predictions = torch.max(sentiment_probs , dim = 1)

                _, topic_groundtruth = torch.max(topics , dim = 1)
                _, sentiment_groundtruth = torch.max(sentiments , dim = 1)




                all_topic_predictions.append(topic_predictions.cpu().numpy())
                all_sentiment_predictions.append(sentiment_predictions.cpu().numpy())
                all_topic_ground_truth.append(topic_groundtruth.cpu().numpy())
                all_sentiment_ground_truth.append(sentiment_groundtruth.cpu().numpy())

                progress.update(1)

        all_topic_predictions = np.concatenate(all_topic_predictions)
        all_sentiment_predictions = np.concatenate(all_sentiment_predictions)
        all_topic_ground_truth = np.concatenate(all_topic_ground_truth)
        all_sentiment_ground_truth = np.concatenate(all_sentiment_ground_truth)

        
        print(all_topic_ground_truth.shape)
        print(all_topic_predictions.shape)


        topic_precision = precision_score(all_topic_ground_truth.flatten(), all_topic_predictions.flatten(), average='weighted')
        topic_recall = recall_score(all_topic_ground_truth.flatten(), all_topic_predictions.flatten(), average='weighted')
        topic_f1 = f1_score(all_topic_ground_truth.flatten(), all_topic_predictions.flatten(), average='weighted')

        sentiment_precision = precision_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')
        sentiment_recall = recall_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')
        sentiment_f1 = f1_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')

        print("Aspect Precision:", topic_precision)
        print("Aspect Recall:", topic_recall)
        print("Aspect F1 Score:", topic_f1)

        print("Sentiment Precision:", sentiment_precision)
        print("Sentiment Recall:", sentiment_recall)
        print("Sentiment F1 Score:", sentiment_f1)


        

        return topic_precision,  topic_recall, topic_f1, sentiment_precision, sentiment_recall, sentiment_f1

   
    
    def save_model(self, model, optimizer, epoch, step, model_dir):
        model_name = f'model_epoch_{epoch}_step_{step}.pth'
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

        best_combined_accuracy = 0
        best_epoch = 0
        best_model_state = None

        try:
        
            for epoch in tqdm(range(self.args.epoch)):
                epoch_progress = tqdm(total=len(self.data_util.train_loader), desc=f'Epoch {epoch+1}/{self.args.epoch}', position=0)

                for step, batch in enumerate(self.data_util.train_loader):
                    inputs = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    topics  = batch['topic'].to(device)
                    sentiments = batch['sentiment'].to(device)

                    optim.zero_grad()
                    topic_output, sentiment_output = self.model(inputs, mask)

                
                
                
                    # Calculate loss

                    topic_probs = torch.sigmoid(topic_output)
                
                
                    sentiment_probs = torch.sigmoid(sentiment_output)
            
                    
                    topic_loss = F.binary_cross_entropy_with_logits( topic_probs, topics )

                    sentiment_loss = F.binary_cross_entropy_with_logits(sentiment_probs ,sentiments )

                    loss = topic_loss + sentiment_loss
                    
                    # loss = self.model.masked_lm_loss(output, labels)
                    total_loss.append(loss.item())

                    # Backpropagation
                    loss.backward()
                    optim.step()
                    if (self.args.wandb_api != ""):
                        wandb.log({"Loss": loss.item()}, step=epoch*len(self.data_util.train_loader) + step)
                    epoch_progress.update(1)
                    epoch_progress.set_postfix({'Loss': loss.item()})

                    if (step + 1) % 100 == 0:
                        elapsed = time.time() - start
                        print(f'Epoch [{epoch + 1}/{self.args.epoch}], Step [{step + 1}/{len(self.data_util.train_loader)}], '
                            f'Loss: {loss.item():.4f}, Total Time: {elapsed:.2f} sec')
                        # topic , sentiment = self.evaluate()
                
                        # print(f"Epoch {epoch} Validation accuracy (Aspect): ", topic)
                        # print(f"Epoch {epoch} Validation accuracy (Sentiment): ", sentiment)
                epoch_progress.close()
                #Valid stage 
                aspect_precision, aspect_recall, topic_f1, sentiment_precision, sentiment_recall, sentiment_f1 = self.evaluate()
                
                print(f"Epoch {epoch} Validation accuracy (Aspect): ", aspect_precision)
                print(f"Epoch {epoch} Validation accuracy (Sentiment): ", sentiment_precision)

                combined_accuracy = (aspect_precision + sentiment_precision) / 2
                if (self.args.wandb_api != ""):
                
                    wandb.log({"Validation Accuracy": combined_accuracy})
                if combined_accuracy > best_combined_accuracy:
                    best_combined_accuracy = combined_accuracy
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

        topic_precision,  topic_recall, topic_f1, sentiment_precision, sentiment_recall, sentiment_f1 = self.test()  

        if (self.args.wandb_api != ""):
              
                wandb.log({"Test sentiment_f1": sentiment_f1})
                wandb.log({"Test sentiment_recall": sentiment_recall})
                wandb.log({"Test aspect_precision": topic_precision})
                wandb.log({"Test sentiment_precision": sentiment_precision})
                wandb.log({"Test aspect_f1": topic_f1})
                wandb.log({"Test aspect_recall": topic_recall})
           
                    
                
                 
        self.save_model(self.model, optim, self.args.epoch, step, self.model_dir)


 