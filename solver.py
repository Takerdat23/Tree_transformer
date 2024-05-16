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
            self.model = ABSA_Tree_transfomer(  vocab_size= self.vocab_size, N = modelConfig['N_layer'], d_model= modelConfig['d_model'], 
                                          d_ff= modelConfig['d_ff'], h= modelConfig['heads'],   dropout = modelConfig['dropout'], no_cuda=args.no_cuda)
        elif args.strategy == 'base' : 

            self.model = ABSA_transfomer( vocab_size= self.vocab_size, N = modelConfig['N_layer'], d_model= modelConfig['d_model'], 
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
    
  

    
    # def evaluate(self):
    #     if self.args.no_cuda == False:
    #         device = "cuda"
    #     else:
    #         device = "cpu"

    #     self.model.to(device)
    #     self.model.eval()

    #     all_aspect_predictions = []
    #     all_sentiment_predictions = []
    #     all_aspect_ground_truth = []
    #     all_sentiment_ground_truth = []

    #     with torch.no_grad():
    #         for step, batch in enumerate(self.data_util.val_loader):
    #             inputs = batch['input_ids'].to(device)
    #             mask = batch['attention_mask'].to(device)
    #             labels = batch['labels'].to(device)

    #             output = self.model(inputs, mask, self.data_util.categories)

    #             output = torch.sigmoid(output)
    #             output = output.float()

               

    #             y_test_argmax = np.argmax(output.cpu(), axis=-1)
    #             print("labels: ", y_test_argmax.shape)


    #             aspect_predictions = (output[:, :, 0] > 0.5).long()
    #             sentiment_predictions = (output[:, :, 1:] > 0.5).long()

    #             aspect_ground_truth = labels[:, :, 0].long()
    #             sentiment_ground_truth = labels[:, :, 1:].long()

    #             all_aspect_predictions.append(aspect_predictions.cpu().numpy())
    #             all_sentiment_predictions.append(sentiment_predictions.cpu().numpy())
    #             all_aspect_ground_truth.append(aspect_ground_truth.cpu().numpy())
    #             all_sentiment_ground_truth.append(sentiment_ground_truth.cpu().numpy())

    #     all_aspect_predictions = np.concatenate(all_aspect_predictions)
    #     all_sentiment_predictions = np.concatenate(all_sentiment_predictions)
    #     all_aspect_ground_truth = np.concatenate(all_aspect_ground_truth)
    #     all_sentiment_ground_truth = np.concatenate(all_sentiment_ground_truth)

        


    #     all_aspect_predictions = np.concatenate(all_aspect_predictions)
    #     all_sentiment_predictions = np.concatenate(all_sentiment_predictions)
    #     all_aspect_ground_truth = np.concatenate(all_aspect_ground_truth)
    #     all_sentiment_ground_truth = np.concatenate(all_sentiment_ground_truth)

    #     aspect_precision = precision_score(all_aspect_ground_truth.flatten(), all_aspect_predictions.flatten())
    #     aspect_recall = recall_score(all_aspect_ground_truth.flatten(), all_aspect_predictions.flatten())
    #     aspect_f1 = f1_score(all_aspect_ground_truth.flatten(), all_aspect_predictions.flatten())

    #     sentiment_precision = precision_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')
    #     sentiment_recall = recall_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')
    #     sentiment_f1 = f1_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')

    #     result = {
    #         "Val Aspect Precision": aspect_precision,
    #         "Val Aspect Recall": aspect_recall,
    #         "Val Aspect F1 Score": aspect_f1,
    #         "Val Sentiment Precision": sentiment_precision,
    #         "Val Sentiment Recall": sentiment_recall,
    #         "Val Sentiment F1 Score": sentiment_f1
    #     }

    #     print(result)



        

    #     return aspect_precision, aspect_recall, aspect_f1, sentiment_precision, sentiment_recall, sentiment_f1
    
    def evaluate(self):
        if not self.args.no_cuda:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        all_aspect_predictions = []
        all_aspect_ground_truths = []
        all_sentiment_predictions = []
        all_sentiment_ground_truths = []

        with torch.no_grad():
            for step, batch in enumerate(self.data_util.val_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = self.model(inputs, mask, self.data_util.categories)
                output = output.cpu()

                # Extracting the index of the maximum score from each prediction
                predictions_indices = np.argmax(output, axis=-1)
                ground_truth_indices = np.argmax(labels.cpu().numpy(), axis=-1)

            

                categories = self.data_util.categories.append(" ")
                sentiment = ['None', "Positive", "Negative", "Neutral"]

                for pred , true in zip(predictions_indices , ground_truth_indices): 
          
                    predictions_Aspect = [self.data_util.categories[idx] if sentiment != 0 else " " for idx , sentiment in enumerate(pred) ]
                    ground_truths_Aspect = [self.data_util.categories[idx] if sentiment != 0 else " " for idx ,  sentiment in enumerate(true) ]

                    predictions_Sentiment = [sentiment[idx]  for idx in pred]
                    ground_truths_Sentiment = [sentiment [idx]  for idx in true ]

                    all_aspect_predictions.extend(predictions_Aspect)
                    all_aspect_ground_truths.extend(ground_truths_Aspect)

                    all_sentiment_predictions.extend(predictions_Sentiment)
                    all_sentiment_ground_truths.extend(ground_truths_Sentiment)

        # Generate classification reports
        report_aspect = classification_report(all_aspect_ground_truths, all_aspect_predictions, target_names=categories)
        print("Aspect" , report_aspect )
        aspect_f1 = f1_score(all_aspect_ground_truths, all_aspect_predictions, average='weighted')

        report_sentiment = classification_report(all_sentiment_ground_truths, all_sentiment_predictions, target_names=sentiment)
        print("Sentiment" , report_sentiment )
        Sentiment_f1 = f1_score(all_sentiment_ground_truths, all_sentiment_predictions, average='weighted')

        return aspect_f1 , Sentiment_f1

    
    # def test(self):
    #     if self.args.no_cuda == False:
    #         device = "cuda"
    #     else:
    #         device = "cpu"

    #     self.model.to(device)
    #     self.model.eval()

    #     all_aspect_predictions = []
    #     all_sentiment_predictions = []
    #     all_aspect_ground_truth = []
    #     all_sentiment_ground_truth = []

    #     with torch.no_grad():
    #         for step, batch in enumerate(self.data_util.test_loader):
    #             inputs = batch['input_ids'].to(device)
    #             mask = batch['attention_mask'].to(device)
    #             labels = batch['labels'].to(device)

    #             output = self.model(inputs, mask, self.data_util.categories)

    #             output = torch.sigmoid(output)
    #             output = output.float()

    #             aspect_predictions = (output[:, :, 0] > 0.5).long()
    #             sentiment_predictions = (output[:, :, 1:] > 0.5).long()

    #             aspect_ground_truth = labels[:, :, 0].long()
    #             sentiment_ground_truth = labels[:, :, 1:].long()

    #             all_aspect_predictions.append(aspect_predictions.cpu().numpy())
    #             all_sentiment_predictions.append(sentiment_predictions.cpu().numpy())
    #             all_aspect_ground_truth.append(aspect_ground_truth.cpu().numpy())
    #             all_sentiment_ground_truth.append(sentiment_ground_truth.cpu().numpy())

    #     all_aspect_predictions = np.concatenate(all_aspect_predictions)
    #     all_sentiment_predictions = np.concatenate(all_sentiment_predictions)
    #     all_aspect_ground_truth = np.concatenate(all_aspect_ground_truth)
    #     all_sentiment_ground_truth = np.concatenate(all_sentiment_ground_truth)

    #     aspect_precision = precision_score(all_aspect_ground_truth.flatten(), all_aspect_predictions.flatten())
    #     aspect_recall = recall_score(all_aspect_ground_truth.flatten(), all_aspect_predictions.flatten())
    #     aspect_f1 = f1_score(all_aspect_ground_truth.flatten(), all_aspect_predictions.flatten())

    #     sentiment_precision = precision_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')
    #     sentiment_recall = recall_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')
    #     sentiment_f1 = f1_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten(), average='weighted')

    #     ressult = {
    #         "Test Aspect Precision": aspect_precision,
    #         "Test Aspect Recall": aspect_recall,
    #         "Test Aspect F1 Score": aspect_f1,
    #         "Test Sentiment Precision": sentiment_precision,
    #         "Test Sentiment Recall": sentiment_recall,
    #         "Test Sentiment F1 Score": sentiment_f1
    #     }
    #     print(ressult)

    #     return aspect_precision, aspect_recall, aspect_f1, sentiment_precision, sentiment_recall, sentiment_f1


    def test(self):
        if not self.args.no_cuda:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        all_aspect_predictions = []
        all_aspect_ground_truths = []
        all_sentiment_predictions = []
        all_sentiment_ground_truths = []

        with torch.no_grad():
            for step, batch in enumerate(self.data_util.test_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = self.model(inputs, mask, self.data_util.categories)
                output = output.cpu()

                # Extracting the index of the maximum score from each prediction
                predictions_indices = np.argmax(output, axis=-1)
                ground_truth_indices = np.argmax(labels.cpu().numpy(), axis=-1)

            

                categories = self.data_util.categories.append(" ")
                sentiment = ['None', "Positive", "Negative", "Neutral"]

                for pred , true in zip(predictions_indices , ground_truth_indices): 
          
                    predictions_Aspect = [self.data_util.categories[idx] if sentiment != 0 else " " for idx , sentiment in enumerate(pred) ]
                    ground_truths_Aspect = [self.data_util.categories[idx] if sentiment != 0 else " " for idx ,  sentiment in enumerate(true) ]

                    predictions_Sentiment = [sentiment[idx]  for idx in pred]
                    ground_truths_Sentiment = [sentiment [idx]  for idx in true ]

                    all_aspect_predictions.extend(predictions_Aspect)
                    all_aspect_ground_truths.extend(ground_truths_Aspect)

                    all_sentiment_predictions.extend(predictions_Sentiment)
                    all_sentiment_ground_truths.extend(ground_truths_Sentiment)

        # Generate classification reports
        report_aspect = classification_report(all_aspect_ground_truths, all_aspect_predictions, target_names=categories)
        print("Aspect" , report_aspect )
        aspect_precision = precision_score(all_aspect_ground_truths, all_aspect_predictions, average='weighted')
        aspect_recall = recall_score(all_aspect_ground_truths, all_aspect_predictions, average='weighted')
        aspect_f1 = f1_score(all_aspect_ground_truths, all_aspect_predictions, average='weighted')

        report_sentiment = classification_report(all_sentiment_ground_truths, all_sentiment_predictions, target_names=sentiment)
        print("Sentiment" , report_sentiment )
        
        sentiment_precision = precision_score(all_sentiment_ground_truths, all_sentiment_predictions, average='weighted')
        sentiment_recall = recall_score(all_sentiment_ground_truths, all_sentiment_predictions, average='weighted')
        Sentiment_f1 = f1_score(all_sentiment_ground_truths, all_sentiment_predictions, average='weighted')

        return aspect_precision, aspect_recall, aspect_f1, sentiment_precision,  sentiment_recall,  Sentiment_f1

   
    
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
                    labels = batch['labels'].to(device)

                    optim.zero_grad()
                    output = self.model.forward(inputs, mask, self.data_util.categories)  # Assuming categories are not used for now
                
                    # Calculate loss
                
                    output = output.float()
                    labels = labels.float()

                    # output_argmax = torch.argmax(output, dim = -1)

                    # label_argmax = torch.argmax(labels, dim = -1)



                    loss = loss_fn(output , labels)

                
              
                    total_loss += loss.item()
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
                      
                epoch_progress.close()
                #Valid stage 
                aspect_f1, Sentiment_f1 = self.evaluate()
                
                print(f"Epoch {epoch} Validation accuracy (Aspect): ", aspect_f1)
                print(f"Epoch {epoch} Validation accuracy (Sentiment): ", Sentiment_f1)

                combined_accuracy = (aspect_f1 + Sentiment_f1) / 2
                if (self.args.wandb_api != ""):
                
                    wandb.log({"Validation Accuracy": combined_accuracy})
                
                if combined_accuracy > best_combined_accuracy:
                    best_combined_accuracy = combined_accuracy
                    best_epoch = epoch
                    best_model_state = self.model.state_dict()
                
                self.save_model(self.model, optim, epoch, step, self.model_dir)
                

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

        aspect_precision, aspect_recall, aspect_f1,sentiment_precision,  sentiment_recall, sentiment_f1  = self.test()
       
        if (self.args.wandb_api != ""):
              
                wandb.log({"Test aspect_precision": aspect_precision})
                wandb.log({"Test aspect_recall":aspect_recall})
                wandb.log({"Test aspect_f1": aspect_f1})
                wandb.log({"Test sentiment_precision": sentiment_precision})
                wandb.log({"Test sentiment_recall": sentiment_recall})
                wandb.log({"Test sentiment_f1":  sentiment_f1})
           
                    
                 
                 
        