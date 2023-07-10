from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ( get_linear_schedule_with_warmup,
                           RobertaForSequenceClassification)
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import torch
import logging
import torch.nn as nn
import re
from tqdm import tqdm
from transformers import RobertaForSequenceClassification

logging.basicConfig(filename='linelevel.log', level=logging.DEBUG)

#  same as classification layer in LineVul model
class Clasificationlayer(nn.Module):
    """Classification layer for function-level classification."""
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
         # get CLS vector which represent the whole function
        cls = features[:, 0, :] 
        x = self.dropout(cls)
        # dense layer
        cls = self.dense(cls)
        # tanh activation function
        cls = torch.tanh(cls)
        cls = self.dropout(cls)
        # last dense layer to get the prob for each class
        cls = self.out_proj(cls)
        return cls

class Model(RobertaForSequenceClassification,pl.LightningModule):
    """Model class for function-level classification."""

    def __init__(self, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        # get the pretrained model
        self.encoder = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)  
        self.tokenizer = tokenizer
        self.classifier = Clasificationlayer(config)
        self.args = args
    
    def forward(self, input_ids=None, labels=None,attention_mask=None, output_attentions=False):
        """ Forward pass for function-level classification. """

        if output_attentions:  
            outputs = self.encoder.roberta(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            # get the attention for 12 layer torch.Size([1, 12, 512, 512])
            # each token has the attention with the other 512 token
            attentions = outputs.attentions
            # give the last hidden state fromthe last transformer layer to the classifier layer to get the prob for each class
            predictions = self.classifier(outputs.last_hidden_state)
            prob = torch.softmax(predictions, dim=-1)
            # get the loss if the labels is exist
            if labels is not None:
                weight = torch.FloatTensor(self.args.class_weight).cuda()
                cross_entropy_function = nn.CrossEntropyLoss()
                loss =cross_entropy_function(predictions, labels) 
                return loss, prob, attentions
            else:
                return prob, attentions
        else:    
            outputs = self.encoder.roberta(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            # give the last hidden state fromthe last transformer layer to the classifier layer to get the prob for each class
            predictions = self.classifier(outputs.last_hidden_state)
            prob = torch.softmax(predictions, dim=-1)
            # get the loss if the labels is exist
            if labels is not None:
                weight = torch.FloatTensor(self.args.class_weight).cuda()
                cross_entropy_function = nn.CrossEntropyLoss()
                loss =cross_entropy_function(predictions, labels) 
                return loss, prob
            else:
                return prob
    def training_step(self, batch, batch_index):
        """ Training step for function-level classification. """
        # get the loss and prob for each class by calling the forward function on the batch
        loss, outputs = self(**batch)
        self.log("train loss ", loss, prog_bar = True, logger=True)
        return {"loss":loss, "predictions":outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        """ Validation step for function-level classification. """
        # get the loss and prob for each class by calling the forward function on the batch
        loss, outputs = self(**batch)
        self.log("validation loss ", loss, prog_bar = True, logger=True)
        return {"val_loss": loss, "predictions":outputs, "labels": batch["labels"]}
    
    def validation_epoch_end(self, outputs):
        """ The values that returned from the validation step will be passed to this function in the outputs list
            and we use this to get the predictions and the labels and then calculate the validation report"""
        y_preds = torch.cat([torch.argmax(x["predictions"],dim=1) for x in outputs])
        y_true = torch.cat([x["labels"] for x in outputs])
        # convert the predictions and labels to numpy array to be able to calculate the report
        y_preds = y_preds.cpu().numpy()
        y_true = y_true.cpu().numpy()
        report = classification_report(y_true, y_preds)
        logging.info("Validation report %s", report)
    
    def test_step(self, batch, batch_index):
        """ Test step for function-level classification. """
        # get the loss and prob for each class by calling the forward function on the batch
        loss, outputs = self(**batch)
        predictions = torch.argmax(outputs, dim=1)
        return {"predictions":predictions, "labels": batch["labels"]}
    
    def test_epoch_end(self, outputs):
        """ The values that returned from the test step will be passed to this function in the outputs list
            and we use this to get the predictions and the labels and then calculate the test report"""
        y_preds = torch.cat([x["predictions"] for x in outputs])
        y_true = torch.cat([x["labels"] for x in outputs])
        # convert the predictions and labels to numpy array to be able to calculate the report
        y_preds = y_preds.cpu().numpy()
        y_true = y_true.cpu().numpy()
        report = classification_report(y_true, y_preds)
        logging.info("Test report %s", report)

        
    def predict_step(self, batch, batch_index):
        """ Predict step for function-level classification. """
        loss, outputs = self(input_ids=batch[0],labels=batch["labels"])
        predictions = torch.argmax(outputs, dim=1)
        return predictions
  


    def configure_optimizers(self):
         # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        #self.args.warmup_steps = self.args.max_steps // 5
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=self.args.max_steps)
        #return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return [optimizer], [scheduler]
    

class UCC_Dataset(Dataset):
    def __init__(self, tokenizer, args, data_path=None,function=None):
        self.data_path  = data_path
        self.function=function
        self.tokenizer = tokenizer
        self.tokens_list=[]
        self.args=args
        if data_path:
            self._prepare_data()     
        else:
            self._prepare_func()

    def _prepare_data(self):
        # check if the file .csv or .pkl to read it as csv or pickle file is the only available options
        if self.data_path.endswith('.csv'):
            data = pd.read_csv(self.data_path)
        else:
            data = pd.read_pickle(self.data_path)
        # get the functions and labels from the data as list    
        functions = data[self.args.function_column].tolist()
        labels = (data[self.args.target_column]*1).tolist()
        # loop over the functions and tokenize each function and add it to the tokens list
        for i in tqdm(range(len(functions))):
            tokens = self.tokenizer.encode_plus(self.remove_comments(functions[i]),
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.block_size,
                                            return_attention_mask = True)
            # check if the length of the tokens is equal to the block size
            # if the length is greater than the block size, then we will not add it to the list 
            if (tokens.attention_mask.flatten().sum().item()<self.args.block_size):
                self.tokens_list.append({'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(),
                                        'labels': torch.tensor(labels[i],dtype=torch.long)})
    def _prepare_func(self):
        """Prepare the function to be tokenized and added to the tokens list."""
        tokens = self.tokenizer.encode_plus(self.remove_comments(self.function),
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.args.block_size,
                                            return_attention_mask = True)
        self.tokens_list.append({'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten()})
        
    def remove_comments(self,text):
        """Delete comments from code."""

        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE,
        )
        return re.sub(pattern, replacer, text)
    
    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, index):
        return self.tokens_list[index]
    

class UCC_Data_Module(pl.LightningDataModule):
    """Data module for function-level classification. 
       It takes the data path and tokenizer and return the train, validation and test dataloaders."""
    
    def __init__(self, args,tokenizer):
        super().__init__()
        self.args=args
        self.tokenizer = tokenizer

    def setup(self, stage = None):
        """Setup the data module.
           It takes the stage which can be fit or test and return the corresponding dataset."""
        if stage in (None, "fit"):
            self.train_dataset = UCC_Dataset(data_path=self.args.train_data_file, tokenizer=self.tokenizer,args=self.args)
            self.args.max_steps= int((len(self.train_dataset) / self.args.train_batch_size) * self.args.epochs)
            self.val_dataset = UCC_Dataset(data_path=self.args.eval_data_file, tokenizer=self.tokenizer,args=self.args)
        if stage == 'test':
            self.test_dataset = UCC_Dataset(data_path=self.args.test_data_file, tokenizer=self.tokenizer,args=self.args)
            

    def train_dataloader(self):
        """ Return the train dataloader.
            number_workers is number of worker processes to use for data loading."""
        return DataLoader(self.train_dataset, batch_size = self.args.train_batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        """ Return the validation dataloader."""
        return DataLoader(self.val_dataset, batch_size = self.args.eval_batch_size, num_workers=4, shuffle=False)
    
    def test_dataloader(self):
        """ Return the test dataloader."""
        return DataLoader(self.test_dataset, batch_size = self.args.test_batch_size, num_workers=4, shuffle=False)

