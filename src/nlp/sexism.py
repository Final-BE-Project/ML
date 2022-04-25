import pandas as pd
import numpy as np
import math as m
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
import os
import math
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from transformers import XLNetTokenizerFast

tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased', do_lower_case=True)

nltk.download('stopwords')
nltk.download ('punkt')
nltk.download('wordnet')

model = ""

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
  
  def __init__(self, num_labels=2):
    super(XLNetForMultiLabelSequenceClassification, self).__init__()
    self.num_labels = num_labels
    self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
    self.classifier = torch.nn.Linear(768, num_labels)

    torch.nn.init.xavier_normal_(self.classifier.weight)

  def forward(self, input_ids, token_type_ids=None,\
              attention_mask=None, labels=None):
    # last hidden layer
    last_hidden_state = self.xlnet(input_ids=input_ids,\
                                   attention_mask=attention_mask,\
                                   token_type_ids=token_type_ids)
    # pool the outputs into a mean vector
    mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
    logits = self.classifier(mean_last_hidden_state)
        
    if labels is not None:
      loss_fct = BCEWithLogitsLoss()
      loss = loss_fct(logits.view(-1, self.num_labels),\
                      labels.view(-1, self.num_labels))
      return loss
    else:
      return logits
    
  def freeze_xlnet_decoder(self):
    """
    Freeze XLNet weight parameters. They will not be updated during training.
    """
    for param in self.xlnet.parameters():
      param.requires_grad = False
    
  def unfreeze_xlnet_decoder(self):
    """
    Unfreeze XLNet weight parameters. They will be updated during training.
    """
    for param in self.xlnet.parameters():
      param.requires_grad = True
    
  def pool_hidden_state(self, last_hidden_state):
    """
    Pool the output vectors into a single mean vector 
    """
    last_hidden_state = last_hidden_state[0]
    mean_last_hidden_state = torch.mean(last_hidden_state, 1)
    return mean_last_hidden_state
    

def load_model(save_path):
  """
  Load the model from the path directory provided
  """
  checkpoint = torch.load(save_path)
  model_state_dict = checkpoint['state_dict']
  model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
  model.load_state_dict(model_state_dict)

  epochs = checkpoint["epochs"]
  lowest_eval_loss = checkpoint["lowest_eval_loss"]
  train_loss_hist = checkpoint["train_loss_hist"]
  valid_loss_hist = checkpoint["valid_loss_hist"]
  
  return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist  

def generate_predictions(model, df, device="cpu", batch_size=16):
  num_iter = math.ceil(df.shape[0]/batch_size)
  
  pred_probs = []

  #model.to(device)
  #model.eval()
  
  for i in range(num_iter):
    df_subset = df.iloc[i*batch_size:(i+1)*batch_size,:]
    X = df_subset["features"].values.tolist()
    masks = df_subset["masks"].values.tolist()
    X = torch.tensor(X)
    masks = torch.tensor(masks, dtype=torch.long)
    X = X.to(device)
    masks = masks.to(device)
    with torch.no_grad():
      logits = model(input_ids=X, attention_mask=masks)
      logits = logits.sigmoid().detach().cpu().numpy()
#       pred_probs = np.vstack([pred_probs, logits])
      pred_probs.extend(logits.tolist())
        
  return pred_probs

def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids

def create_attn_masks(input_ids):
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


def init_model():
    device = torch.device("cuda")
    model_save_name = 'detection.pt'
    path = F"src/models/{model_save_name}"
    global model

    model = XLNetForMultiLabelSequenceClassification()
    model.load_state_dict(torch.load(path))
    model.to(device)

def take_input(sentence):
    valid = pd.DataFrame()
    valid['HATESPEECH'] = sentence
    validid = tokenize_inputs(sentence, tokenizer, num_embeddings=250)
    attention_masks = create_attn_masks(validid)
    valid["features"] = validid.tolist()
    valid["masks"] = attention_masks
    num_labels = 2
    pred_probs = generate_predictions(model, valid, device="cuda", batch_size=16)

    u = list(pred_probs)
    v = []
    vv = []
    for i in u:
        if i[0] < 0.40 and i[1]<0.40:
            v.append(0)
            vv.append("Clear")
        elif i[0]>i[1]:
            v.append(round(i[0],4))
            vv.append("LGBT+")
        else:
            v.append(round(i[1],4))
            vv.append("WOMEN+")
    
    uu = valid['HATESPEECH']
    ans_frame = pd.DataFrame()
    ans_frame['text'] = uu
    ans_frame['tag'] = vv
    ans_frame['probability'] = v

    print(ans_frame)
    return ans_frame


def detect_sexism(sentence):
    init_model()
    answer_df = take_input([sentence])
    print(answer_df["probability"].iloc[0])
    return answer_df["probability"].iloc[0], answer_df["tag"].iloc[0]

if __name__ == "__main__":
    input_ = ["you are a fucking slut! Go suck some dick!","faggots in this webspace are over-smart"]
    init_model()
    answer_df = take_input(input_)
    print(answer_df["probability"].iloc[1])
    
