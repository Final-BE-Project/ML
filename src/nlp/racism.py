import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = ""
max_seq_len = 25

# specify GPU
device = torch.device("cuda")

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert 
      self.dropout = nn.Dropout(0.1)
      self.relu =  nn.ReLU()
      self.fc1 = nn.Linear(768,512)
      self.fc2 = nn.Linear(512,2)
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)
      return x

def init_model():
    # pass the pre-trained BERT to our define architecture
    global model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)
    # push the model to GPU
    model = model.to(device)

    model_save_name = 'saved_weights.pt'
    model_path = f"src/models/{model_save_name}"
    model.load_state_dict(torch.load(model_path))

def run_input(sentences):
    testing_sentences = tokenizer.batch_encode_plus(
    sentences,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
    )

    # for test set
    testing_sentences_seq = torch.tensor(testing_sentences['input_ids'])
    testing_sentences_mask = torch.tensor(testing_sentences['attention_mask'])
    # testing_sentences_y = torch.tensor(test_labels.tolist())
    

    with torch.no_grad():
        preds_new = model(testing_sentences_seq.to(device), testing_sentences_mask.to(device))
        preds_new = preds_new.detach().cpu().numpy()

    preds_new = np.argmax(preds_new, axis = 1)
    print(preds_new)
    # testing_sentences_labels = [1, 0, 1, 0]

def detect_racism(sentence):
    output_ = run_input([sentence])
    print(output_)
    return output_

if __name__ == "__main__":
    init_model()
    # run_input(["Fuck you chinese nigger", "Hello earth. How are you.", "Nigger", "african"])
    detect_racism("Fuck you chinese nigger")
