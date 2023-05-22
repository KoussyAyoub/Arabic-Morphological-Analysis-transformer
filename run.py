from flask import Flask, request, render_template
import torch
import torch.nn as nn
import re
import os
import numpy as np
import string
import torch.optim as optim
import random
import ast
import time
from bs4 import BeautifulSoup

from flask_cors import CORS

app = Flask(__name__)

CORS(app)

class model(nn.Module): 

    def __init__(self, data, batch_size ,embedding_size, hidden_size,num_layers ,dropout, teacher_forcing_ratio, learning_rate):
        super().__init__()
        
        self.sow = '$'
        self.eow = '£'
        self.lr = learning_rate
        self.ratio = 0.9
        self.batch_size = batch_size
        self.data = data
        self.batches, self.vocab, self.char_index_dic = self.prepare_data(self.data)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_dense = nn.Linear(self.hidden_size * 2,self.embedding_size)
        
        self.num_layers = num_layers
        
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings = len(self.vocab), embedding_dim = self.embedding_size, padding_idx = self.char_index_dic['%']) 
        self.Dropout = nn.Dropout(self.dropout / 2)
        self.BILSTM = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout = self.dropout)
        self.LSTM = nn.LSTM(input_size= self.embedding_size ,hidden_size = self.hidden_size*2, num_layers = self.num_layers, batch_first = True , dropout = self.dropout)
                
        self.criterion = nn.CrossEntropyLoss(ignore_index =self.char_index_dic['%'])
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.Linear = nn.Linear(self.hidden_size * 2,len(self.vocab)) 
        self.opt1 = optim.Adam(self.BILSTM.parameters(), lr = self.lr )
        self.opt2 = optim.Adam([*self.LSTM.parameters(), *self.input_dense.parameters()], lr = self.lr)
    
    
    def prepare_data(self, data):
        pad_char = '%'
        padded_data = []
        ls_words = []
        ls_roots = []
        for instance in data : 
            ls_words.append(instance[0])
            ls_roots.append(instance[1])
        max_len_words = max([len(item) for item in ls_words])
        max_len_roots = max([len(item) for item in ls_roots])
        for instance in data: 
            tmp = []
            word,root = instance[0], instance[1]
            while(len(word) != max_len_words):
                word += pad_char
            tmp.append(word)
            while(len(root) != max_len_roots):
                root += pad_char
            tmp.append(root)
            padded_data.append(tmp)

        vocab = []
        for word in padded_data :
            for item in word : 
                tmp = set(item)
                for k in tmp : 
                    if k not in vocab : 
                        vocab.append(k)

        char_to_idx_map = {char: idx for idx, char in enumerate(vocab)}
        final_data = []
        for instance in padded_data : 
            tmp = []
            word = [char_to_idx_map[char] for char in instance[0]]
            root = [char_to_idx_map[char] for char in instance[1]]
            tmp.append(word)
            tmp.append(root)
            final_data.append(tmp)

        size= self.batch_size 
        batches = [final_data[i:i + size] for i in range(0, len(final_data), size)]
        return batches , vocab , char_to_idx_map
    
    
    def word_to_seq(self, word):
        word_char_idx_seq =[self.char_index_dic[char] for char in word]    
        return word_char_idx_seq # word sequence
        
    def encode(self, batch):    
        word_batch = [] # list of words in the batch
        root_batch = [] # list of roots in the batch
        
        for instance in batch : 
            word_batch.append(instance[0])
            root_batch.append(instance[1])
            
        word_batch = torch.tensor(word_batch)
        root_batch = torch.tensor(root_batch)        
        embedded_word_batch = self.embedding(word_batch)
        init_hid = nn.init.xavier_normal_(torch.zeros(2*self.num_layers, len(batch), self.hidden_size), gain=0.5)
        init_ce = nn.init.xavier_normal_(torch.zeros(2*self.num_layers, len(batch), self.hidden_size), gain=0.5)
        outputs, (hidden, cell) = self.BILSTM(embedded_word_batch, (init_hid, init_ce)) # we pass the emebedded vector through the bi-GRU 
        final_hid, final_ce = [], []
        for k in range(0,hidden.size(0), 2):
            tmp_hid = hidden[k:k+2 , :, :]
            tmp_ce = cell[k:k+2, :, :]
            cct_hid = torch.cat((tmp_hid[0], tmp_hid[1]), dim  = 1).tolist()
            cct_ce = torch.cat((tmp_ce[0], tmp_ce[1]), dim  = 1).tolist()
            final_hid.append(cct_hid)
            final_ce.append(cct_ce)
        final_hid, final_ce = torch.tensor(final_hid), torch.tensor(final_ce)
        return root_batch , outputs ,(final_hid, final_ce)

    def decode(self, encoder_outputs ,encoder_hidden_cell , batch, teacher_forcing_bool, epoch):
        (hidden_layer , cell) , root_batch = encoder_hidden_cell , batch 
        embedded_char = self.embedding(torch.unsqueeze(root_batch[:, 0], 1))
        outputs = []
        for i in range(root_batch.size(1)): 
            self.Dropout(embedded_char)
            decoder_output , (hidden_layer, cell) = self.LSTM(embedded_char, (hidden_layer, cell))
            input_decoder_output = self.input_dense(decoder_output)
            embedded_char = input_decoder_output
            mask = np.where([random.random() <= (self.teacher_forcing_ratio) for i in range(root_batch.size(0))])[0]
            teacher_forcing_input = self.embedding(torch.unsqueeze(torch.clone(root_batch[:, i]), 1))
            if teacher_forcing_bool : 
                embedded_char[mask] = teacher_forcing_input[mask] 
            Dense_decoded_output = self.Linear(decoder_output)
            soft = nn.Softmax(dim = 2)
            soft_out = soft(Dense_decoded_output)
            outputs.append(soft_out)
            
        return outputs 
                            
    def train_model(self, batches, teacher_forcing_bool, epoch):
        train_batches = batches        
        epoch_loss = 0
        n = 0            
        test_word = '$' + 'تحليل' + '£'
        for batch in train_batches :
            #print(self.predict(test_word))
            self.opt1.zero_grad()
            self.opt2.zero_grad()
            root_batch, encoder_output, encoder_states = self.encode(batch)
            outputs = self.decode(encoder_output,encoder_states, root_batch, teacher_forcing_bool, epoch)
            a = [torch.squeeze(item, 1) for item in outputs]
            a = [torch.unsqueeze(item, 0) for item in a]
            output = torch.cat(a, dim = 0)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = root_batch.transpose(0, 1)
            trg = trg.reshape(-1)
            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*self.LSTM.parameters(), *self.BILSTM.parameters()], 1)
            self.opt1.step()
            self.opt2.step()
            #self.optimizer.step()
            epoch_loss+=loss.item()
            n+=1
            print('the loss of the train batch ', n ,' is : ', loss.item())
        return epoch_loss/n

    def evaluate_model(self, batches, teacher_forcing_bool, epoch):
        self.eval()
        val_batches = batches
        n = 0
        epoch_loss = 0
        with torch.no_grad() :
            for batch in val_batches :
                root_batch, encoder_output ,encoder_states = self.encode(batch)
                outputs = self.decode(encoder_output ,encoder_states, root_batch, teacher_forcing_bool, epoch)
                a = [torch.squeeze(item, 1) for item in outputs]
                a = [torch.unsqueeze(item, 0) for item in a]
                output = torch.cat(a, dim = 0)
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)
                trg = root_batch.transpose(0, 1)
                trg = trg.reshape(-1)
                loss = self.criterion(output, trg)
                epoch_loss+=loss.item()
                n+=1
                print('the loss of the val batch ', n ,' is : ', loss.item())
        return epoch_loss / n
    
    def predict(self, word):
        word_seq = self.word_to_seq(word)
        embedded_word = self.embedding(torch.tensor(word_seq))
        init_hid = nn.init.xavier_normal_(torch.zeros(2*self.num_layers, self.hidden_size), gain=0.5)
        init_ce = nn.init.xavier_normal_(torch.zeros(2*self.num_layers, self.hidden_size), gain=0.5)
        outputs, (hidden, cell) = self.BILSTM(embedded_word, (init_hid, init_ce))
        final_hid, final_ce = [], []
        for k in range(0,hidden.size(0), 2):
            tmp_hid = hidden[k:k+2 ,:]
            tmp_ce = cell[k:k+2, :]
            cct_hid = torch.cat((tmp_hid[0], tmp_hid[1]), dim  = -1).tolist()
            cct_ce = torch.cat((tmp_ce[0], tmp_ce[1]), dim  = -1).tolist()
            final_hid.append(cct_hid)
            final_ce.append(cct_ce)
        final_hidden, final_cell = torch.tensor(final_hid), torch.tensor(final_ce)
        embedded_char = torch.unsqueeze(self.embedding(torch.tensor(self.char_index_dic[self.sow])), 0)
        prediction_output = [] # a list of the outputs of the decoder 
        soft = nn.Softmax(dim = 1)
        key_list = list(self.char_index_dic.keys())
        val_list = list(self.char_index_dic.values())
        for i in range(5):
            decoder_output , (final_hidden, final_cell) = self.LSTM(embedded_char, (final_hidden, final_cell))
            input_dense = nn.Linear(self.hidden_size * 2,self.embedding_size)
            input_decoder_output = input_dense(decoder_output)
            embedded_char = input_decoder_output
            Dense_decoded_output = self.Linear(decoder_output)
            prediction_output.append(soft(Dense_decoded_output).tolist())
        prediction_output = torch.squeeze(torch.tensor(prediction_output), 1)
        test_word_seq = word_seq[1:]
        test_word_seq = test_word_seq[:-1]
        precision = 5
        top_idx = torch.topk(prediction_output, precision, dim = 1).indices
        init_char = 0
        final_char = 0        
        if self.char_index_dic[self.sow] in top_idx[0] : 
            init_char = self.char_index_dic[self.sow]
        else : 
            init_char = top_idx[0][0]
        if self.char_index_dic[self.eow] in top_idx[-1] : 
            final_char = self.char_index_dic[self.eow]
        else : 
            final_char = top_idx[-1][0]
        grid = []
        for i in range(precision): 
            for j in range(precision):
                for k in range(precision):
                    tmp = []
                    tmp.append((top_idx[1][i]).item())
                    tmp.append((top_idx[2][j]).item())
                    tmp.append((top_idx[3][k]).item())
                    grid.append(tmp)
        best_cases = []
        print(grid)
        for case in grid : 
            s = [item for item in case if item in set(test_word_seq)] # we select elts from a that are in l 
            b = [item for item in test_word_seq if item in set(s)] # 
            if s == b and s != [] : 
                best_cases.append(case)
        pot_seq = []
        for item in best_cases : 
            tmp = [init_char] + item  + [final_char]
            pot_seq.append(tmp)           
        final_roots =[]
        for seq in pot_seq : 
            position = [val_list.index(item) for item in seq]
            result_char = [key_list[pos] for pos in position]
            predicted_root = ''.join(result_char)
            final_roots.append(predicted_root)
        return final_roots

    
    def fit(self, num_epochs):
        print(f'The model has {self.count_parameters():,} trainable parameters')
        data = self.data
        data = random.sample(data, len(data))
        data_size = len(data)
        middle_index = int(data_size * self.ratio)        
        train_data , val_data = data[:middle_index], data[middle_index:]
        train_batches, voc, dic = self.prepare_data(train_data)
        val_batches ,voc , dic = self.prepare_data(val_data)
        epochs = list(range(num_epochs))
        best_val_loss = 1000
        best_model_par = 0
        losses =[]
        predicted_roots = []
        test_word = '$' + 'تحليل' + '£'
        for epoch in epochs : 
            print('epoch num : ', epoch) 
            print(self.char_index_dic)
            t1 = time.time()
            train_batches = random.sample(train_batches , len(train_batches))
            train_loss= self.train_model(train_batches, 1, epoch)
            val_loss = self.evaluate_model(val_batches, 0, epoch) # we set the teacher forcing to false            
            t2 = time.time()
            predicted_root = self.predict(test_word)
            print(predicted_root)
            predicted_roots.append(predicted_root)
            tmp = [train_loss, val_loss]
            losses.append(tmp)
            print('the training loss : ', train_loss , 'the val loss :', val_loss)
            print('epoch num : ' ,epoch , ' lasted : ', t2 - t1 , 'seconds')
            if val_loss < best_val_loss :
                best_val_loss = val_loss 
                best_model_par = self.state_dict()
        torch.save(best_model_par, 'best_model.pt')
            
        return losses
    
    def count_parameters(self):
        return sum(torch.numel(p) for p in self.parameters() if p.requires_grad)

directory = 'corpus_morphological_analysis'
file_paths = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    file_paths.append(f)   
temp_file_paths = file_paths[:4000] # we take the first 5000 files out of 29000 files 
def identify(word_l):
    if len(word_l) < 4 :
        return None
    dictt = {}
    dictt['word'] = word_l[0]
    if word_l[2] != '' and word_l[2] != ' ' :   
        if word_l[4] not in word_l[0]: 
            if word_l[5] in word_l[0] and word_l[5] != '': 
                dictt['prefixe'] = word_l[2]
                dictt['root'] = word_l[8]
                dictt['suffixe'] = word_l[9]
            elif word_l[3] in word_l[0] and word_l[3] != '':
                dictt['prefixe'] = word_l[2]
                dictt['root'] = word_l[3]
                dictt['suffixe'] = ''
        else :
            dictt['prefixe'] = word_l[2]
            dictt['root'] = word_l[7]
            dictt['suffixe'] = word_l[8]
    else : 
        if word_l[2] == '' : 
            dictt['prefixe'] = word_l[2]
            dictt['root'] = word_l[6]
            dictt['suffixe'] = word_l[7]
        elif  word_l[2] == ' ' :
            dictt['prefixe'] = ''
            dictt['root'] = word_l[3]
            dictt['suffixe'] = ''    
    return dictt
content = []
for filepath in temp_file_paths :
    with open(filepath, encoding='utf-8') as f :
        html = f.read()
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()  
    text = soup.get_text()
    content.append(text)
split_list = []
for item in content : 
    tmp = item.splitlines()
    split_list.append(tmp)
work_list = []
for k in split_list :
    l = [item for item in k if 'لا توجد نتائج لتحليل هذه الكلمة' not in item]
    tmp_l = [item.replace("#",'') for item in l]
    work_list.append(tmp_l)
final_list = []
for k in work_list :
    tst = [item.split(':') for item in k]
    final_list.append(tst)

def word_to_dict_list(wordlist):
    dictlist = []
    for k in wordlist : 
        dictlist.append(identify(k))
    return dictlist
final = []
for k in final_list: 
    for j in k :
        s = identify(j)
        if s == None :
            continue
        final.append(identify(j))

def dic_to_list(listt):
    L = []
    for k in listt : 
        tmp = []
        #print(k)
        if len(k) == 4 : 
            tmp.append(k['word'])
            tmp.append(k['prefixe'])
            tmp.append(k['root'])
            tmp.append(k['suffixe'])
            L.append(tmp)
    return L
data = dic_to_list(final)
final_l = dic_to_list(final)
root_data = []
for word in data : 
    tmp =[]
    tmp.append(word[0])
    tmp.append(word[2])
    root_data.append(tmp)
data_root = []
for item in root_data : 
    tmp = []
    if len(item[1]) <= 3 and len(item[1]) != 0:
        tmp.append('$'+item[0]+'£')
        tmp.append('$'+item[1]+'£')
        data_root.append(tmp)

print(len(data_root))
for item in data_root :
    if len(item[0])==15 or len(item[0])==16:
        data_root.pop(data_root.index(item))
print(len(data_root))
d = []
for item in data_root:
    if len(item[0]) > 4 :
        d.append(item)
print(len(d))

test_model = model(d, 512, 64 , 100 , 3 , 0.2 ,0.35, 0.0005)
test_model.load_state_dict(torch.load('best_model.pt'), strict=False)

def clean_arabic(text):
    """
    This function takes an input string and removes any non-Arabic characters
    from it, returning the cleaned string.
    """
    arabic_chars = 'ابتةثجحخدذرزسشصضطظعغفقكلمنهويءئؤ'
    clean_text = ''.join(c for c in text if c in arabic_chars)
    return clean_text

def clean_data(data):
    """
    This function takes a list of strings, cleans each string using the `clean_arabic` function,
    and returns a single string with all the cleaned strings concatenated together with spaces
    between them.
    """
    cleaned_strings = [clean_arabic(s) for s in data]
    return ' '.join(cleaned_strings)

from flask import jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        texte1 = request.form['texte1']
        word = '$' + texte1 + '£'
        output_text = test_model.predict(word)
        output_text = output_text[:4]
        output_text = clean_data(output_text)
        texte2 = "{}".format(output_text)
    else:
        texte1 = ""
        texte2 = ""
    response = {"results": texte2}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
