import torch
import torch.nn as nn
import torch.nn.functional as F
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__() 
        
        # Self-Attention
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(2*hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.W1(matching_inputs)
        x = self.W2(x)
        x = self.W3(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.compress = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, self.hidden_size) 
        output, hidden_state = self.gru(self.dropout(input))

        return output, hidden_state
   
        
        
        
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_dim, helper=None, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.hidden_size, self.vocab_size, self.embed_dim, self.helper = hidden_size, vocab_size, embed_dim, helper

        # layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size+embed_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.linear = nn.Linear(hidden_size,vocab_size)


    def forward(self, encoder_hidden=None, encoder_output=None, targets=None, mode=None, teacher_ratio=0.7):
        batch_size, _, _ = encoder_output.size()
        hidden_state = (self.init_state(batch_size).unsqueeze(0)).to(device)
        seq_logProb = []
        caption_preds = []
        # implement schedule sampling
        targets = self.embedding(targets) # (batch, max_seq_len, embedding_size) embeddings of target labels of ground truth sentences
        _, seq_len, _ = targets.size()
    
        embed = targets[:, 0]           
        for i in range(seq_len-1):  
            context = self.attention(hidden_state, encoder_output)
            gru_input = torch.cat([embed, context], dim=1).unsqueeze(1)
            gru_output, hidden_state = self.gru(gru_input, hidden_state)
            logprob = self.linear(self.dropout(gru_output.squeeze(1)))
            seq_logProb.append(logprob.unsqueeze(1))
            
            use_teacher_forcing = True if random.random() < teacher_ratio else False
            if use_teacher_forcing:
                embed = targets[:, i+1]
            else:
                decoder_input = logprob.unsqueeze(1).max(2)[1]
                embed = self.embedding(decoder_input).squeeze(1)

        # after calculating all word prob, concatenate seq_logProb into dim(batch, seq_len, output_size)
        seq_logProb = torch.cat(seq_logProb, dim=1)
        caption_preds = seq_logProb.max(2)[1]
        return seq_logProb, caption_preds

    
    def inference(self, encoder_hidden, encoder_output, vocab):
        batch_size, _, _ = encoder_output.size()
        hidden_state = (self.init_state(batch_size).unsqueeze(0)).to(device)
        decoder_input = torch.tensor(1).view(1,-1).to(device)
        seq_logProb = []
        caption_preds = []
        max_seq_len = 30
        
        for i in range(max_seq_len-1):
            embed = self.embedding(decoder_input).squeeze(1)
            context = self.attention(hidden_state, encoder_output)
            gru_input = torch.cat([embed, context], dim=1).unsqueeze(1)
            gru_output, hidden_state = self.gru(gru_input, hidden_state)
            logprob = self.linear(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_input = logprob.unsqueeze(1).max(2)[1]
            
            if vocab.itow[decoder_input.item()] == "<EOS>":
                break

        seq_logProb = torch.cat(seq_logProb, dim=1)
        caption_preds = seq_logProb.max(2)[1]
        return seq_logProb, caption_preds
    
        
    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))
    
    
class S2VTMODEL(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, embed_dim):
        super(S2VTMODEL, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, vocab_size, embed_dim)


    def forward(self, features, target_captions=None, mode=None):
        encoder_outputs, encoder_hidden = self.encoder(features)
        if mode == 'train':
            seq_logProb, caption_preds = self.decoder(encoder_hidden=encoder_hidden, encoder_output=encoder_outputs, targets=target_captions, mode=mode)
        elif mode == 'test':
            seq_logProb, caption_preds = self.decoder.inference(encoder_hidden=encoder_hidden, encoder_output=encoder_outputs, vocab=None)
        else:
            raise KeyError('mode is not valid')
        return seq_logProb, caption_preds
    
