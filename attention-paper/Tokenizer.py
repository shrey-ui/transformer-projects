import pandas as pd
import numpy as np 
import torch.nn as nn
import torch 
from collections import defaultdict
import re
from transformers import AutoTokenizer, BigBirdPegasusModel
from ast import literal_eval

def pretokenization_process(sentence_corpus):
    tokenizer= AutoTokenizer.from_pretrained('gpt2')
    corpus= ""
    for sent in sentence_corpus:
        corpus+= sent + " "
    pretoken_process= tokenizer.backend.pre_tokenizer.pre_tokenize_str(corpus)

    return pretoken_process
    
    

class BPETokenizer():
    def __init__(self, sentences_corpus, vocab_size, ):
        self.corpus= sentences_corpus
        self.merges_ct= vocab_size

        #self.words= pretokenization_process(self.sentences_corpus)
        self.pretokenizer= AutoTokenizer.from_pretrained('gpt2')
        self.word_count= BPETokenizer.get_word_count(self)
        self.base_vocab= [chr(i) for i in range(128)]
        self.german_chars= ["ẞ","ß", "ä", "ö", "ü", "Ä", "Ö", "Ü"]
        self.base_vocab.extend(self.german_chars)
        self.splits= {word : [c for c in word] for word in self.word_count.keys()}
        self.merges= {}

    def get_word_count(self):
        count_dict= defaultdict(int)
        for sent in self.corpus:
            word_offset= self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)
            new_word= [word for word, offset in word_offset]
            for word in new_word:
                count_dict[word]+= 1
        print("Init Count Dict: ", count_dict)
        return count_dict

    def pair_freqs(self):
        pairs= defaultdict(int)
        for word, freq in self.word_count.items():
            split= self.splits[word]
            if(len(split) ==1):
                continue
            for i in range(len(split)-1):
                pair= (split[i], split[i+1])
                pairs[pair]+= freq
        return pairs
    
    def merge_pair(self, a, b, splits):
        for word in self.word_count:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def train(self):
        
        while len(self.base_vocab) < self.merges_ct:
            pair_freqs = BPETokenizer.pair_freqs(self)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            self.splits = BPETokenizer.merge_pair(self, *best_pair, self.splits)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            print("Merge : ", best_pair)
            self.base_vocab.append(best_pair[0] + best_pair[1])
        
        #print("Vocab: ", self.base_vocab)


    def tokenize(self, text):
        pre_tokenized= self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_token_word = [word for word, offset in pre_tokenized]
        
        splits = [[l for l in word] for word in pre_token_word]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i =0
                while i < len(split) -1:
                    if(split[i] == pair[0] and split[i+1] == pair[1]):
                        split= split[:i] + [merge] + split[i+2:]
                    else: 
                        i+=1
                splits[idx] = split
        return sum(splits, [])


if __name__ == "__main__": 
    sentences= ["this is an test example", 'yes, this is an example indeed', 
                'what is this example for?' , "this exmap is for testing the tokenizer"]
    
    df_test= pd.read_csv('./dataset/wmt14/dataset_init_test.csv', encoding= 'utf-8')
    english_sentences= []
    german_sentences= []
    print(df_test['translation'][0])
    for ind, rows in df_test.iterrows():
        print(literal_eval(rows['translation'])['de'])
        english_sentences.append(literal_eval(rows['translation'])['en'])
        german_sentences.append(literal_eval(rows['translation'])['de'])
    

    english_sentences.extend(german_sentences)
    #print(english_sentences[:5])
    tokenizer= BPETokenizer(english_sentences, 1024)
    
    tokenizer.train()
    #print(tokenizer.base_vocab)
    print(tokenizer.tokenize("this is an example to test"))
    for sent in german_sentences[:10]:
        print(tokenizer.tokenize(sent))


