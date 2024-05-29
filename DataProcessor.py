import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.tokenizer, self.max_seq_length = self.tokenize_data()

    def load_data(self, file_path):
        return pd.read_csv(file_path, sep=',')

    def tokenize_data(self):
        self.data.fillna('', inplace=True)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.data['Frage'] + self.data['Antwort'])
        question_seqs = tokenizer.texts_to_sequences(self.data['Frage'])
        answer_seqs = tokenizer.texts_to_sequences(self.data['Antwort'])
        max_seq_length = max(max(len(seq) for seq in question_seqs), max(len(seq) for seq in answer_seqs))
        return tokenizer, max_seq_length
