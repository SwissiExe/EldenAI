from tf_keras.src.utils import pad_sequences
import tensorflow as tf

class Chatbot:
    def __init__(self, model_path, data_processor):
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = data_processor.tokenizer
        self.max_seq_length = data_processor.max_seq_length

    def chat(self):
        print("Willkommen zum Chatbot! Du kannst Fragen stellen und Antworten erhalten. Um den Chat zu beenden, tippe 'exit'.")
        while True:
            user_input = input("Du: ")
            if user_input.lower() == 'exit':
                break
            user_input_seq = self.tokenizer.texts_to_sequences([user_input])
            user_input_padded = pad_sequences(user_input_seq, maxlen=self.max_seq_length, padding='post')

            predicted_seq = self.model.predict(user_input_padded)[0]
            predicted_answer = self.generate_answer(predicted_seq)

            print(f"Chatbot: {predicted_answer}")

    def generate_answer(self, predicted_seq):
        answer = ''
        for token_index in predicted_seq:
            predicted_word = self.tokenizer.index_word.get(tf.argmax(token_index).numpy(), '')
            if predicted_word == 'endseq':
                break
            answer += predicted_word + ' '

        if answer.strip() == '':
            return "Entschuldigung, ich habe dich nicht verstanden."
        else:
            return answer
