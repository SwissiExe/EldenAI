import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_model_optimization as tfmot

class ModelTrainer:
    def __init__(self, data_processor):
        self.tokenizer = data_processor.tokenizer
        self.max_seq_length = data_processor.max_seq_length

    def create_model(self):
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=2000,
            end_step=20000
        )

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=200))  # Increase embedding dimension

        model.add(tf.keras.layers.GRU(256, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.GRU(256, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.GRU(256, return_sequences=True))

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(len(self.tokenizer.word_index) + 1, activation='softmax')
        ))

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                pruned_weights = tfmot.sparsity.keras.prune_low_magnitude(layer.get_weights()[0], pruning_schedule=pruning_schedule)
                layer.set_weights([pruned_weights] + layer.get_weights()[1:])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def preprocess_data(self, data_processor):
        question_seqs_padded = pad_sequences(self.tokenizer.texts_to_sequences(data_processor.data['Frage']), maxlen=self.max_seq_length, padding='post')
        answer_seqs_padded = pad_sequences(self.tokenizer.texts_to_sequences(data_processor.data['Antwort']), maxlen=self.max_seq_length, padding='post')
        return question_seqs_padded, answer_seqs_padded

    def train_model(self, model, data_processor):
        question_seqs_padded, answer_seqs_padded = self.preprocess_data(data_processor)
        dataset = tf.data.Dataset.from_tensor_slices((question_seqs_padded, answer_seqs_padded))
        dataset = dataset.shuffle(buffer_size=1024)

        val_size = int(0.2 * len(question_seqs_padded))
        val_dataset = dataset.take(val_size).batch(64).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        train_dataset = dataset.skip(val_size).batch(64).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        model.fit(train_dataset, epochs=180, validation_data=val_dataset)
        return model

    def save_model(self, model, model_path):
        model.save(model_path)
