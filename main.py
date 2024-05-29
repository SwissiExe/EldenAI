from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from chatbot import Chatbot


def train(data_processor):

    model_trainer = ModelTrainer(data_processor)
    model = model_trainer.create_model()
    model = model_trainer.train_model(model, data_processor)
    model_trainer.save_model(model, 'chatbot_model.h5')

def main():
    data_processor = DataProcessor('output.csv')

    #train(data_processor)  # <- Create Model
    chatbot = Chatbot('chatbot_model.h5', data_processor)
    chatbot.chat()

if __name__ == "__main__":
    main()
