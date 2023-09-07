import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig, EncoderDecoderModel


class Seq2SeqTransformers:
    def __init__(self, max_sequence_length, tokenizer_path, model_path):
        self.max_sequence_length = max_sequence_length
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = EncoderDecoderModel.from_pretrained(model_path)

    def preprocess_text(self, text):
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='pt'
        )
        return inputs

    def predict(self, input_text):
        input_data = self.preprocess_text(input_text)
        input_ids = input_data['input_ids']
        attention_mask = input_data['attention_mask']

        # Generate fake review detection output
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_length=self.max_sequence_length)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return decoded_output

