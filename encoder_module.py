import torch
from transformers import AutoTokenizer, AutoModel

#LaBSE энкодер текста
class Encoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        self.model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

    def encode(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output.pooler_output
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings
