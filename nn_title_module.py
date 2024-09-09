import torch
from encoder_module import Encoder

# Модель для классификации предложений
class NNModule:
    def __init__(self):
        self.dict = {0: "Жилищное строительство",
          1: "Социальные объекты",
          2: "Транспортная инфраструктура",
          3: "Финансовые показатели",
          4: "Перспективы и планы на будущее"}
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=768, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=5)
        )
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=768, out_features=64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=64, out_features=64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=64, out_features=5)
        #     )
        self.model.load_state_dict(torch.load("ml/clf_model.pth", map_location=torch.device('cpu'), weights_only=True))
        self.model.eval()
        self.encoder = Encoder()
    
    def predict(self, text):
        embedding = self.encoder.encode(text)
        logits = self.model(embedding)
        pred = torch.softmax(logits, dim=1).argmax(dim=1)
        return self.dict[pred[0].item()]
    
    
    def many_predict(self, texts):
        embeddings = self.encoder.encode(texts)
        logits = self.model(embeddings)
        pred = torch.softmax(logits, dim=1).argmax(dim=1)
        response = []
        for tensor in pred:
            response.append(self.dict[tensor.item()])
        return response
