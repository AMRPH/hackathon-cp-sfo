from summarizer_module import Summarizer
from processing_module import Processing
from nn_title_module import NNModule
import pandas as pd

#Пайплан работы всех моделей, суммаризация и разбиение по темам
class Pipeline:
    def __init__(self):
        self.summarizer = Summarizer()
        self.processing = Processing()
        self.model = NNModule()
    
    def work(self, text):
        summary = self.summarizer.summary(text)
        sentences = self.processing.split_by_sentences(summary)
        titles= self.model.many_predict(sentences)
        df = pd.DataFrame({'titles': titles, 'sentences': sentences})
        title_list = ["Жилищное строительство","Социальные объекты", "Транспортная инфраструктура", "Финансовые показатели", "Перспективы и планы на будущее"]

        dict = {}
        for title in title_list:
            if (df[df['titles'] == title]['sentences'].size > 0):
                dict[title] = df[df['titles'] == title]['sentences'].to_list()
        return dict