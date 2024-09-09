import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Модуль суммаризации текста
class Summarizer:
    def __init__(self):
        model = "basic-go/FRED-T5-large-habr-summarizer"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)

    def __preprocess_text(self, text):
        clean_expr = re.compile(r"[\xa0\x1a\x16\x1b\x17\x15\u2004]")
        spaces_expr = re.compile(r"\s{2,}")
        text = clean_expr.sub(" ", text)
        text = spaces_expr.sub(" ", text)

        if "." in text:
            index = text.rindex(".")
            text = text[:index + 1]

        return text

    def __postprocess_text(self, text):
        clean_expr = re.compile(r"[\xa0\x1a\x16\x1b\x17\x15\u2004]")
        spaces_expr = re.compile(r"\s{2,}")
        text = clean_expr.sub(" ", text)
        text = spaces_expr.sub(" ", text)

        if "." in text:
            index = text.rindex(".")
            text = text[:index + 1]
            
        return text

    def summary(self, text):
        text = self.__preprocess_text(text)

        input_ids = torch.tensor([self.tokenizer.encode(text)])

        outputs = self.model.generate(
            input_ids, 
            max_new_tokens=420,
            num_beams=2, 
            do_sample=True,
            top_k=100,
            repetition_penalty=2.5, 
            length_penalty=1.0,
            early_stopping = True)
        
        summary = self.__postprocess_text(self.tokenizer.decode(outputs[0][1:]))

        return summary