import itertools
from typing import Optional, Dict, Union
import sys
import os

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# Add the question_generation package
sys.path.append(os.path.join(os.path.dirname(__file__), "../question_generation"))
from pipelines import QGPipeline, MultiTaskQAQGPipeline
from mteqa.utils import split_list


class MTEQGPipeline(QGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs: str, _self_answers=None):
        if _self_answers is None:
            super().__call__(inputs)
        else:
            inputs = " ".join(inputs.split())
            sents, _ = self._extract_answers(inputs)
            answers = _self_answers
            flat_answers = list(itertools.chain(*answers))

            if len(flat_answers) == 0:
                return []

            if self.qg_format == "prepend":
                qg_examples = self._prepare_inputs_for_qg_from_answers_prepend(inputs, answers)
            else:
                qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)

            qg_inputs = [example['source_text'] for example in qg_examples]
            questions = self._generate_questions(qg_inputs)
            output = [{'answer': example['answer'], 'question': que} for example, que in zip(qg_examples, questions)]
            return output

    def _generate_questions(self, inputs):
        split_input = split_list(inputs, 4)
        _questions = []
        for _inputs in split_input:
            questions = super()._generate_questions(_inputs)
            _questions.extend(questions)
        return _questions


class MTEQAMultiTaskQAQGPipeline(MTEQGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs: Union[Dict, str], _self_answers=None):
        if type(inputs) is str:
            # do qg
            return super().__call__(inputs, _self_answers)
        else:
            # do qa
            return self._extract_answer(inputs["question"], inputs["context"])

    def _prepare_inputs_for_qa(self, question, context):
        source_text = f"question: {question}  context: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        return source_text

    def _extract_answer(self, question, context):
        source_text = self._prepare_inputs_for_qa(question, context)
        inputs = self._tokenize([source_text], padding=False)

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            max_length=16,
        )

        answer = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        return answer


def mteqa_pipeline(
    model: Optional[str] = None,
    use_cuda: Optional[bool] = True
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    return MTEQAMultiTaskQAQGPipeline(
        model=model,
        tokenizer=tokenizer,
        ans_model=model, ans_tokenizer=tokenizer,
        qg_format="highlight", use_cuda=use_cuda)
