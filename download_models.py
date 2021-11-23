#!/usr/bin/env python

import os

import spacy_udpipe
import stanza
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if __name__ == '__main__':
    # Download QG/QA model adn tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qa-qg-hl")
    model.save_pretrained("./models/qa_qg/t5-base-qa-qg-hl")
    tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qa-qg-hl")
    tokenizer.save_pretrained("./models/qa_qg/t5-base-qa-qg-hl")
    # Download taggers
    stanza.download('en', "./models/stanza")
    spacy_udpipe.download("en")
