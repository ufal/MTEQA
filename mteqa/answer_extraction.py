import json

from nltk.tokenize import sent_tokenize
import numpy as np
import spacy_udpipe
import stanza

PATTERNS = [['NOUN'], ['NUM'], ['DET', 'NOUN'], ['NUM', 'NOUN'],
            ['NUM', 'NOUN', 'NOUN'], ['NOUN', 'NOUN'],
            ['DET', 'NOUN', 'NOUN'], ['ADJ', 'NOUN'],
            ['DET', 'ADJ', 'NOUN'], ['DET', 'ADJ', 'ADJ', 'NOUN'],
            ['DET', 'NOUN', 'PUNCT', 'NOUN'],
            ['DET', 'VERB', 'NOUN'],
            ['NUM', 'PUNCT', 'NUM'],
            ['DET', 'ADJ', 'NOUN', 'NOUN'],
            ['DET', 'ADV', 'VERB', 'NOUN'],
            ['DET', 'ADV', 'VERB', 'ADJ', 'NOUN'],
            ['DET', 'ADV', 'ADJ', 'NOUN'], ['DET', 'NOUN', 'ADP', 'NOUN'],
            ['DET', 'ADJ', 'CCONJ', 'ADJ', 'NOUN']]


def parse_stanza(sentence):
    nlp_stanza = stanza.Pipeline('en', dir="./models/stanza", use_gpu=False, verbose=False)
    doc = nlp_stanza(sentence)
    ner_tags = [token.ner for token in doc.sentences[0].tokens]
    words = [word.text for word in doc.sentences[0].words]
    ents = [ent.text for sent in doc.sentences for ent in sent.ents]
    return words, ner_tags, ents


def parse_udpipe(sentence):
    nlp = spacy_udpipe.load("en")
    doc = nlp(sentence)
    words = [token.text for token in doc]
    upos = [token.pos_ for token in doc]
    return words, upos


def get_phrases(text):
    words, tags = parse_udpipe(text)
    all_indices = []

    for pattern in PATTERNS:
        idxs = [list(range(x, x + len(pattern))) for x in range(len(tags)) if tags[x:x + len(pattern)] == pattern]
        all_indices += idxs

    filtered = filter(lambda f: not any(set(f) < set(g) for g in all_indices), all_indices)
    filtered_indices = [item for item in filtered]

    words = np.array(words)
    phrases = [' '.join(list(words[ids])) for ids in filtered_indices]
    return phrases


def get_named_entities(text):
    sentences = sent_tokenize(text)
    all_ents = []
    for sent in sentences:
        _, _, ents = parse_stanza(sent)
        all_ents += ents
    return all_ents


def extract_answers(sentence):
    _all_ans = []
    for _sentence in sent_tokenize(sentence):
        entities = get_named_entities(_sentence)
        phrases = get_phrases(_sentence)
        all_ans = list(set(entities + phrases))
        all_ans = [ans.split() for ans in all_ans]
        filtered = filter(lambda f: not any(set(f) < set(g) for g in all_ans), all_ans)
        all_ans = [' '.join(item) for item in filtered]
        _all_ans.append(all_ans)
    return _all_ans
