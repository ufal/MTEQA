#!/usr/bin/env python

import argparse

import torch

from mteqa.answer_extraction import extract_answers
from mteqa.mteqa_pipelines import mteqa_pipeline
from mteqa.utils import compare_answers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Question based MT evaluation')
    parser.add_argument('--reference', required=True, help='Path to the file with reference translations')
    parser.add_argument('--hypothesis', required=True, help='Path to the file with output of MT system')
    parser.add_argument('--lang', required=True, help='Language of reference file')
    parser.add_argument('--cpu', action='store_true', help='Run the interference on cpu')
    parser.add_argument('--baseline_qe', action='store_true', help='Use the baseline version of answer extraction')
    parser.add_argument('--gen_from_out', action='store_true', help='Generate questions from model outputs')
    parser.add_argument('--verbose', action='store_true', help='Print scores for each pair of segments')
    args = parser.parse_args()

    if args.lang != "en":
        raise NotImplementedError("Languages other than English are not yet supported!")

    with open(args.reference) as ref:
        references = [line.strip() for line in ref]
    with open(args.hypothesis) as hyp:
        outputs = [line.strip() for line in hyp]

    assert len(references) == len(outputs), "Reference and MT output files have different number of lines!"

    mteqa = mteqa_pipeline(model="./models/qa_qg/t5-base-qa-qg-hl",
                           use_cuda=torch.cuda.is_available() and not args.cpu)

    # Make sure that the Stanza and UDPipe models are downloaded

    qa_pairs = []

    # Generating Question-Answer pairs from reference
    for _ref in references:
        if not args.baseline_qe:
            try:
                _answers = extract_answers(_ref)
                _qa_pairs = mteqa(_ref, _self_answers=_answers)
                qa_pairs.append(_qa_pairs)
            except (ValueError, AssertionError):
                qa_pairs.append([])

        else:
            try:
                _qa_pairs = mteqa(_ref)
                qa_pairs.append(_qa_pairs)
            except (ValueError, AssertionError):
                qa_pairs.append([])

    _values = {}
    for metric in ["f1", "EM", "chrf", "bleu"]:
        _values[metric] = []

    for _qa_pairs, _output in zip(qa_pairs, outputs):
        f1, EM, chrf, bleu = (0, 0, 0, 0)
        norm_factor = 1
        if _qa_pairs:
            _qa_pairs_d = [dict(_dictt) for _dictt in set(tuple(_dict.items()) for _dict in _qa_pairs)]
            for _qa_pair in _qa_pairs_d:
                ref_answer = _qa_pair["answer"]
                pred_answer = mteqa({"question": _qa_pair["question"], "context": _output})
                _f1, _EM, _chrf, _bleu = compare_answers(pred_answer, ref_answer)
                f1 += _f1
                EM += _EM
                chrf += _chrf
                bleu += _bleu
            norm_factor = len(_qa_pairs_d)

        _values["f1"].append(f1 / norm_factor)
        _values["EM"].append(EM / norm_factor)
        _values["chrf"].append(chrf / norm_factor)
        _values["bleu"].append(bleu / norm_factor)

    if args.gen_from_out:
        for _iter, (_ref, _output) in enumerate(zip(references, outputs)):
            f1, EM, chrf, bleu = (0, 0, 0, 0)
            norm_factor = 1
            if not args.baseline_qe:
                try:
                    _answers = extract_answers(_output)
                    _qa_pairs = mteqa(_output, _self_answers=_answers)
                except (ValueError, AssertionError):
                    _qa_pairs = None
            else:
                try:
                    _qa_pairs = mteqa(_output)
                except (ValueError, AssertionError):
                    _qa_pairs = None
            if _qa_pairs:
                _qa_pairs_d = [dict(_dictt) for _dictt in set(tuple(_dict.items()) for _dict in _qa_pairs)]
                for _qa_pair in _qa_pairs_d:
                    ref_answer = _qa_pair["answer"]
                    pred_answer = mteqa({"question": _qa_pair["question"], "context": references})
                    _f1, _EM, _chrf, _bleu = compare_answers(pred_answer, ref_answer)
                    f1 += _f1
                    EM += _EM
                    chrf += _chrf
                    bleu += _bleu
                norm_factor = len(_qa_pairs_d)

            _values["f1"][_iter] += f1 / norm_factor
            _values["EM"][_iter] += EM / norm_factor
            _values["chrf"][_iter] += chrf / norm_factor
            _values["bleu"][_iter] += bleu / norm_factor

        # If we generate the QA pairs also from the MT output, metrics values must be normalized
        _values["f1"][_iter] /= 2
        _values["EM"][_iter] /= 2
        _values["chrf"][_iter] /= 2
        _values["bleu"][_iter] /= 2

    if args.verbose:
        print("REF\tOUTPUT\tF1\tEM\tchrf\tbleu")
        for _iter, (_ref, _output) in enumerate(zip(references, outputs)):
            print(f"{_ref}\t{_output}\t" +
                  f"{_values['f1'][_iter]}\t" +
                  f"{_values['EM'][_iter]}\t" +
                  f"{_values['chrf'][_iter]}\t" +
                  f"{_values['bleu'][_iter]}")
    else:
        num_lines = len(references)
        print("F1\tEM\tchrf\tbleu")
        print(f"{sum(_values['f1']) / num_lines}\t" +
              f"{sum(_values['EM']) / num_lines}\t" +
              f"{sum(_values['chrf']) / num_lines}\t" +
              f"{sum(_values['bleu']) / num_lines}")
