"""Streaming adapters. Gold labels/reference notes stay outside simulator cases."""
from __future__ import annotations
import ast
import csv
import io
import itertools
import json
from pathlib import Path
import zipfile

ACI_SPLITS = {'train': 'train.csv', 'valid': 'valid.csv', 'test1': 'clinicalnlp_taskB_test1.csv',
              'test2': 'clinicalnlp_taskC_test2.csv', 'test3': 'clef_taskC_test3.csv'}


def stringify(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f'{prefix} / {key}' if prefix else key)
    else:
        yield prefix, stringify(value)


def agentclinic_rows(path, offset, limit):
    with path.open(encoding='utf-8') as stream:
        for index, line in itertools.islice(enumerate(stream), offset, offset + limit):
            raw = json.loads(line)['OSCE_Examination']
            actor = raw['Patient_Actor']
            physical = raw.get('Physical_Examination_Findings') or {}
            symptoms = actor.get('Symptoms', {})
            chief = symptoms.get('Primary_Symptom', '') if isinstance(symptoms, dict) else symptoms
            initial = {'opening': f"{stringify(actor.get('Demographics', ''))}. {chief or '진료를 받으러 왔어요.'}",
                       'vital_signs': stringify(physical.get('Vital_Signs', '미제공')) if isinstance(physical, dict) else '미제공'}
            case = {'case_id': f'agentclinic-{index:04d}', 'start_information': initial,
                    'patient_script': actor,
                    'exam_results': [{'exam': key, 'result': value} for key, value in flatten(physical)]}
            yield {'case': case, 'gold': stringify(raw['Correct_Diagnosis']),
                   'provenance': {'dataset': 'AgentClinic-MedQA', 'split': 'extended', 'row': index,
                                  'file': str(path.resolve()), 'withheld_test_results': bool(raw.get('Test_Results'))}}


def decode_ddx(row, evidences):
    present = {}
    for item in ast.literal_eval(row['EVIDENCES']):
        key, _, value = item.partition('_@_')
        present.setdefault(key, []).append(value)
    answers = {}
    for key, meta in evidences.items():
        if meta['data_type'] == 'B':
            answer = '예 (Yes)' if key in present else '아니요 (No)'
        else:
            values = present.get(key, [str(meta['default_value'])])
            meanings = meta.get('value_meaning', {})
            decoded = [str(meanings.get(v, {}).get('en', v)) for v in values]
            answer = ', '.join(decoded)
            if answer in ('NA', 'N/A'):
                answer = '해당 없음 (데이터 기본값 NA)'
        answers[key] = {'question': meta['question_en'], 'answer': answer}
    return answers


def ddx_rows(root, split, offset, limit):
    split = 'validate' if split == 'valid' else split
    if split not in ('train', 'validate', 'test'):
        raise ValueError('DDXPlus split: train, validate, test')
    path = root / f'release_{split}_patients.zip'
    evidences = json.loads((root / 'release_evidences.json').read_text())
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if not n.endswith('/') and not n.startswith('__MACOSX/')]
        if len(members) != 1:
            raise ValueError(f'Expected one CSV member in {path}')
        with archive.open(members[0]) as binary, io.TextIOWrapper(binary, encoding='utf-8-sig') as stream:
            for index, row in itertools.islice(enumerate(csv.DictReader(stream)), offset, offset + limit):
                answers = decode_ddx(row, evidences)
                initial_key = row['INITIAL_EVIDENCE'].split('_@_')[0]
                first = answers[initial_key]
                case = {'case_id': f'ddxplus-{split}-{index:07d}',
                        'start_information': {'opening': f"{row['AGE']}세 {row['SEX']}. {first['question']} {first['answer']}",
                                              'vital_signs': '미제공'},
                        'patient_script': {}, 'exam_results': [], 'ddx_answers': answers,
                        'demographics': {'age': row['AGE'], 'sex': row['SEX']}}
                yield {'case': case, 'gold': row['PATHOLOGY'],
                       'provenance': {'dataset': 'DDXPlus', 'split': split, 'row': index, 'file': str(path.resolve()),
                                      'withheld_fields': ['PATHOLOGY', 'DIFFERENTIAL_DIAGNOSIS']}}


def aci_rows(root, split, offset, limit):
    if split not in ACI_SPLITS:
        raise ValueError('ACI-Bench split: train, valid, test1, test2, test3')
    path = root / 'aci-bench-corpus/challenge_data' / ACI_SPLITS[split]
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for index, row in itertools.islice(enumerate(csv.DictReader(stream)), offset, offset + limit):
            case = {'case_id': 'aci-' + row['encounter_id'],
                    'start_information': {'opening': '진료 대화를 초진 기록으로 정리합니다.', 'vital_signs': '대화에서 확인'},
                    'patient_script': {}, 'exam_results': []}
            yield {'case': case, 'dialogue': row['dialogue'], 'reference_note': row['note'],
                   'provenance': {'dataset': 'ACI-Bench', 'split': split, 'row': index, 'file': str(path.resolve()),
                                  'encounter_id': row['encounter_id']}}


def load_rows(root, dataset, split='validate', offset=0, limit=1):
    root = Path(root)
    if dataset == 'agentclinic':
        return agentclinic_rows(root / 'AgentClinic-MedQA/agentclinic_medqa_extended.jsonl', offset, limit)
    if dataset == 'ddxplus':
        return ddx_rows(root / 'DDXPlus', split, offset, limit)
    if dataset == 'aci':
        return aci_rows(root / 'ACI-Bench', 'valid' if split == 'validate' else split, offset, limit)
    raise ValueError(dataset)
