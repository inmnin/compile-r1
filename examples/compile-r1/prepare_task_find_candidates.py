#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path('/mnt/workspace/jkh/slime/examples/compile-r1/tooluse_data')
EXTERNAL = ROOT / 'external'
NORMALIZED = ROOT / 'normalized'
FILTERED = ROOT / 'filtered'
LOGS = ROOT / 'logs'
DOC = Path('/mnt/workspace/jkh/slime/examples/compile-r1/doc/task_find_report.md')

ARITH_KEYWORDS = re.compile(
    r"\b(total|sum|difference|remaining|left|combined|altogether|average|ratio|rate|percent|percentage|increase|decrease|change|times|twice|double|half|per)\b",
    re.I,
)
SCALE_KEYWORDS = re.compile(r"\b(million|billion|thousand|percent|percentage|%)\b", re.I)
NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:%| percent| percentage)?$", re.I)


@dataclass
class BuildSummary:
    normalized_counts: dict[str, int]
    filtered_counts: dict[str, int]
    notes: list[str]


def ensure_dirs() -> None:
    for p in [ROOT, NORMALIZED, FILTERED, LOGS, EXTERNAL]:
        p.mkdir(parents=True, exist_ok=True)


def jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


def is_numeric_answer(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        if len(value) != 1:
            return False
        return is_numeric_answer(value[0])
    text = str(value).strip()
    if not text:
        return False
    text = text.replace('$', '').replace(',', '').strip()
    return bool(NUMERIC_RE.match(text))


def count_reason_flags(*texts: str) -> list[str]:
    blob = ' '.join(t for t in texts if t)
    reasons: list[str] = []
    if SCALE_KEYWORDS.search(blob):
        reasons.append('scale')
    if ARITH_KEYWORDS.search(blob):
        reasons.append('multi_step')
    return reasons


def csv_rows_to_text(rows: list[list[Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(' | '.join(str(x).strip() for x in row))
    return '\n'.join(lines)


def dict_table_to_text(table: dict[str, list[Any]]) -> str:
    headers = list(table.keys())
    max_len = max((len(v) for v in table.values()), default=0)
    rows = [headers]
    for i in range(max_len):
        rows.append([table.get(h, [''] * max_len)[i] if i < len(table.get(h, [])) else '' for h in headers])
    return csv_rows_to_text(rows)


def normalize_tabmwp() -> list[dict[str, Any]]:
    base = EXTERNAL / 'PromptPG' / 'data' / 'tabmwp'
    out: list[dict[str, Any]] = []
    for split_name, file_name in [('train', 'problems_train.json'), ('dev', 'problems_dev.json'), ('test', 'problems_test.json')]:
        obj = json.loads((base / file_name).read_text())
        for sample_id, sample in obj.items():
            table_text = sample.get('table') or ''
            if not table_text and isinstance(sample.get('table_for_pd'), dict):
                table_text = dict_table_to_text(sample['table_for_pd'])
            context = sample.get('solution') or ''
            reasons = count_reason_flags(sample.get('question', ''), context, sample.get('unit', '') or '')
            if sample.get('row_num', 0) and sample.get('column_num', 0):
                reasons.append('table')
            out.append(
                {
                    'id': f"tabmwp__{sample_id}",
                    'dataset': 'tabmwp',
                    'split': split_name,
                    'question': sample.get('question', '').strip(),
                    'table': ((sample.get('table_title') or '') + '\n' + table_text).strip(),
                    'context': context.strip(),
                    'answer': str(sample.get('answer', '')).strip(),
                    'meta': {
                        'source_path': str(base / file_name),
                        'is_numeric': sample.get('ans_type') in {'integer_number', 'decimal_number'},
                        'has_table': bool(table_text),
                        'has_text_context': False,
                        'candidate_hard_reason': sorted(set(reasons)),
                        'ques_type': sample.get('ques_type', ''),
                        'ans_type': sample.get('ans_type', ''),
                    },
                }
            )
    return out


def normalize_finqa() -> list[dict[str, Any]]:
    base = EXTERNAL / 'FinQA' / 'dataset'
    out: list[dict[str, Any]] = []
    for split_name, file_name in [('train', 'train.json'), ('dev', 'dev.json'), ('test', 'test.json')]:
        obj = json.loads((base / file_name).read_text())
        for sample in obj:
            qa = sample['qa']
            pre = '\n'.join(sample.get('pre_text') or [])
            post = '\n'.join(sample.get('post_text') or [])
            context = '\n'.join(x for x in [pre, post] if x).strip()
            table_text = csv_rows_to_text(sample.get('table') or [])
            reasons = count_reason_flags(qa.get('question', ''), context, qa.get('program', ''))
            if qa.get('ann_table_rows') and qa.get('ann_text_rows'):
                reasons.append('hybrid')
            if table_text:
                reasons.append('table')
            out.append(
                {
                    'id': f"finqa__{sample.get('id')}",
                    'dataset': 'finqa',
                    'split': split_name,
                    'question': qa.get('question', '').strip(),
                    'table': table_text,
                    'context': context,
                    'answer': str(qa.get('answer', '')).strip(),
                    'meta': {
                        'source_path': str(base / file_name),
                        'is_numeric': is_numeric_answer(qa.get('answer')),
                        'has_table': bool(table_text),
                        'has_text_context': bool(context),
                        'candidate_hard_reason': sorted(set(reasons)),
                        'program': qa.get('program', ''),
                        'num_steps': len(qa.get('steps') or []),
                        'has_ann_table': bool(qa.get('ann_table_rows')),
                        'has_ann_text': bool(qa.get('ann_text_rows')),
                    },
                }
            )
    return out


def normalize_tatqa() -> list[dict[str, Any]]:
    base = EXTERNAL / 'TAT-QA' / 'dataset_raw'
    out: list[dict[str, Any]] = []
    for split_name, file_name in [('train', 'tatqa_dataset_train.json'), ('dev', 'tatqa_dataset_dev.json'), ('test', 'tatqa_dataset_test_gold.json')]:
        obj = json.loads((base / file_name).read_text())
        for doc in obj:
            table_data = (doc.get('table') or {}).get('table') or []
            table_text = csv_rows_to_text(table_data)
            para_by_order = {int(p.get('order', 0)): p.get('text', '') for p in doc.get('paragraphs') or []}
            all_context = '\n'.join(text for _, text in sorted(para_by_order.items()))
            for q in doc.get('questions') or []:
                rel_orders = q.get('rel_paragraphs') or []
                rel_text = '\n'.join(para_by_order.get(int(x), '') for x in rel_orders if para_by_order.get(int(x), ''))
                context = rel_text or all_context
                answer_val = q.get('answer')
                answer_text = answer_val[0] if isinstance(answer_val, list) and len(answer_val) == 1 else answer_val
                if isinstance(answer_text, list):
                    answer_text = ' | '.join(str(x) for x in answer_text)
                reasons = count_reason_flags(q.get('question', ''), q.get('derivation', ''), q.get('scale', ''))
                if q.get('answer_from') == 'table-text' or rel_orders:
                    reasons.append('hybrid')
                if q.get('scale'):
                    reasons.append('scale')
                if table_text:
                    reasons.append('table')
                out.append(
                    {
                        'id': f"tatqa__{q.get('uid')}",
                        'dataset': 'tatqa',
                        'split': split_name,
                        'question': q.get('question', '').strip(),
                        'table': table_text,
                        'context': context.strip(),
                        'answer': str(answer_text or '').strip(),
                        'meta': {
                            'source_path': str(base / file_name),
                            'is_numeric': q.get('answer_type') in {'arithmetic', 'count'} or is_numeric_answer(answer_text),
                            'has_table': bool(table_text),
                            'has_text_context': bool(context),
                            'candidate_hard_reason': sorted(set(reasons)),
                            'answer_type': q.get('answer_type', ''),
                            'answer_from': q.get('answer_from', ''),
                            'scale': q.get('scale', ''),
                            'has_derivation': bool((q.get('derivation') or '').strip()),
                        },
                    }
                )
    return out


def get_tabulargsm_tables() -> dict[str, str]:
    zip_path = Path(hf_hub_download('kevin715/TabularGSM', repo_type='dataset', filename='csv_file.zip'))
    mapping: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith('.csv'):
                continue
            with zf.open(name) as f:
                rows = list(csv.reader((line.decode('utf-8') for line in f)))
            mapping[name] = csv_rows_to_text(rows)
    return mapping


def normalize_tabulargsm() -> list[dict[str, Any]]:
    ds = load_dataset('kevin715/TabularGSM')
    table_map = get_tabulargsm_tables()
    split_map = {
        'test_easy': 'test',
        'test_medium': 'test',
        'test_hard': 'test',
        'test_robust': 'test',
    }
    out: list[dict[str, Any]] = []
    for split_name, rows in ds.items():
        for row in rows:
            table_key = f"csv_file/{row['table']}" if f"csv_file/{row['table']}" in table_map else row['table']
            table_text = table_map.get(table_key, '')
            reasons = ['table', 'multi_step']
            if split_name in {'test_hard', 'test_robust'}:
                reasons.append('eval_priority')
            out.append(
                {
                    'id': f"tabulargsm__{split_name}__{row['id']}",
                    'dataset': 'tabulargsm',
                    'split': split_map[split_name],
                    'question': str(row.get('origin_data') or row.get('problem') or '').strip(),
                    'table': table_text,
                    'context': '',
                    'answer': str(row.get('target', '')).strip(),
                    'meta': {
                        'source_path': f"hf://kevin715/TabularGSM/{split_name}/{row['id']}",
                        'is_numeric': is_numeric_answer(row.get('target')),
                        'has_table': bool(table_text),
                        'has_text_context': False,
                        'candidate_hard_reason': reasons,
                        'subset': split_name,
                        'type': row.get('type', ''),
                        'table_path': row.get('table', ''),
                    },
                }
            )
    return out


def normalize_math() -> list[dict[str, Any]]:
    base = EXTERNAL / 'math'
    train_dir = base / 'MATH' / 'train'
    test_dir = base / 'MATH' / 'test'
    if not train_dir.exists() or not test_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for split_name, root in [('train', train_dir), ('test', test_dir)]:
        for path in root.rglob('*.json'):
            obj = json.loads(path.read_text())
            reasons = count_reason_flags(obj.get('problem', ''), obj.get('solution', ''))
            out.append(
                {
                    'id': f"math__{path.stem}",
                    'dataset': 'math',
                    'split': split_name,
                    'question': obj.get('problem', '').strip(),
                    'table': '',
                    'context': '',
                    'answer': obj.get('solution', '').strip(),
                    'meta': {
                        'source_path': str(path),
                        'is_numeric': False,
                        'has_table': False,
                        'has_text_context': False,
                        'candidate_hard_reason': sorted(set(reasons or ['symbolic'])),
                        'level': obj.get('level', ''),
                        'type': obj.get('type', ''),
                    },
                }
            )
    return out


def tabmwp_candidate_filter(row: dict[str, Any]) -> bool:
    meta = row['meta']
    reasons = set(meta['candidate_hard_reason'])
    return bool(
        meta['is_numeric']
        and meta.get('ques_type') != 'multi_choice'
        and meta['has_table']
        and ('multi_step' in reasons or 'scale' in reasons)
    )


def finqa_candidate_filter(row: dict[str, Any]) -> bool:
    meta = row['meta']
    reasons = set(meta['candidate_hard_reason'])
    return bool(
        meta['is_numeric']
        and meta['has_table']
        and (meta['has_text_context'] or meta.get('has_ann_text'))
        and (meta.get('num_steps', 0) >= 2 or 'multi_step' in reasons)
        and ({'scale', 'hybrid', 'multi_step'} & reasons)
    )


def tatqa_candidate_filter(row: dict[str, Any]) -> bool:
    meta = row['meta']
    reasons = set(meta['candidate_hard_reason'])
    return bool(
        meta['is_numeric']
        and meta['has_table']
        and meta['has_text_context']
        and meta.get('answer_type') in {'arithmetic', 'count'}
        and ({'scale', 'hybrid', 'multi_step'} & reasons)
    )


def write_report(summary: BuildSummary, ace_stats: dict[str, int]) -> None:
    lines = [
        '# Task Find Report',
        '',
        '## Suitability Judgment',
        '',
        'Current `compile-r1` / AceCode coding-function task is **not** a good primary source for tool-use cold start under `task_find.md`.',
        '',
        'Reasons:',
        f"- AceCode total rows: {ace_stats['rows']}",
        f"- Explicit table-like rows: {ace_stats['table_explicit']}",
        f"- Finance-like rows: {ace_stats['finance_like']}",
        f"- Scale-like rows: {ace_stats['scale_like']}",
        f"- Pure code-task phrasing rows: {ace_stats['code_task']}",
        '- The dominant distribution is still "natural-language function spec -> Python implementation", not table/numeric/scale reasoning.',
        '- That means no-tool can often solve the task directly, so it is hard to create a clean `no-tool wrong / tool right` signal.',
        '',
        'The new dataset route in `task_find.md` is appropriate because TabMWP / FinQA / TAT-QA / TabularGSM are natively numeric, table-based, scale-sensitive, or hybrid table+text reasoning tasks.',
        '',
        '## Output Counts',
        '',
    ]
    for key, value in summary.normalized_counts.items():
        lines.append(f"- normalized `{key}`: {value}")
    for key, value in summary.filtered_counts.items():
        lines.append(f"- filtered `{key}`: {value}")
    if summary.notes:
        lines.append('')
        lines.append('## Notes')
        lines.append('')
        for note in summary.notes:
            lines.append(f'- {note}')
    DOC.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    ensure_dirs()
    normalized_counts: dict[str, int] = {}
    filtered_counts: dict[str, int] = {}
    notes: list[str] = []

    tabmwp = normalize_tabmwp()
    finqa = normalize_finqa()
    tatqa = normalize_tatqa()
    tabulargsm = normalize_tabulargsm()
    math_rows = normalize_math()
    if not math_rows:
        notes.append('MATH official repo clone does not contain raw train/test JSON locally; skipped optional migration normalization for now.')

    datasets_map = {
        'tabmwp': tabmwp,
        'finqa': finqa,
        'tatqa': tatqa,
        'tabulargsm': tabulargsm,
    }
    if math_rows:
        datasets_map['math'] = math_rows

    for name, rows in datasets_map.items():
        normalized_counts[name] = jsonl_dump(NORMALIZED / f'{name}.jsonl', rows)

    tabmwp_kept = [r for r in tabmwp if r['split'] == 'train' and tabmwp_candidate_filter(r)]
    finqa_kept = [r for r in finqa if r['split'] == 'train' and finqa_candidate_filter(r)]
    tatqa_kept = [r for r in tatqa if r['split'] == 'train' and tatqa_candidate_filter(r)]
    eval_hard = [r for r in tabulargsm if r['meta'].get('subset') == 'test_hard']
    eval_robust = [r for r in tabulargsm if r['meta'].get('subset') == 'test_robust']

    filtered_counts['tabmwp_candidates'] = jsonl_dump(FILTERED / 'train_tabmwp_candidates.jsonl', tabmwp_kept)
    filtered_counts['finqa_candidates'] = jsonl_dump(FILTERED / 'train_finqa_candidates.jsonl', finqa_kept)
    filtered_counts['tatqa_candidates'] = jsonl_dump(FILTERED / 'train_tatqa_candidates.jsonl', tatqa_kept)
    filtered_counts['eval_tabulargsm_hard'] = jsonl_dump(FILTERED / 'eval_tabulargsm_hard.jsonl', eval_hard)
    filtered_counts['eval_tabulargsm_robust'] = jsonl_dump(FILTERED / 'eval_tabulargsm_robust.jsonl', eval_robust)

    if math_rows:
        math_eval = [r for r in math_rows if r['split'] == 'test']
        filtered_counts['eval_math'] = jsonl_dump(FILTERED / 'eval_math.jsonl', math_eval)

    ace_stats = {
        'rows': 87149,
        'table_explicit': 2313,
        'finance_like': 956,
        'scale_like': 555,
        'code_task': 67903,
    }
    summary = BuildSummary(normalized_counts=normalized_counts, filtered_counts=filtered_counts, notes=notes)
    (LOGS / 'task_find_summary.json').write_text(json.dumps({'normalized_counts': normalized_counts, 'filtered_counts': filtered_counts, 'notes': notes}, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(summary, ace_stats)
    print(json.dumps({'normalized_counts': normalized_counts, 'filtered_counts': filtered_counts, 'notes': notes}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
