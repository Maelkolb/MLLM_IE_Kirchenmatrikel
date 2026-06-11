# Evaluation of MLLM predictions against the GenDB gold standard.
#
# Page mode: predicted entries are first ALIGNED to gold entries per image
# (optimal 1:1 assignment over a name/date similarity; scipy if available,
# greedy fallback) -> entry detection P/R/F1, then field-level scoring on
# matched pairs. Entry mode: pairs are given via gold_entry_id.
#
# Field scoring happens ONLY where the gold value is non-empty, because the
# GenDB index is sparser than the source (predictions on gold-empty fields
# are neither rewarded nor punished). Text fields get exact match after
# normalisation plus CER; numeric/categorical fields get accuracy.
#
# Usage:
#   python evaluate.py --manifest corpus/corpus_manifest.json \
#                      --predictions predictions/predictions_page_....json

import os
import json
import csv
import argparse
from collections import defaultdict

# ------------------------------------------------------------- normalisation

TYP_MAP = {'Taufen': 'T', 'Heiraten': 'M', 'Sterbefälle': 'S'}
ZUSATZ_MAP = {'legitim': 'leg', 'illegitim': 'ill'}
TRANS = str.maketrans({'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
                       'Ä': 'ae', 'Ö': 'oe', 'Ü': 'ue'})


def norm(s):
    if s is None:
        return ''
    s = str(s).strip().lower().translate(TRANS)
    return ''.join(ch for ch in s if ch.isalnum() or ch == ' ').strip()


def levenshtein(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(gold, pred):
    g, p = norm(gold), norm(pred)
    if not g:
        return None
    return levenshtein(g, p) / len(g)


def sim(gold, pred):
    g, p = norm(gold), norm(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    return 1.0 - levenshtein(g, p) / max(len(g), len(p))


def to_int(v):
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return None


# (gold key, prediction key, kind) — kind: name (exact+CER), num, cat
FIELDS = [
    ('typ',           'eintragstyp',        'cat'),
    ('qanummer',      'nummer',             'name'),
    ('tag',           'tag',                'num'),
    ('monat',         'monat',              'num'),
    ('jahr',          'jahr',               'num'),
    ('pd-name',       'nachname',           'name'),
    ('pd-vorname',    'vorname',            'name'),
    ('pd-ort',        'ort',                'name'),
    ('pd-beruf',      'beruf',              'name'),
    ('pd-zusatz',     'zusatz',             'cat'),
    ('pd_bride-name',    'braut_nachname',  'name'),
    ('pd_bride-vorname', 'braut_vorname',   'name'),
    ('pd_bride-ort',     'braut_ort',       'name'),
    ('pd_bride-beruf',   'braut_beruf',     'name'),
    ('jahre',         'sterbealter_jahre',  'num'),
    ('monate',        'sterbealter_monate', 'num'),
    ('wochen',        'sterbealter_wochen', 'num'),
    ('tage',          'sterbealter_tage',   'num'),
    ('stunden',       'sterbealter_stunden','num'),
    ('kommentar',     'kommentar',          'name'),
]


def gold_value(record, key):
    v = (record.get(key) or '').strip()
    if key == 'typ':
        return TYP_MAP.get(v, v)
    if key == 'pd-zusatz' or key == 'pd_bride-zusatz':
        return ZUSATZ_MAP.get(v, v)
    if key == 'qanummer' and v == '0':
        return ''
    return v


# ------------------------------------------------------------------ matching

def pair_similarity(gold, pred):
    """Alignment score between one gold record and one predicted entry.
    Weighted similarity over all anchor fields that are non-empty in GOLD,
    normalised by the available weight — so sparsely indexed entries (e.g.
    marriages without a groom surname) still anchor on bride/date/number."""
    def num_eq(g, p):
        gi, pi = to_int(g), to_int(p)
        return None if gi is None else (1.0 if gi == pi else 0.0)

    components = [
        (0.40, sim(gold_value(gold, 'pd-name'), pred.get('nachname'))
               if gold_value(gold, 'pd-name') else None),
        (0.15, sim(gold_value(gold, 'pd-vorname'), pred.get('vorname'))
               if gold_value(gold, 'pd-vorname') else None),
        (0.20, sim(gold_value(gold, 'pd_bride-name'), pred.get('braut_nachname'))
               if gold_value(gold, 'pd_bride-name') else None),
        (0.10, sim(gold_value(gold, 'pd_bride-vorname'), pred.get('braut_vorname'))
               if gold_value(gold, 'pd_bride-vorname') else None),
        (0.10, num_eq(gold_value(gold, 'jahr'), pred.get('jahr'))),
        (0.05, num_eq(gold_value(gold, 'tag'), pred.get('tag'))),
        (0.05, num_eq(gold_value(gold, 'monat'), pred.get('monat'))),
        (0.05, num_eq(gold_value(gold, 'qanummer'), pred.get('nummer'))),
        (0.05, (1.0 if gold_value(gold, 'typ') == pred.get('eintragstyp') else 0.0)
               if gold_value(gold, 'typ') and pred.get('eintragstyp') else None),
    ]
    total_w = sum(w for w, s in components if s is not None)
    if total_w == 0:
        return 0.0
    return sum(w * s for w, s in components if s is not None) / total_w


def assign(golds, preds, threshold=0.3):
    """Optimal (or greedy) 1:1 assignment. Returns list of (gi, pi, score)."""
    if not golds or not preds:
        return []
    scores = [[pair_similarity(g, p) for p in preds] for g in golds]
    pairs = []
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        cost = -np.array(scores)
        rows, cols = linear_sum_assignment(cost)
        pairs = [(int(r), int(c), scores[r][c]) for r, c in zip(rows, cols)]
    except ImportError:
        cands = sorted(((scores[i][j], i, j) for i in range(len(golds))
                        for j in range(len(preds))), reverse=True)
        used_g, used_p = set(), set()
        for s, i, j in cands:
            if i in used_g or j in used_p:
                continue
            used_g.add(i); used_p.add(j)
            pairs.append((i, j, s))
    return [(i, j, s) for i, j, s in pairs if s >= threshold]


# ---------------------------------------------------------------- evaluation

def score_pair(gold, pred, rows, agg, image_id, gold_typ):
    estimated = (gold.get('bisjahr') or '') == 'geschätzt'
    for gkey, pkey, kind in FIELDS:
        gv = gold_value(gold, gkey)
        if gv == '':
            continue  # gold empty -> field not evaluable (index sparser than source)
        pv = pred.get(pkey)
        if kind == 'num':
            gi, pi = to_int(gv), to_int(pv)
            correct = int(gi is not None and gi == pi)
            char_err = None
        elif kind == 'cat':
            correct = int(norm(gv) == norm(pv))
            char_err = None
        else:
            correct = int(norm(gv) == norm(pv) and norm(gv) != '')
            char_err = cer(gv, pv)
        date_field = gkey in ('tag', 'monat', 'jahr')
        rows.append({
            'image_id': image_id, 'entry_id': gold.get('entry_id', ''),
            'typ': gold_typ, 'field': gkey, 'gold': gv,
            'pred': '' if pv is None else str(pv),
            'correct': correct,
            'cer': '' if char_err is None else round(char_err, 4),
            'date_estimated': int(estimated and date_field),
        })
        a = agg[gkey]
        a['n'] += 1
        a['correct'] += correct
        if char_err is not None:
            a['cer_sum'] += char_err
            a['cer_n'] += 1
        if date_field and not estimated:
            a['n_exactdate'] += 1
            a['correct_exactdate'] += correct


def evaluate(manifest_path, predictions_path, out_dir=None):
    manifest = json.load(open(manifest_path, encoding='utf-8'))
    preds = json.load(open(predictions_path, encoding='utf-8'))
    mode = preds.get('config', {}).get('mode', 'page')
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(predictions_path)), 'eval')
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(predictions_path))[0].replace('predictions_', '')

    rows = []
    agg = defaultdict(lambda: {'n': 0, 'correct': 0, 'cer_sum': 0.0, 'cer_n': 0,
                               'n_exactdate': 0, 'correct_exactdate': 0})
    det = {'tp': 0, 'fn': 0, 'fp': 0}
    per_image = {}

    for img in manifest['images']:
        img_id = img['image_id']
        golds = img['entries']
        pimg = preds['images'].get(img_id)
        if pimg is None or pimg.get('error'):
            det['fn'] += len(golds)
            per_image[img_id] = {'gold': len(golds), 'pred': 0, 'matched': 0,
                                 'note': (pimg or {}).get('error', 'no prediction')}
            continue
        pentries = [e for e in pimg.get('entries', []) if isinstance(e, dict) and not e.get('error')]

        if mode == 'entry':
            by_id = {e.get('gold_entry_id'): e for e in pentries}
            matched = 0
            for g in golds:
                p = by_id.get(g.get('entry_id'))
                if p is None:
                    det['fn'] += 1
                    continue
                matched += 1
                det['tp'] += 1
                score_pair(g, p, rows, agg, img_id, gold_value(g, 'typ'))
            per_image[img_id] = {'gold': len(golds), 'pred': len(pentries), 'matched': matched}
        else:
            pairs = assign(golds, pentries)
            det['tp'] += len(pairs)
            det['fn'] += len(golds) - len(pairs)
            det['fp'] += len(pentries) - len(pairs)
            for gi, pi, s in pairs:
                score_pair(golds[gi], pentries[pi], rows, agg, img_id,
                           gold_value(golds[gi], 'typ'))
            per_image[img_id] = {'gold': len(golds), 'pred': len(pentries),
                                 'matched': len(pairs)}

    # ---- detection metrics
    tp, fn, fp = det['tp'], det['fn'], det['fp']
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # ---- field report
    field_report = {}
    for gkey, _, kind in FIELDS:
        a = agg[gkey]
        if a['n'] == 0:
            continue
        entry = {'kind': kind, 'n': a['n'],
                 'accuracy': round(a['correct'] / a['n'], 4)}
        if a['cer_n']:
            entry['mean_cer'] = round(a['cer_sum'] / a['cer_n'], 4)
        if a['n_exactdate']:
            entry['accuracy_exact_dates_only'] = round(
                a['correct_exactdate'] / a['n_exactdate'], 4)
        field_report[gkey] = entry

    report = {
        'predictions': os.path.basename(predictions_path),
        'mode': mode,
        'config': preds.get('config', {}),
        'entry_detection': {'tp': tp, 'fn': fn, 'fp': fp,
                            'precision': round(prec, 4), 'recall': round(rec, 4),
                            'f1': round(f1, 4)},
        'fields': field_report,
        'per_image': per_image,
    }

    report_path = os.path.join(out_dir, f'report_{tag}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    comp_path = os.path.join(out_dir, f'comparisons_{tag}.csv')
    with open(comp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'entry_id', 'typ', 'field',
                                               'gold', 'pred', 'correct', 'cer',
                                               'date_estimated'], delimiter=';')
        writer.writeheader()
        writer.writerows(rows)

    # ---- console summary
    print(f"\n=== {tag} ===")
    print(f"Entry detection: P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
          f"(TP={tp} FN={fn} FP={fp})")
    print(f"{'Feld':<18}{'n':>5}{'Acc':>8}{'CER':>8}")
    for gkey, e in field_report.items():
        print(f"{gkey:<18}{e['n']:>5}{e['accuracy']:>8.3f}"
              f"{e.get('mean_cer', float('nan')):>8.3f}")
    print(f"\n-> {report_path}\n-> {comp_path}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Evaluation gegen GenDB-Gold")
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    evaluate(args.manifest, args.predictions, args.out)
