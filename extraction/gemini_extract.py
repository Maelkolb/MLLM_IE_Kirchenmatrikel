# pip install google-genai
#
# MLLM-based information extraction from Matricula register scans with
# Gemini 3.5 Flash. Reads the corpus_manifest.json produced by the scraper,
# sends each scan to Gemini with a structured-output JSON schema mirroring
# the GenDB fields, and writes resumable prediction files.
#
# Two task settings:
#   --mode page   one call per image, extract ALL entries on the page
#   --mode entry  one call per gold entry, extract ONLY the located entry
#                 (locator: Eintragsnummer if recorded, else ordinal position)
#
# Experiment variables: --thinking-level, --media-resolution, --few-shot.
#
# Usage:
#   export GEMINI_API_KEY=...
#   python gemini_extract.py --manifest corpus/corpus_manifest.json --mode page
#   python gemini_extract.py --manifest ... --mode entry --thinking-level low

import os
import json
import time
import argparse
import mimetypes

from google import genai
from google.genai import types

MODEL_DEFAULT = "gemini-3.5-flash"

# --------------------------------------------------------------- JSON schema

NULLABLE_STR = {"type": "string", "nullable": True}
NULLABLE_INT = {"type": "integer", "nullable": True}

ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "eintragstyp": {"type": "string", "enum": ["T", "M", "S"],
                        "description": "T=Taufe, M=Heirat, S=Sterbefall"},
        "nummer": NULLABLE_STR,
        "tag": NULLABLE_INT,
        "monat": NULLABLE_INT,
        "jahr": NULLABLE_INT,
        "nachname": NULLABLE_STR,
        "vorname": NULLABLE_STR,
        "ort": NULLABLE_STR,
        "beruf": NULLABLE_STR,
        "zusatz": {"type": "string", "enum": ["leg", "ill"], "nullable": True,
                   "description": "leg=ehelich/legitim, ill=unehelich/illegitim"},
        "braut_nachname": NULLABLE_STR,
        "braut_vorname": NULLABLE_STR,
        "braut_ort": NULLABLE_STR,
        "braut_beruf": NULLABLE_STR,
        "sterbealter_jahre": NULLABLE_INT,
        "sterbealter_monate": NULLABLE_INT,
        "sterbealter_wochen": NULLABLE_INT,
        "sterbealter_tage": NULLABLE_INT,
        "sterbealter_stunden": NULLABLE_INT,
        "kommentar": NULLABLE_STR,
    },
    "required": ["eintragstyp"],
}

PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "eintraege": {"type": "array", "items": ENTRY_SCHEMA},
    },
    "required": ["eintraege"],
}

# ------------------------------------------------------------------- prompts

FIELD_RULES = """Felder pro Eintrag (nicht Lesbares/nicht Vorhandenes = null):
- eintragstyp: "T" (Taufe), "M" (Heirat/Trauung), "S" (Sterbefall/Begräbnis)
- nummer: laufende Eintragsnummer auf der Seite, falls vorhanden
- tag, monat, jahr: Datum des Ereignisses als Zahlen. Monatsnamen umrechnen,
  auch lateinische Kurzformen (7bris/September=9, 8bris=10, 9bris=11,
  Xbris/10bris=12, Jänner=1). Das Jahr steht oft nur im Seitenkopf oder beim
  ersten Eintrag und gilt fortlaufend.
- nachname, vorname: Hauptperson des Eintrags (Täufling bei T, Bräutigam bei M,
  Verstorbener bei S). Schreibweise originalgetreu übernehmen, nicht
  modernisieren. Lateinische Vornamen in der Form der Quelle belassen.
- ort: Wohn-/Herkunftsort der Hauptperson, falls genannt
- beruf: Beruf/Stand der Hauptperson, falls genannt
- zusatz: "leg" bei ehelich/legitimus, "ill" bei unehelich/illegitimus
- braut_nachname, braut_vorname, braut_ort, braut_beruf: nur bei Heiraten (M)
- sterbealter_*: nur bei Sterbefällen (S), Altersangabe aufgeschlüsselt
  (z. B. "47 annor." -> sterbealter_jahre=47; "15 hebdom." -> wochen=15)
- kommentar: sonstige relevante Angaben in Kurzform (z. B. Eltern, Zeugen,
  Todesursache), sonst null"""

PAGE_PROMPT = """Du siehst den Scan einer Seite aus einem historischen Kirchenbuch (Matrikel) des Bistums Passau, 17.–19. Jahrhundert, in deutscher Kurrentschrift und/oder Latein, oft tabellarisch.

Aufgabe: Extrahiere ALLE Einträge dieser Seite vollständig und in Lesereihenfolge (von oben nach unten, ggf. spaltenweise).

{rules}

Gib ausschließlich das JSON-Objekt zurück."""

ENTRY_PROMPT = """Du siehst den Scan einer Seite aus einem historischen Kirchenbuch (Matrikel) des Bistums Passau, 17.–19. Jahrhundert, in deutscher Kurrentschrift und/oder Latein, oft tabellarisch.

Aufgabe: Extrahiere NUR {locator}. Ignoriere alle anderen Einträge der Seite.

{rules}

Gib ausschließlich das JSON-Objekt für diesen einen Eintrag zurück."""


def entry_locator(record, ordinal, total):
    num = (record.get('qanummer') or '').strip()
    if num and num != '0':
        return f"den Eintrag mit der laufenden Nummer {num}"
    return f"den {ordinal}. Eintrag von oben (von insgesamt {total} Einträgen auf der Seite)"


# ----------------------------------------------------------------- API layer

def get_client():
    key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not key:
        try:  # Colab secret fallback
            from google.colab import userdata
            key = userdata.get('GEMINI_API_KEY')
        except Exception:
            pass
    if not key:
        raise SystemExit("GEMINI_API_KEY nicht gesetzt (Env-Variable oder Colab-Secret).")
    return genai.Client(api_key=key)


def make_config(schema, thinking_level=None, media_resolution=None):
    kwargs = {
        'response_mime_type': 'application/json',
        'response_schema': schema,
    }
    if thinking_level:
        kwargs['thinking_config'] = types.ThinkingConfig(thinking_level=thinking_level)
    if media_resolution:
        kwargs['media_resolution'] = 'MEDIA_RESOLUTION_' + media_resolution.upper()
    return types.GenerateContentConfig(**kwargs)


def image_part(path):
    mime = mimetypes.guess_type(path)[0] or 'image/jpeg'
    with open(path, 'rb') as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)


def parse_json_response(response):
    """Prefer SDK-parsed structured output, fall back to robust text parsing."""
    parsed = getattr(response, 'parsed', None)
    if parsed is not None:
        return parsed if isinstance(parsed, (dict, list)) else json.loads(json.dumps(parsed, default=lambda o: o.__dict__))
    text = (response.text or '').strip()
    text = text.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    return json.loads(text)


def call_gemini(client, model, contents, config, retries=4):
    last = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config)
            data = parse_json_response(response)
            usage = getattr(response, 'usage_metadata', None)
            usage_dict = {
                'prompt_tokens': getattr(usage, 'prompt_token_count', None),
                'output_tokens': getattr(usage, 'candidates_token_count', None),
                'thoughts_tokens': getattr(usage, 'thoughts_token_count', None),
            } if usage else {}
            return data, (response.text or ''), usage_dict, None
        except Exception as e:
            last = f"{e.__class__.__name__}: {e}"
            print(f"    ! Versuch {attempt}/{retries} fehlgeschlagen: {last[:160]}")
            if attempt < retries:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    return None, '', {}, last


# ------------------------------------------------------------- few-shot demo

def load_few_shot(path):
    """few_shot.json: [{"image": "path/to/scan.jpg", "page": {"eintraege": [...]}}]
    Each example becomes a (user image+prompt, model JSON) demonstration pair."""
    examples = json.load(open(path, encoding='utf-8'))
    contents = []
    for ex in examples:
        contents.append(types.Content(role='user', parts=[
            image_part(ex['image']),
            types.Part.from_text(text=PAGE_PROMPT.format(rules=FIELD_RULES)),
        ]))
        contents.append(types.Content(role='model', parts=[
            types.Part.from_text(text=json.dumps(ex['page'], ensure_ascii=False)),
        ]))
    return contents


# -------------------------------------------------------------------- runner

def run(args):
    manifest = json.load(open(args.manifest, encoding='utf-8'))
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))

    tag = f"{args.mode}_{args.model.replace('/', '-')}" \
          f"_tl-{args.thinking_level or 'default'}" \
          f"_mr-{args.media_resolution or 'default'}" \
          f"{'_fs' if args.few_shot else ''}"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"predictions_{tag}.json")

    # resume: keep finished images
    if os.path.exists(out_path):
        preds = json.load(open(out_path, encoding='utf-8'))
        print(f"Resume: {out_path} ({len(preds.get('images', {}))} Bilder vorhanden)")
    else:
        preds = {
            'config': {
                'mode': args.mode, 'model': args.model,
                'thinking_level': args.thinking_level,
                'media_resolution': args.media_resolution,
                'few_shot': bool(args.few_shot),
                'manifest': os.path.abspath(args.manifest),
            },
            'images': {},
        }

    client = get_client()
    schema = ENTRY_SCHEMA if args.mode == 'entry' else PAGE_SCHEMA
    config = make_config(schema, args.thinking_level, args.media_resolution)
    fs_contents = load_few_shot(args.few_shot) if args.few_shot else []

    images = manifest['images'][:args.limit] if args.limit else manifest['images']
    for img in images:
        img_id = img['image_id']
        done = preds['images'].get(img_id)
        if done and not done.get('error'):
            continue
        local = img.get('local_image', '')
        path = local if os.path.isabs(local) else os.path.join(os.path.dirname(manifest_dir) or '.', local)
        if not local or not os.path.exists(path):
            # also try relative to manifest dir directly
            alt = os.path.join(manifest_dir, os.path.basename(os.path.dirname(local) or 'scans'),
                               os.path.basename(local)) if local else ''
            if alt and os.path.exists(alt):
                path = alt
            else:
                print(f"  ! Bild fehlt für {img_id}: {local}")
                preds['images'][img_id] = {'error': 'image not found', 'entries': []}
                continue

        print(f"  > {img_id} ({args.mode})...")
        result = {'entries': [], 'error': None, 'usage': [], 'raw': []}

        if args.mode == 'page':
            contents = fs_contents + [types.Content(role='user', parts=[
                image_part(path),
                types.Part.from_text(text=PAGE_PROMPT.format(rules=FIELD_RULES)),
            ])]
            if args.dry_run:
                print(PAGE_PROMPT.format(rules=FIELD_RULES)); return
            data, raw, usage, err = call_gemini(client, args.model, contents, config)
            if err:
                result['error'] = err
            else:
                result['entries'] = data.get('eintraege', []) if isinstance(data, dict) else data
                result['usage'].append(usage)
                result['raw'].append(raw if args.keep_raw else '')
        else:  # entry mode: one call per gold entry
            gold_entries = img['entries']
            for i, gold in enumerate(gold_entries, start=1):
                loc = entry_locator(gold, i, len(gold_entries))
                prompt = ENTRY_PROMPT.format(locator=loc, rules=FIELD_RULES)
                contents = fs_contents + [types.Content(role='user', parts=[
                    image_part(path), types.Part.from_text(text=prompt),
                ])]
                if args.dry_run:
                    print(prompt); return
                data, raw, usage, err = call_gemini(client, args.model, contents, config)
                entry = data if isinstance(data, dict) else {}
                entry['gold_entry_id'] = gold.get('entry_id', '')
                entry['locator'] = loc
                if err:
                    entry['error'] = err
                result['entries'].append(entry)
                result['usage'].append(usage)
                result['raw'].append(raw if args.keep_raw else '')
                time.sleep(args.delay)

        preds['images'][img_id] = result
        with open(out_path, 'w', encoding='utf-8') as f:  # save after every image
            json.dump(preds, f, ensure_ascii=False, indent=2)
        time.sleep(args.delay)

    n_err = sum(1 for v in preds['images'].values() if v.get('error'))
    print(f"\nFertig: {len(preds['images'])} Bilder, {n_err} mit Fehlern.")
    print(f"-> {out_path}")
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Gemini 3.5 Flash IE auf Matrikel-Scans")
    ap.add_argument('--manifest', required=True, help="corpus_manifest.json des Scrapers")
    ap.add_argument('--mode', choices=['page', 'entry'], default='page')
    ap.add_argument('--model', default=MODEL_DEFAULT)
    ap.add_argument('--thinking-level', choices=['minimal', 'low', 'medium', 'high'], default=None)
    ap.add_argument('--media-resolution', choices=['low', 'medium', 'high', 'ultra_high'], default=None)
    ap.add_argument('--few-shot', default=None, help="Pfad zu few_shot.json (optional)")
    ap.add_argument('--out', default='predictions', help="Output-Ordner")
    ap.add_argument('--limit', type=int, default=None, help="nur erste N Bilder (Kostenkontrolle)")
    ap.add_argument('--delay', type=float, default=1.0, help="Pause zwischen API-Calls (s)")
    ap.add_argument('--keep-raw', action='store_true', help="rohe Modellantworten mitspeichern")
    ap.add_argument('--dry-run', action='store_true', help="nur Prompt anzeigen, kein API-Call")
    run(ap.parse_args())
