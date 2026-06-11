# pip install beautifulsoup4 requests
#
# GenDB scraper — legacy query mode (unchanged behaviour) plus:
#   * fetch_page_entries(): all GenDB entries for one register page
#   * build_corpus(): page list -> deduplicated scans + corpus_manifest.json
#     (the gold-standard exchange format for the MLLM-IE pipeline)

import os
import re
import csv
import json
import time
import base64
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

# --- 1. Setup Session & Endpoints ---
session = requests.Session()
base_url = "http://gendb.bistum-passau.de"
search_page_url = f"{base_url}/"
ajax_url = f"{base_url}/ajax-register-query/"
matricula_url = f"{base_url}/ajax-matricula-image-source/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': search_page_url
}

# Separate session for Matricula downloads, with a full browser-like header set
# (a complete fingerprint is less likely to be soft-blocked than a bare UA).
image_session = requests.Session()
image_session.headers.update({
    'User-Agent': ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'),
    'Accept': ('text/html,application/xhtml+xml,application/xml;q=0.9,'
               'image/avif,image/webp,image/apng,*/*;q=0.8'),
    'Accept-Language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7',
})

PAGE_DELAY = 1.5   # polite delay between result pages
SCAN_DELAY = 0.7   # polite delay between Matricula scan lookups
ENTRY_TYPES = ['T', 'M', 'S']

CT_EXT = {
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg',
    'image/pjpeg': '.jpg',
    'image/png': '.png',
    'image/tiff': '.tif',
    'image/webp': '.webp',
    'image/gif': '.gif',
}

# Mapping backend fields -> human readable column names (full form coverage)
FIELD_MAPPING = {
    'entry_id': 'ID',
    'typ': 'Eintragstyp',
    'query_typ': 'Eintragstyp (Abfrage)',
    'qapfarrei': 'Pfarrei',
    'qaband': 'Band',
    'teilband': 'Teilband',
    'qaseite': 'Seite',
    'teilseite': 'Teilseite',
    'rv': 'Recto/Verso',
    'qanummer': 'Nummer',
    'nz-zusatz': 'Nummer Zusatz',
    'tag': 'Tag',
    'monat': 'Monat',
    'jahr': 'Jahr',
    'bisjahr': 'Datum geschätzt?',
    'pd-name': 'Nachname',
    'pd-vorname': 'Vorname',
    'vorname': 'Vorname',
    'pd-fkt': 'Rolle',
    'pd-ort': 'Ort',
    'pd-beruf': 'Beruf',
    'pd-zusatz': 'Zusatz',
    'kommentar': 'Kommentar',
    'pd_bride-name': 'Nachname (Braut)',
    'pd_bride-vorname': 'Vorname (Braut)',
    'pd_bride_vorname-vorname': 'Vorname (Braut)',
    'pd_bride-ort': 'Ort (Braut)',
    'pd_bride-beruf': 'Beruf (Braut)',
    'pd_bride-fkt': 'Rolle (Braut)',
    'pd_bride-zusatz': 'Zusatz (Braut)',
    'pd_bride-kommentar': 'Kommentar (Braut)',
    'pd_bride_kommentar-kommentar': 'Kommentar (Braut)',
    'jahre': 'Sterbealter Jahre',
    'monate': 'Sterbealter Monate',
    'wochen': 'Sterbealter Wochen',
    'tage': 'Sterbealter Tage',
    'stunden': 'Sterbealter Stunden',
    'image_id': 'Bild-ID',
    'matricula_scan_url': 'Matricula-Scan',
}

# The browser always submits the COMPLETE form (all fields, empty or not).
# Omitting e.g. the bride fields makes the server answer HTTP 500 for typ=M,
# so every query payload starts from these defaults.
FULL_FORM_DEFAULTS = {
    'typ': '', 'qapfarrei': '', 'qaband': '', 'qaseite': '', 'qanummer': '',
    'tag': '', 'monat': '', 'jahr': '', 'bisjahr': '',
    'pd-name': '', 'vorname': '', 'pd-ort': '', 'pd-beruf': '',
    'pd-fkt': '', 'pd-zusatz': '', 'kommentar': '',
    'pd_bride-name': '', 'pd_bride_vorname-vorname': '', 'pd_bride-ort': '',
    'pd_bride-beruf': '', 'pd_bride-fkt': '3', 'pd_bride-zusatz': '',
    'pd_bride_kommentar-kommentar': '',
    'teilband': '', 'teilseite': '', 'rv': '', 'nz-zusatz': '',
    'jahre': '', 'monate': '', 'wochen': '', 'tage': '', 'stunden': '',
}

# (record field, result-<li> element id) pairs that identify the scanned page
LOCATOR_SPEC = [
    ('pfarrei',   'input',  'id_result_qapfarrei'),
    ('band',      'input',  'id_result_qaband'),
    ('teilband',  'input',  'id_result_teilband'),
    ('seite',     'input',  'id_result_qaseite'),
    ('teilseite', 'input',  'id_result_teilseite'),
    ('rv',        'select', 'id_result_rv'),
]


# ---------------------------------------------------------------- scan utils

def build_scan_payload(res):
    """Build the POST payload for the Matricula image-source endpoint from a
    single result <li>, mirroring the site's showImageScan() logic."""
    pfarrei = res.find('input', id='id_result_qapfarrei')
    band = res.find('input', id='id_result_qaband')
    if pfarrei is None or band is None:
        return None

    payload = {
        'qa_pfarrei': pfarrei.get('value', ''),
        'qa_band': band.get('value', ''),
    }

    teilband = res.find('input', id='id_result_teilband')
    if teilband is not None:
        payload['teilband'] = teilband.get('value', '')

    seite = res.find('input', id='id_result_qaseite')
    if seite is not None:
        payload['qa_seite'] = seite.get('value', '')

    teilseite = res.find('input', id='id_result_teilseite')
    if teilseite is not None:
        payload['teilseite'] = teilseite.get('value', '')

    rv = res.find('select', id='id_result_rv')
    if rv is not None:
        opt = rv.find('option', selected=True)
        payload['rectoverso'] = opt.get('value', '') if opt else ''

    return payload


_scan_url_cache = {}  # frozenset(payload.items()) -> (url_or_None, reason)


def resolve_scan_url(scan_headers, payload, retries=3):
    """Ask GenDB for the Matricula scan URL of one entry (cached per page
    locator, so identical pages trigger only one lookup).
    Returns (url_or_None, reason)."""
    cache_key = frozenset(payload.items())
    if cache_key in _scan_url_cache:
        return _scan_url_cache[cache_key]

    last = "no attempt"
    result = (None, last)
    for attempt in range(1, retries + 1):
        try:
            r = session.post(matricula_url, data=payload, headers=scan_headers, timeout=30)
        except requests.RequestException as e:
            last = f"error: {e.__class__.__name__}"
        else:
            if r.status_code != 200:
                last = f"http {r.status_code}"
            else:
                try:
                    data = json.loads(r.text)
                except ValueError:
                    last = f"non-JSON response ({r.text[:80].strip()})"
                else:
                    if data.get('success') and data.get('matricula_scan_url'):
                        result = (data['matricula_scan_url'], "ok")
                        _scan_url_cache[cache_key] = result
                        return result
                    last = f"success=false ({r.text[:120].strip()})"
        if attempt < retries:
            time.sleep(1.5 * attempt)
    result = (None, last)
    _scan_url_cache[cache_key] = result
    return result


def _b64decode_loose(s):
    """Decode a (possibly unpadded, url-safe or standard) base64 string to text."""
    s = s + '=' * (-len(s) % 4)
    try:
        return base64.urlsafe_b64decode(s).decode('utf-8', 'replace')
    except Exception:
        try:
            return base64.b64decode(s).decode('utf-8', 'replace')
        except Exception:
            return ''


def matricula_image_url_from_viewer(html, viewer_url):
    """Resolve the proxied image URL for the specific page a Matricula viewer
    URL points to. The viewer embeds a MatriculaDocView with a "labels" array;
    ?pg=N (1-based) indexes into it, and each page image is served via
    /image/<base64>/ where the base64 decodes to the real image filename."""
    q = parse_qs(urlparse(viewer_url).query)
    try:
        pg = int(q.get('pg', ['1'])[0])
    except (ValueError, TypeError):
        pg = 1

    m = re.search(r'"labels"\s*:\s*(\[[^\]]*\])', html)
    if not m:
        return None
    try:
        labels = json.loads(m.group(1))
    except ValueError:
        return None
    if not labels or not (1 <= pg <= len(labels)):
        return None

    target = str(labels[pg - 1]).lower()
    for b64 in dict.fromkeys(re.findall(r'/image/([^/"\'\\\s]+)/', html)):
        decoded = _b64decode_loose(b64).lower()
        if decoded.endswith(target + '.jpg') or decoded.endswith(target + '.jpeg'):
            return urljoin(viewer_url, '/image/' + b64 + '/')
    return None


def _decoded_direct_urls(proxy_url):
    """From a Matricula /image/<b64>/ proxy URL, recover the real hosted image
    URL the base64 encodes (try both http and https)."""
    m = re.search(r'/image/([^/]+)/?$', proxy_url)
    if not m:
        return []
    dec = _b64decode_loose(m.group(1))
    if not dec.startswith('http'):
        return []
    urls = [dec]
    if dec.startswith('http://'):
        urls.append('https://' + dec[len('http://'):])
    return urls


def _save_image_response(r, dest_base):
    """Save a response body to dest_base if it is an image. Returns (path, ctype)."""
    content_type = r.headers.get('Content-Type', '').split(';')[0].strip().lower()
    ext = CT_EXT.get(content_type)
    if not ext:
        return None, content_type or 'unknown'
    dest_path = dest_base + ext
    if os.path.exists(dest_path):
        return dest_path, content_type  # idempotent: skip re-download
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    return dest_path, content_type


def _get(url, referer=None, retries=3):
    """GET via the browser-like image_session, with light retries."""
    extra = {'Referer': referer} if referer else {}
    last = "no attempt"
    for attempt in range(1, retries + 1):
        try:
            r = image_session.get(url, headers=extra, stream=True, timeout=60)
        except requests.RequestException as e:
            last = f"error: {e.__class__.__name__}"
        else:
            if r.status_code == 200:
                return r, "ok"
            last = f"http {r.status_code}"
        if attempt < retries:
            time.sleep(1.0 * attempt)
    return None, last


def download_scan_image(url, dest_base):
    """Download a scan. Returns (saved_path, reason).
    Handles a direct image URL or a Matricula viewer page; for viewer pages it
    resolves the specific page image and tries the proxy URL, then the decoded
    direct hosted URL as a fallback."""
    if url.startswith('/'):
        url = base_url + url

    r, reason = _get(url)
    if r is None:
        return None, reason

    content_type = r.headers.get('Content-Type', '').split(';')[0].strip().lower()
    if content_type in CT_EXT:                      # already a direct image
        return _save_image_response(r, dest_base)
    if 'html' not in content_type:
        return None, content_type or 'unknown'

    viewer_final_url = r.url
    proxy_url = matricula_image_url_from_viewer(r.text, viewer_final_url)
    if not proxy_url:
        return None, 'viewer page: could not locate image URL'

    last = 'no image candidate'
    for cand in [proxy_url] + _decoded_direct_urls(proxy_url):
        ir, why = _get(cand, referer=viewer_final_url)
        if ir is None:
            last = f"image {why}"
            continue
        saved, ctype = _save_image_response(ir, dest_base)
        if saved:
            return saved, ctype
        last = f"image content-type {ctype}"
    return None, last


# ----------------------------------------------------------- record handling

def _init_csrf():
    """Load the search page once and arm the session/headers with the CSRF
    token. Returns the form token (or None on failure)."""
    response = session.get(search_page_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to load main page. Status: {response.status_code}")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    csrf_input = soup.find('input', {'name': 'csrfmiddlewaretoken'})
    if not csrf_input:
        print("Error: Could not find CSRF token in the HTML.")
        return None
    csrf_token = csrf_input['value']
    csrf_cookie = session.cookies.get('csrftoken')
    headers['X-CSRFToken'] = csrf_cookie or csrf_token
    return csrf_token


# Entry-level fields that GenDB repeats inside person blocks (display_initial_extra)
ENTRY_LEVEL_DUPES = {'typ', 'qapfarrei', 'qaband', 'teilband', 'qaseite',
                     'teilseite', 'qanummer', 'tag', 'monat', 'jahr',
                     'bisjahr', 'rv', 'nz-zusatz'}


def _fields_from(el):
    """name -> value for every named <input>/<select> inside el (in order)."""
    fields = {}
    for inp in el.find_all('input'):
        name = inp.get('name')
        if name and name not in fields:
            fields[name] = inp.get('value', '').strip()
    for sel in el.find_all('select'):
        name = sel.get('name')
        if name and name not in fields:
            opt = sel.find('option', selected=True)
            val = opt.text.strip() if opt else ''
            fields[name] = '' if val == '---------' else val
    return fields


def extract_record(res, all_keys=None):
    """Extract one result <li>. Entry-level fields come from everything outside
    div.person_data; person blocks are parsed separately because GenDB renders
    the bride in a SECOND person_data block reusing the same input names
    (pd-name, pd-vorname, ...). The bride block (Rolle 'Braut', or any block
    after the first) is stored under pd_bride-* keys."""
    record_data = {}
    keys = all_keys if all_keys is not None else []

    def put(name, val):
        if name not in record_data:
            record_data[name] = val
            if name not in keys:
                keys.append(name)

    put('entry_id', res.get('id', ''))

    # entry-level: inputs/selects not inside a person block
    for tag in res.find_all(['input', 'select']):
        if tag.find_parent('div', class_='person_data'):
            continue
        name = tag.get('name')
        if not name:
            continue
        if tag.name == 'select':
            opt = tag.find('option', selected=True)
            val = opt.text.strip() if opt else ''
            val = '' if val == '---------' else val
        else:
            val = tag.get('value', '').strip()
        put(name, val)

    # person blocks: first non-bride block -> pd-*, bride block -> pd_bride-*
    primary_done = False
    for block in res.find_all('div', class_='person_data'):
        fields = {k: v for k, v in _fields_from(block).items()
                  if k not in ENTRY_LEVEL_DUPES}
        if not fields:
            continue
        is_bride = fields.get('pd-fkt') == 'Braut' or primary_done
        for name, val in fields.items():
            if is_bride:
                name = 'pd_bride-' + (name[3:] if name.startswith('pd-') else name)
            put(name, val)
        if not is_bride:
            primary_done = True

    return record_data


def locator_from_result(res):
    """Page locator (pfarrei/band/teilband/seite/teilseite/rv) of a result <li>."""
    loc = {}
    for key, tag, elem_id in LOCATOR_SPEC:
        el = res.find(tag, id=elem_id)
        if el is None:
            continue
        if tag == 'select':
            opt = el.find('option', selected=True)
            val = (opt.get('value', '') if opt else '').strip()
        else:
            val = el.get('value', '').strip()
        if val:
            loc[key] = val
    return loc


def image_id_from_locator(loc):
    """Stable, filesystem-safe id for one scanned page."""
    parts = [loc.get('pfarrei', 'unknown'), 'b' + loc.get('band', '0')]
    if loc.get('teilband'):
        parts.append('tb' + loc['teilband'])
    parts.append('s' + loc.get('seite', '0'))
    if loc.get('teilseite'):
        parts.append('ts' + loc['teilseite'])
    if loc.get('rv'):
        parts.append(loc['rv'])
    return re.sub(r'[^A-Za-z0-9_-]+', '_', '_'.join(parts))


def run_query(payload, all_keys, debug_html_dir=None, max_pages=None):
    """Paginate through one AJAX query. Returns a list of (record, <li> soup)
    tuples; record keys are appended to all_keys in encounter order."""
    results_out = []
    current_page = 1

    while True:
        print(f"  > Fetching page {current_page}...")
        payload['page'] = current_page

        ajax_response = session.post(ajax_url, data=payload, headers=headers)
        if ajax_response.status_code != 200:
            print(f"  ! HTTP {ajax_response.status_code} on page {current_page}, retrying once...")
            time.sleep(3)
            ajax_response = session.post(ajax_url, data=payload, headers=headers)
        if ajax_response.status_code != 200:
            print(f"  X Query failed on page {current_page}. Status: {ajax_response.status_code}")
            break

        result_soup = BeautifulSoup(ajax_response.text, 'html.parser')
        results = result_soup.find_all('li', class_='query_result')
        if not results:
            print(f"  * No more results found!")
            break

        for res in results:
            record_data = extract_record(res, all_keys)
            record_data['query_typ'] = payload.get('typ', '')
            if 'query_typ' not in all_keys:
                all_keys.append('query_typ')
            if debug_html_dir:
                os.makedirs(debug_html_dir, exist_ok=True)
                fname = (record_data.get('entry_id') or f"entry_{len(results_out)}") + '.html'
                fpath = os.path.join(debug_html_dir, re.sub(r'[^A-Za-z0-9_.-]+', '_', fname))
                if not os.path.exists(fpath):
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(res.prettify())
            results_out.append((record_data, res))

        current_page += 1
        if max_pages and current_page > max_pages:
            print(f"  * Reached max_pages={max_pages}, stopping.")
            break
        time.sleep(PAGE_DELAY)

    return results_out


# --------------------------------------------------------------- legacy mode

def query_gendb(search_params, output_prefix="gendb_results",
                scrape_scans=False, download_scans=False,
                output_dir=None, scan_dir=None,
                write_manifest=False, debug_html=False):
    """
    Queries GenDB using a flexible set of parameters (legacy mode, unchanged
    outputs: CSV + HTML viewer; optionally scan URLs / image downloads).

    :param search_params: dict of form fields to search for.
                          Example: {'qapfarrei': 'Altoetting', 'vorname': 'Ludwig'}
    :param output_prefix: base name for the CSV/HTML files.
    :param scrape_scans:  if True, resolve the Matricula scan URL for every entry.
    :param download_scans: if True (implies scrape_scans), also download the scan
                          image for every entry into the scan folder.
    :param output_dir:    folder for all outputs (default: <output_prefix>).
    :param scan_dir:      folder for downloaded scans (default: <output_dir>/scans).
    :param write_manifest: if True, additionally write <output_prefix>_manifest.json
                          grouping entries per scanned page (requires scrape_scans).
    :param debug_html:    if True, dump each raw result <li> to <output_dir>/debug_html/
                          (to verify field coverage of the extractor).
    """
    if download_scans:
        scrape_scans = True
    if write_manifest:
        scrape_scans = True
    if output_dir is None:
        output_dir = output_prefix
    if scan_dir is None:
        scan_dir = os.path.join(output_dir, "scans")

    os.makedirs(output_dir, exist_ok=True)

    query_summary = " | ".join([f"{FIELD_MAPPING.get(k, k)}: {v}" for k, v in search_params.items()])
    print(f"\n--- Starting Search ---")
    print(f"Parameters: {query_summary}")
    print(f"Output folder: {output_dir}/")
    if scrape_scans:
        print("Scan scraping: ENABLED" + (" (with image download)" if download_scans else " (URLs only)"))

    csrf_token = _init_csrf()
    if not csrf_token:
        return
    print("Token secured. Starting pagination loop...")

    payload = dict(FULL_FORM_DEFAULTS)
    payload.update({
        'csrfmiddlewaretoken': csrf_token,
        'typ': 'T',                       # Default to Taufen (overridable by search_params)
        'ordering': json.dumps(["date"]), # Default sorting by date
        'direction': 'ASC',
    })
    payload.update(search_params)
    # Mirror the site's JS: marriage searches pre-select groom/bride roles.
    if payload.get('typ') == 'M' and not search_params.get('pd-fkt'):
        payload['pd-fkt'] = '2'

    if download_scans:
        os.makedirs(scan_dir, exist_ok=True)

    all_keys = []
    debug_dir = os.path.join(output_dir, "debug_html") if debug_html else None
    pairs = run_query(payload, all_keys, debug_html_dir=debug_dir)

    extracted_records = []
    images = {}  # image_id -> manifest dict
    scans_resolved = 0
    scans_downloaded = 0
    first_resolve_fail = None
    first_dl_fail = None

    for record_data, res in pairs:
        if scrape_scans:
            scan_payload = build_scan_payload(res)
            if scan_payload:
                scan_url, scan_reason = resolve_scan_url(headers, scan_payload)
            else:
                scan_url, scan_reason = None, "no locator fields in result"
            record_data['matricula_scan_url'] = scan_url or ''
            if 'matricula_scan_url' not in all_keys:
                all_keys.append('matricula_scan_url')

            loc = locator_from_result(res)
            img_id = image_id_from_locator(loc)
            record_data['image_id'] = img_id
            if 'image_id' not in all_keys:
                all_keys.append('image_id')

            if img_id not in images:
                images[img_id] = {
                    'image_id': img_id,
                    'locator': loc,
                    'matricula_viewer_url': scan_url or '',
                    'local_image': '',
                    'entries': [],
                }
                if scan_url:
                    scans_resolved += 1
                    if download_scans:
                        dest_base = os.path.join(scan_dir, img_id)
                        saved, why = download_scan_image(scan_url, dest_base)
                        if saved:
                            scans_downloaded += 1
                            images[img_id]['local_image'] = saved
                        elif first_dl_fail is None:
                            first_dl_fail = (scan_url, why)
                elif first_resolve_fail is None:
                    first_resolve_fail = (record_data.get('entry_id', ''), scan_reason)
                time.sleep(SCAN_DELAY)
            images[img_id]['entries'].append(record_data)

        extracted_records.append(record_data)

    if not extracted_records:
        print("\nNo results found for these parameters.")
        return

    print(f"\nSuccessfully extracted a total of {len(extracted_records)} records!")
    if scrape_scans:
        print(f"-> {len(images)} unique scanned page(s); resolved {scans_resolved} Matricula scan URL(s).")
        if scans_resolved < len(images) and first_resolve_fail:
            print("   Some scan URLs did not resolve. Example:")
            print(f"     Entry  : {first_resolve_fail[0]}")
            print(f"     Reason : {first_resolve_fail[1]}")
        if download_scans:
            print(f"-> Downloaded {scans_downloaded} scan image(s) into '{scan_dir}/'.")
            if scans_resolved > scans_downloaded and first_dl_fail:
                print("   Some scans resolved but did not download. Example:")
                print(f"     URL         : {first_dl_fail[0]}")
                print(f"     Server sent : {first_dl_fail[1]}")

    headers_nice = [FIELD_MAPPING.get(k, k) for k in all_keys]

    csv_filename = os.path.join(output_dir, f"{output_prefix}.csv")
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(headers_nice)
        for row in extracted_records:
            writer.writerow([row.get(k, '') for k in all_keys])
    print(f"-> Saved data to {csv_filename}")

    if write_manifest:
        manifest = {
            'created': datetime.now(timezone.utc).isoformat(),
            'source': base_url,
            'query': search_params,
            'images': list(images.values()),
        }
        manifest_filename = os.path.join(output_dir, f"{output_prefix}_manifest.json")
        with open(manifest_filename, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"-> Saved manifest to {manifest_filename}")

    html_filename = os.path.join(output_dir, f"{output_prefix}.html")
    html_content = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>GenDB Results</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; padding: 20px; }}
            h1 {{ color: #333; }}
            h3 {{ color: #666; margin-top: -10px; margin-bottom: 20px; font-weight: normal; }}
            table {{ border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 15px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #0056b3; color: white; text-transform: uppercase; font-size: 0.9em; }}
            tr:hover {{ background-color: #f1f1f1; }}
            a {{ color: #0056b3; }}
        </style>
    </head>
    <body>
        <h1>GenDB Search Results</h1>
        <h3><b>Suchparameter:</b> {query_summary} <br><b>Gefundene Einträge:</b> {len(extracted_records)}</h3>
        <table>
            <thead>
                <tr>
    """
    for h in headers_nice:
        html_content += f"<th>{h}</th>"
    html_content += "</tr></thead><tbody>"

    for row in extracted_records:
        html_content += "<tr>"
        for k in all_keys:
            val = row.get(k, '')
            if k == 'matricula_scan_url' and val:
                html_content += f'<td><a href="{val}" target="_blank">Scan öffnen</a></td>'
            else:
                html_content += f"<td>{val}</td>"
        html_content += "</tr>"

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    with open(html_filename, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)
    print(f"-> Saved styled visual results to {html_filename}\n")


# ----------------------------------------------------------------- page mode

def fetch_page_entries(pfarrei, band, seite, typ=None,
                       teilband=None, teilseite=None, rv=None,
                       debug_html_dir=None):
    """All GenDB entries of one register page (the gold standard for one scan).
    Queries each entry type in T/M/S (or only `typ` if given) and merges the
    results, deduplicated by entry_id, ordered by Nummer where possible.

    Returns a list of (record, <li> soup) tuples. Requires an armed CSRF
    session (call _init_csrf() once before batches; build_corpus does this)."""
    types = [typ] if typ else ENTRY_TYPES
    seen = set()
    merged = []
    all_keys = []

    for t in types:
        payload = dict(FULL_FORM_DEFAULTS)
        payload.update({
            'typ': t,
            'qapfarrei': pfarrei,
            'qaband': str(band),
            'qaseite': str(seite),
            'ordering': json.dumps(["qanummer"]),
            'direction': 'ASC',
        })
        if t == 'M':
            payload['pd-fkt'] = '2'   # mirror site JS: groom role pre-selected
        if teilband is not None:
            payload['teilband'] = str(teilband)
        if teilseite is not None:
            payload['teilseite'] = str(teilseite)
        if rv is not None:
            payload['rv'] = rv
        print(f"  [{pfarrei} Bd.{band} S.{seite}] Eintragstyp {t}:")
        for record, res in run_query(payload, all_keys, debug_html_dir=debug_html_dir):
            eid = record.get('entry_id', '')
            if eid and eid in seen:
                continue
            seen.add(eid)
            merged.append((record, res))
        time.sleep(PAGE_DELAY)

    def _num(rec):
        try:
            return (0, int(rec[0].get('qanummer', '')))
        except (ValueError, TypeError):
            return (1, 0)
    merged.sort(key=_num)
    return merged


def build_corpus(pages, output_dir="corpus", typ=None,
                 download_scans=True, debug_html=False):
    """Build the evaluation corpus: for each page spec, fetch ALL entries
    (gold standard), resolve + download the scan once, and write
    corpus_manifest.json + gold.csv.

    :param pages: list of dicts, e.g.
                  [{'pfarrei': 'Altoetting', 'band': 5, 'seite': 12},
                   {'pfarrei': 'Vilshofen', 'band': 3, 'seite': 101, 'rv': 'r'}]
                  Optional keys per page: teilband, teilseite, rv, typ.
    :param typ:   restrict all pages to one entry type ('T'/'M'/'S'); default all.
    """
    os.makedirs(output_dir, exist_ok=True)
    scan_dir = os.path.join(output_dir, "scans")
    if download_scans:
        os.makedirs(scan_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug_html") if debug_html else None

    if not _init_csrf():
        return None
    print("Token secured. Building corpus...\n")

    images = {}
    all_keys = []
    n_entries = 0

    for spec in pages:
        pairs = fetch_page_entries(
            pfarrei=spec['pfarrei'], band=spec['band'], seite=spec['seite'],
            typ=spec.get('typ', typ),
            teilband=spec.get('teilband'), teilseite=spec.get('teilseite'),
            rv=spec.get('rv'), debug_html_dir=debug_dir,
        )
        if not pairs:
            print(f"  ! Keine Einträge für {spec} gefunden.\n")
            continue

        for record, res in pairs:
            for k in record:
                if k not in all_keys:
                    all_keys.append(k)
            loc = locator_from_result(res)
            img_id = image_id_from_locator(loc)
            record['image_id'] = img_id

            if img_id not in images:
                scan_payload = build_scan_payload(res)
                scan_url, reason = (resolve_scan_url(headers, scan_payload)
                                    if scan_payload else (None, "no locator fields"))
                local = ''
                if scan_url and download_scans:
                    saved, why = download_scan_image(scan_url, os.path.join(scan_dir, img_id))
                    local = saved or ''
                    if not saved:
                        print(f"  ! Scan-Download fehlgeschlagen für {img_id}: {why}")
                elif not scan_url:
                    print(f"  ! Scan-URL nicht auflösbar für {img_id}: {reason}")
                images[img_id] = {
                    'image_id': img_id,
                    'locator': loc,
                    'page_spec': spec,
                    'matricula_viewer_url': scan_url or '',
                    'local_image': local,
                    'entries': [],
                }
                time.sleep(SCAN_DELAY)
            images[img_id]['entries'].append(record)
            n_entries += 1

        print(f"  = {len(pairs)} Einträge für {spec['pfarrei']} Bd.{spec['band']} S.{spec['seite']}\n")

    if 'image_id' not in all_keys:
        all_keys.append('image_id')

    manifest = {
        'created': datetime.now(timezone.utc).isoformat(),
        'source': base_url,
        'page_specs': pages,
        'n_images': len(images),
        'n_entries': n_entries,
        'images': list(images.values()),
    }
    manifest_path = os.path.join(output_dir, "corpus_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "gold.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([FIELD_MAPPING.get(k, k) for k in all_keys])
        for img in images.values():
            for rec in img['entries']:
                writer.writerow([rec.get(k, '') for k in all_keys])

    print(f"Korpus fertig: {len(images)} Bilder, {n_entries} Gold-Einträge.")
    print(f"-> {manifest_path}")
    print(f"-> {csv_path}")
    return manifest_path


# ==========================================
# EXAMPLES OF HOW TO USE
# ==========================================

if __name__ == "__main__":

    # --- Legacy mode (unchanged): Taufen in Altoetting, Vorname "Ernst" ---
    # query_gendb(
    #     search_params={'typ': 'T', 'qapfarrei': 'Altoetting', 'vorname': 'Ernst'},
    #     output_prefix="Altoetting_Ernst3",
    #     scrape_scans=True,
    #     download_scans=True
    # )

    # --- Page mode: full gold standard + scan for specific register pages ---
    build_corpus(
        pages=[
            {'pfarrei': 'Altoetting', 'band': 5, 'seite': 12},
            {'pfarrei': 'Altoetting', 'band': 5, 'seite': 13},
        ],
        output_dir="corpus_test",
        download_scans=True,
        debug_html=True,   # dump raw <li> HTML to verify field coverage
    )
