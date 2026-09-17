"""Field-bound measurements; source text is evidence, never executable content."""
import datetime
import json
from pathlib import Path
import re
from decimal import Decimal
from fractions import Fraction
from html import unescape
from html.parser import HTMLParser
import urllib.request

from .contracts import PipelineError, digest, read_json, write_json

UNITS = {'m': 1., 'cm': .01, 'mm': .001, 'in': .0254, 'inch': .0254, 'inches': .0254, 'ft': .3048, 'foot': .3048, 'feet': .3048,
         '″': .0254, '"': .0254, "''": .0254, '′': .3048, "'": .3048}  # inch/foot marks as spec pages print them
CACHE = Path(__file__).with_name('dimensions.json')
QUANTITY = re.compile(r'(\d+(?:\.\d+)?|\d+/\d+)\s*(mm|cm|m|inches|inch|in|feet|foot|ft|″|"|\'\'|′|\')(?![\w.])')


def unit_factor(unit):
    key = unit.strip().rstrip('.').lower()
    if key not in UNITS:
        raise PipelineError('UNIT', f'Unsupported unit: {unit}', dict(supported=sorted(UNITS)))
    return UNITS[key]


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def normalized_text(document):
    parser = Text()
    parser.feed(document)
    return ' '.join(unescape(' '.join(parser.parts)).split())


def measurement(document, *, identity, label, number, unit, url):
    text = normalized_text(document)
    factor = unit_factor(unit)
    if identity not in text:
        raise PipelineError('SOURCE_IDENTITY', 'Product identity absent from evidence')
    # Label/value/unit must occur as one field, not separately anywhere on a page.
    # Pages print `Width: 144″` as often as `Width 144 in`, so whitespace is optional.
    quote = f'{label} {number} {unit}'
    match = re.search(re.escape(label)+r'\s*'+re.escape(number)+r'\s*'+re.escape(unit)+r'(?![\w.])', text)
    if not match:
        raise PipelineError('UNBOUND_MEASUREMENT', f'Field not found: {quote}')
    quote = match.group(0)
    value = float(Fraction(number)) * factor
    if value <= 0:
        raise PipelineError('DIMENSION', 'Measurement must be positive')
    return dict(value_m=value, kind='sourced', identity=identity, field=label, quote=quote,
                source_url=url, document_sha256=digest(document))


def fetch(url, timeout=30):
    request = urllib.request.Request(url, headers={'User-Agent': 'scene-pipeline/0.1 dimension-evidence'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode('utf-8', 'replace')
    except (OSError, ValueError) as exc:  # URLError, HTTPError, timeouts, bad URLs
        raise PipelineError('EVIDENCE_FETCH', f'Could not fetch evidence page: {exc}', dict(url=url)) from exc


def quoted_values(text):
    """Every number-with-unit in the text, in metres. Numbers without units bind nothing."""
    return sorted({round(float(Fraction(n))*unit_factor(u), 9) for n, u in QUANTITY.findall(text)})


def resolve(item, prompt, *, default=None, cache_path=CACHE, fetcher=fetch):
    """Return (dimensions_m, record) or None when nothing verifiable exists.

    Every axis is user-quoted, harness-verified from a fetched page, cached from such a
    page, or a declared family default. No axis comes from the agent's memory.
    """
    evidence = item.get('dimension_evidence'); category = item['category']
    if evidence is None:
        cache = read_json(cache_path) if Path(cache_path).exists() else {}
        entry = cache.get(category)
        return (entry['dimensions_m'], dict({k: v for k, v in entry.items() if k != 'dimensions_m'}, cached=True)) if entry else None
    if 'prompt_quote' in evidence:
        quote = evidence['prompt_quote']
        if quote not in prompt:
            raise PipelineError('UNBOUND_MEASUREMENT', 'prompt_quote absent from prompt', item['id'])
        values = quoted_values(quote); dims = list(item['dimensions_m']); bound = []; defaulted = []; unbound = []
        for i, (axis, value) in enumerate(zip('wdh', dims)):
            if any(abs(value-x) <= 1e-6 for x in values): bound.append(axis)
            elif default is not None and abs(value-default[i]) <= 1e-9: defaulted.append(axis)
            else: unbound.append(axis)
        if not bound:
            raise PipelineError('UNBOUND_MEASUREMENT', f'Quote binds no axis of {item["id"]}', dict(quote=quote))
        if not unbound:
            return dims, dict(basis='user_quoted', quote=quote, defaulted_axes=defaulted)
        # Mixed: the user fixed some axes; the rest must come from a sourced entry, never from the agent.
        cache = read_json(cache_path) if Path(cache_path).exists() else {}
        entry = cache.get(category)
        if entry is None:
            raise PipelineError('UNBOUND_MEASUREMENT', f'Axes {unbound} of {item["id"]} are neither quoted nor sourced', dict(quote=quote, unbound_axes=unbound))
        for axis in unbound: dims['wdh'.index(axis)] = entry['dimensions_m']['wdh'.index(axis)]
        return dims, dict(basis='user_quoted_and_sourced', quote=quote, quoted_axes=bound, sourced_axes=unbound,
                          **{k: v for k, v in entry.items() if k not in ('dimensions_m', 'basis', 'family')})
    document = fetcher(evidence['url']); quotes = []
    dims = []
    for axis in ('width', 'depth', 'height'):
        label, number, unit = evidence['fields'][axis]
        found = measurement(document, identity=evidence['identity'], label=label, number=number, unit=unit, url=evidence['url'])
        dims.append(found['value_m']); quotes.append(found['quote'])
    record = dict(basis='sourced', url=evidence['url'], identity=evidence['identity'], quotes=quotes,
                  document_sha256=digest(document), accessed=datetime.date.today().isoformat())
    cache = read_json(cache_path) if Path(cache_path).exists() else {}
    cache[category] = dict(dimensions_m=dims, **record, **({'family': item['family']} if item.get('family') else {}))
    write_json(cache_path, dict(sorted(cache.items())))
    return dims, record
