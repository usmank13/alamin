"""Field-bound measurements; source text is evidence, never executable content."""
import datetime
import json
import math
from pathlib import Path
import re
from fractions import Fraction
from html import unescape
from html.parser import HTMLParser
import urllib.request
import os
import fcntl
import http.client
import ipaddress
import socket
import ssl
import tempfile
import time
from urllib.parse import urlsplit, urljoin

from .contracts import PipelineError, digest, read_json, write_json

UNITS = {'m': 1., 'cm': .01, 'mm': .001, 'in': .0254, 'inch': .0254, 'inches': .0254, 'ft': .3048, 'foot': .3048, 'feet': .3048,
         '″': .0254, '"': .0254, "''": .0254, '′': .3048, "'": .3048}  # inch/foot marks as spec pages print them
CACHE = Path(os.environ.get('SCENE_PIPELINE_CACHE',Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'scene-pipeline'))/'dimensions.json'
QUANTITY = re.compile(r'(?<![\w./+-])([+-]?(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?))\s*(mm|cm|m|inches|inch|in|feet|foot|ft|″|"|\'\'|′|\')(?![\w.])')


def measurement_number(number):
    """Parse decimal, fractional, and mixed-fraction product measurements."""
    number = ' '.join(number.split())
    try:
        mixed = re.fullmatch(r'([+-]?)(\d+) (\d+/\d+)', number)
        if mixed:
            value = float(Fraction(mixed[2]) + Fraction(mixed[3]))
            if mixed[1] == '-':
                value = -value
        else:
            value = float(Fraction(number))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise PipelineError('DIMENSION', f'Invalid measurement number: {number}') from exc
    if not math.isfinite(value) or value <= 0:
        raise PipelineError('DIMENSION', 'Measurement must be finite and positive')
    return value


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
        if tag in ('p','li','tr','div','h1','h2','h3','section'):self.parts.append('\n')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def normalized_text(document):
    parser = Text()
    parser.feed(document)
    return ' '.join(unescape(' '.join(parser.parts)).split())


def measurement(document, *, identity, label, number, unit, url, axis=None):
    text = normalized_text(document)
    factor = unit_factor(unit)
    if identity not in text:
        raise PipelineError('SOURCE_IDENTITY', 'Product identity absent from evidence')
    # Label/value/unit must occur as one field, not separately anywhere on a page.
    # Pages print `Width: 144″` as often as `Width 144 in`, so whitespace is optional.
    quote = f'{label} {number} {unit}'
    if axis is not None and not re.fullmatch(r'(?:overall\s+|total\s+|assembled\s+)?'+axis+r'\s*[:.]?',label.strip(),re.I):
        raise PipelineError('UNBOUND_MEASUREMENT',f'Expected a complete {axis} field, not shipping/internal or another axis')
    parser=Text();parser.feed(document)
    fields=[' '.join(unescape(part).split()) for part in ' '.join(parser.parts).split('\n')]
    pattern=re.escape(label)+r'\s*'+re.escape(number)+r'\s*'+re.escape(unit)
    match=next((m for field in fields if (m:=re.fullmatch(pattern,field))),None)
    if not match:
        raise PipelineError('UNBOUND_MEASUREMENT', f'Field not found: {quote}')
    quote = match.group(0)
    value = measurement_number(number) * factor
    return dict(value_m=value, kind='sourced', identity=identity, field=label, quote=quote,
                source_url=url, document_sha256=digest(document))


def public_address(url):
    parsed=urlsplit(url)
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:
        raise PipelineError('EVIDENCE_FETCH','Only public HTTP(S) URLs without credentials are allowed')
    port=parsed.port or (443 if parsed.scheme=='https' else 80)
    addresses=socket.getaddrinfo(parsed.hostname,port,type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise PipelineError('EVIDENCE_FETCH','Private, loopback and reserved evidence destinations are forbidden')
    return parsed,port,addresses[0][4][0]


def fetch_bytes(url, timeout=30, *, limit=2*1024*1024, content_types=('text/html','text/plain','application/xhtml+xml')):
    """Bounded, pinned-address fetching; redirects are checked before following."""
    from .runtime import CURRENT
    if CURRENT.get():timeout=CURRENT.get().remaining(timeout)
    deadline=time.monotonic()+timeout
    try:
        for _ in range(6):
            parsed,port,address=public_address(url)
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Evidence fetch deadline exceeded')
            conn=http.client.HTTPConnection(parsed.hostname,port,timeout=remaining)
            sock=socket.create_connection((address,port),timeout=remaining)
            if parsed.scheme=='https':sock=ssl.create_default_context().wrap_socket(sock,server_hostname=parsed.hostname)
            conn.sock=sock
            try:
                conn.request('GET',(parsed.path or '/')+('?' + parsed.query if parsed.query else ''),
                             headers={'User-Agent':'scene-pipeline/0.2','Accept-Encoding':'identity'})
                response=conn.getresponse()
                if response.status in (301,302,303,307,308):
                    location=response.getheader('Location')
                    if not location:raise ValueError('Redirect without Location')
                    url=urljoin(url,location);continue
                if response.status!=200:raise ValueError(f'HTTP {response.status}')
                if content_types is not None and response.getheader('Content-Type','').split(';')[0] not in content_types:
                    raise ValueError('Evidence must be HTML or plain text')
                chunks=[];size=0
                while True:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError('Evidence fetch deadline exceeded')
                    sock.settimeout(remaining)
                    chunk=response.read1(min(65536,limit+1-size))
                    if not chunk:break
                    chunks.append(chunk);size+=len(chunk)
                    if size>limit:raise ValueError(f'Download exceeds {limit} byte limit')
                return b''.join(chunks)
            finally:conn.close()
        raise ValueError('Too many redirects')
    except (OSError,ValueError,http.client.HTTPException) as exc:
        if isinstance(exc,PipelineError):raise
        raise PipelineError('EVIDENCE_FETCH',f'Could not fetch evidence: {exc}') from exc


def fetch(url, timeout=30):
    return fetch_bytes(url,timeout).decode('utf-8','replace')


def cache_file(path=None):
    return Path(path) if path is not None else Path(os.environ.get('SCENE_PIPELINE_CACHE',CACHE.parent))/'dimensions.json'


def cache_entry(category,path):
    entry=(read_json(path) if path.exists() else {}).get(category)
    if entry is None:return None
    # Legacy caches with only a claimed hash cannot establish the original field binding.
    if 'document' not in entry or 'evidence' not in entry:return None
    evidence=entry['evidence']
    dims=[measurement(entry['document'],identity=evidence['identity'],label=evidence['fields'][axis][0],
                      number=evidence['fields'][axis][1],unit=evidence['fields'][axis][2],url=evidence['url'],axis=axis)['value_m']
          for axis in ('width','depth','height')]
    if dims!=entry['dimensions_m'] or digest(entry['document'])!=entry['document_sha256']:
        raise PipelineError('CACHE_EVIDENCE','Cached dimensions do not match archived source fields')
    return entry


def update_cache(path,category,entry):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cache=read_json(path) if path.exists() else {};cache[category]=entry
        fd,name=tempfile.mkstemp(prefix='.dimensions-',suffix='.json',dir=path.parent)
        try:
            with os.fdopen(fd,'w') as stream:json.dump(dict(sorted(cache.items())),stream,allow_nan=False)
            os.replace(name,path)
        finally:
            if Path(name).exists():Path(name).unlink()


def quoted_axes(text):
    labels={'w':r'(?:width|wide)','d':r'(?:depth|deep)','h':r'(?:height|high|tall)'}
    out={}
    for match in QUANTITY.finditer(text):
        before=text[:match.start()];after=text[match.end():]
        axes=[axis for axis,label in labels.items() if re.search(r'\b'+label+r'\s*[:=]?\s*$',before,re.I)
              or re.match(r'\s*'+label+r'\b',after,re.I)]
        if len(axes)==1:
            axis=axes[0];value=round(measurement_number(match[1])*unit_factor(match[2]),9)
            if axis in out and out[axis]!=value:raise PipelineError('UNBOUND_MEASUREMENT','Ambiguous repeated axis')
            out[axis]=value
    return out


def quoted_values(text):
    """Every number-with-unit in the text, in metres. Numbers without units bind nothing."""
    return sorted({round(measurement_number(n)*unit_factor(u), 9) for n, u in QUANTITY.findall(text)})


def resolve(item, prompt, *, default=None, cache_path=None, fetcher=fetch):
    """Return (dimensions_m, record) or None when nothing verifiable exists.

    Every axis is user-quoted, harness-verified from a fetched page, cached from such a
    page, or a declared family default. No axis comes from the agent's memory.
    """
    evidence = item.get('dimension_evidence'); category = item['category']
    cache_path=cache_file(cache_path)
    if evidence is None:
        entry=cache_entry(category,cache_path)
        return (entry['dimensions_m'], dict({k: v for k, v in entry.items() if k != 'dimensions_m'}, cached=True)) if entry else None
    if 'prompt_quote' in evidence:
        quote = evidence['prompt_quote']
        if quote not in prompt:
            raise PipelineError('UNBOUND_MEASUREMENT', 'prompt_quote absent from prompt', item['id'])
        values = quoted_axes(quote); dims = list(item['dimensions_m']); bound = []; defaulted = []; unbound = []
        for i, (axis, value) in enumerate(zip('wdh', dims)):
            if axis in values:
                if abs(value-values[axis])>1e-6:raise PipelineError('UNBOUND_MEASUREMENT',f'{axis} differs from quoted axis')
                bound.append(axis)
            elif default is not None and abs(value-default[i]) <= 1e-9: defaulted.append(axis)
            else: unbound.append(axis)
        if not bound:
            raise PipelineError('UNBOUND_MEASUREMENT', f'Quote binds no axis of {item["id"]}', dict(quote=quote))
        if not unbound:
            return dims, dict(basis='user_quoted', quote=quote, defaulted_axes=defaulted)
        # Mixed: the user fixed some axes; the rest must come from a sourced entry, never from the agent.
        entry=cache_entry(category,cache_path)
        if entry is None:
            raise PipelineError('UNBOUND_MEASUREMENT', f'Axes {unbound} of {item["id"]} are neither quoted nor sourced', dict(quote=quote, unbound_axes=unbound))
        for axis in unbound: dims['wdh'.index(axis)] = entry['dimensions_m']['wdh'.index(axis)]
        return dims, dict(basis='user_quoted_and_sourced', quote=quote, quoted_axes=bound, sourced_axes=unbound,
                          **{k: v for k, v in entry.items() if k not in ('dimensions_m', 'basis', 'family')})
    document = fetcher(evidence['url']); quotes = []
    dims = []
    for axis in ('width', 'depth', 'height'):
        label, number, unit = evidence['fields'][axis]
        found = measurement(document, identity=evidence['identity'], label=label, number=number, unit=unit, url=evidence['url'],axis=axis)
        dims.append(found['value_m']); quotes.append(found['quote'])
    record = dict(basis='sourced', url=evidence['url'], identity=evidence['identity'], quotes=quotes,
                  document_sha256=digest(document), accessed=datetime.date.today().isoformat(),document=document,evidence=evidence,
                  **({'family':item['family']} if item.get('family') else {}))
    update_cache(cache_path,category,dict(dimensions_m=dims,**record))
    return dims, record
