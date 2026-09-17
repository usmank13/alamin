"""Cache primary-source evidence; no model-generated measurements."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

BIN_URL = 'https://www.rubbermaidcommercial.com/utility-refuse/step-on-containers/streamline-resin-front-step-on-containers/?sku=1883566'
SPOUT_URL = 'https://www.tsbrass.com/download?date=1740783893&file=eyJpdiI6Ikt5RXFVNTZtWmxKTFdBaTFYL0tLOEE9PSIsInZhbHVlIjoiWTZvcklQMnUyRG1zVFVSR1hkbDBqa1dlREJWSDlFWHQ5QUtDczJZMXk2U0NXVUJOODZNNFNJZ1lKQ0ZhNFVINiIsIm1hYyI6IjdjOWZlZDZlMDg1MWNmNTg2NDUxY2E3NDNhZmE2ODgyNzdiNDc0NjYwYzgwYzNlNTRlY2ZkOTMzMjE3MmE4YWEiLCJ0YWciOiIifQ%3D%3D'


def main():
    root = Path('vendor/p4_evidence')
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in [('bin.html', BIN_URL), ('spout.pdf', SPOUT_URL)]:
        dest = root/name
        if not dest.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                dest.write_bytes(response.read())
        valid = name != 'spout.pdf' or dest.read_bytes().startswith(b'%PDF')
        records.append(dict(file=name, url=url, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                            usable=valid, note='' if valid else 'Publisher returned anti-bot HTML; web-index excerpt is recorded separately.'))
    if records[-1]['usable']:
        subprocess.run(['pdftotext', '-layout', str(root/'spout.pdf'), str(root/'spout.txt')], check=True)
        subprocess.run(['pdftoppm', '-scale-to', '1200', '-png', '-singlefile', str(root/'spout.pdf'), str(root/'spout-drawing')], check=True)
    (root/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
