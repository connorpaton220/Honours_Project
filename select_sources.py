from datasets import load_dataset
from pathlib import Path
import json
 
DOMAINS = {
    'science':     ['physics','chemistry','biology','research','study','experiment'],
    'health':      ['health','medicine','disease','treatment','symptoms','medical'],
    'education':   ['education','learning','school','university','teaching','student'],
    'history':     ['history','historical','century','ancient','colonial','war'],
    'finance':     ['economy','finance','business','market','investment','trade'],
    'law':         ['law','legal','rights','court','government','policy'],
    'food':        ['food','recipe','cooking','cuisine','culture','tradition'],
    'news':        ['news','report','event','announced','confirmed','stated'],
    'stories':     ['story','tale','narrative','once','journey','village'],
    'agriculture': ['farm','agriculture','crop','soil','harvest','climate'],
}
 
DOCS_PER_DOMAIN = 60_000  # enough for all 17 languages
MIN_WORDS, MAX_WORDS = 80, 300  # short docs = faster translation
 
out = Path('/scratch/ptncon001/synthetic/sources')
out.mkdir(parents=True, exist_ok=True)
 
counts = {d: 0 for d in DOMAINS}
done = set()
 
ds = load_dataset('HuggingFaceFW/fineweb-2', name='eng_Latn',
                  split='train', streaming=True)
 
for ex in ds:
    if len(done) == len(DOMAINS): break
    text = ex.get('text', '')
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS): continue
    text_lower = text.lower()
    for domain, keywords in DOMAINS.items():
        if domain in done: continue
        if sum(1 for kw in keywords if kw in text_lower) >= 2:
            with (out / f'{domain}.jsonl').open('a') as f:
                f.write(json.dumps({'text': text, 'domain': domain},
                                   ensure_ascii=False) + '\n')
            counts[domain] += 1
            if counts[domain] >= DOCS_PER_DOMAIN:
                done.add(domain)
                print(f'  ✓ {domain}: {counts[domain]} docs')
            break
 
print('Done:', counts)
