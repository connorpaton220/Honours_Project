import json, time, argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import pipeline
 
# ── Language code mapping: curate_cms.py ISO 639-3 -> NLLB BCP-47 ──
NLLB_CODES = {
    'afr': 'afr_Latn', 'swh': 'swh_Latn', 'amh': 'amh_Ethi',
    'hau': 'hau_Latn', 'kin': 'kin_Latn', 'zul': 'zul_Latn',
    'ibo': 'ibo_Latn', 'plt': 'plt_Latn', 'xho': 'xho_Latn',
    'sna': 'sna_Latn', 'yor': 'yor_Latn', 'nya': 'nya_Latn',
    'sot': 'sot_Latn', 'tir': 'tir_Ethi', 'orm': 'gaz_Latn',
    'tsn': 'tsn_Latn', 'som': 'som_Latn',
}
 
LANG_NAMES = {
    'afr':'Afrikaans','swh':'Swahili','amh':'Amharic','hau':'Hausa',
    'kin':'Kinyarwanda','zul':'Zulu','ibo':'Igbo','plt':'Malagasy',
    'xho':'Xhosa','sna':'Shona','yor':'Yoruba','nya':'Nyanja',
    'sot':'Southern Sotho','tir':'Tigrinya','orm':'Oromo',
    'tsn':'Tswana','som':'Somali',
}
 
# ── Load model once at startup ───────────────────────────────────────
def load_translator(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    print(f'Loading {model_name} on device={device}...')
    translator = pipeline(
        'translation',
        model=model_name,
        device=device,
        torch_dtype=torch.float16,  # halves VRAM, no quality loss
    )
    print('Model loaded.')
    return translator
 
 
def translate_batch(translator, texts: list, lang_code: str,
                    batch_size: int = 16) -> list:
    """Translate a list of strings to lang_code, return list of strings."""
    nllb_code = NLLB_CODES[lang_code]
    # NLLB max input = 512 tokens; truncate to ~250 words to be safe
    truncated = [' '.join(t.split()[:250]) for t in texts]
    results = []
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i:i+batch_size]
        out = translator(
            batch,
            src_lang='eng_Latn',
            tgt_lang=nllb_code,
            max_length=512,
        )
        results.extend([r['translation_text'] for r in out])
    return results
 
 
def already_done(out_file: Path) -> int:
    if not out_file.exists(): return 0
    return sum(1 for _ in out_file.open())
 
 
def main(args):
    # Load NLLB-200 once — shared across all language/domain combos
    translator = load_translator(args.model)
 
    src = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    langs = args.langs or list(NLLB_CODES.keys())
    domain_files = sorted(src.glob('*.jsonl'))
 
    for domain_file in domain_files:
        domain = domain_file.stem
        docs = [json.loads(l)['text']
                for l in domain_file.open()][:args.docs_per_lang]
 
        for lang in langs:
            out_file = out / f'{lang}_{domain}.jsonl'
            skip = already_done(out_file)
            if skip >= len(docs):
                print(f'  Skipping {lang}/{domain} — already done')
                continue
 
            print(f'\n  {domain} -> {LANG_NAMES[lang]} '
                  f'({skip}/{len(docs)} done)')
 
            remaining = docs[skip:]
            translated = translate_batch(
                translator, remaining, lang,
                batch_size=args.batch_size
            )
 
            with out_file.open('a', encoding='utf-8') as fout:
                for src_text, tgt_text in zip(remaining, translated):
                    fout.write(json.dumps({
                        'text':   tgt_text,
                        'lang':   lang,
                        'domain': domain,
                    }, ensure_ascii=False) + '\n')
 
            print(f'  Done: {len(translated)} docs')
 
 
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',        default='facebook/nllb-200-distilled-1.3B')
    p.add_argument('--source-dir',   default='/scratch/$USER/synthetic/sources')
    p.add_argument('--output-dir',   default='/scratch/$USER/synthetic/translated')
    p.add_argument('--langs',        nargs='+', default=None)
    p.add_argument('--docs-per-lang',type=int,  default=5000)
    p.add_argument('--batch-size',   type=int,  default=16)
    main(p.parse_args())
