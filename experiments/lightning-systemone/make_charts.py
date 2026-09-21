"""Regenerate publication charts from the saved measurements (standard library)."""
import json
from pathlib import Path
from html import escape

ROOT=Path(__file__).resolve().parent
def load(name):return json.loads((ROOT/name).read_text())['scenarios']
series=[('Lightning original · H100','#757575',load('comparison-lightning.json')+load('comparison-lightning-extra.json')),
        ('Lightning BF16 cache · H100','#1768ac',load('comparison-lightning-optimized.json')),
        ('DiffusionGemma · L40S configured*','#218349',load('comparison-diffusiongemma.json')+load('comparison-diffusiongemma-extra.json')),
        ('TypeSafe Jev 1.13.0 · hosted API','#8545b5',load('comparison-typesafe.json'))]
def begin(title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="0 0 1000 620" role="img" aria-label="{escape(title)}">',
            '<rect width="1000" height="620" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222;font-size:15px}.title{font-size:23px;font-weight:bold}.small{font-size:13px}</style>',
            f'<text x="80" y="35" class="title">{escape(title)}</text>']
def txt(x,y,s,extra=''):return f'<text x="{x}" y="{y}" {extra}>{escape(str(s))}</text>'
def footer():
    return [txt(80,563,'Median client-observed latency · 1,000 state tokens (Lightning tokenizer) · 10 trials · warmup excluded','class="small"'),
            txt(80,586,'*NVCF configured for L40S; placement not re-verified. Different network paths. H100 thermal throttling observed.','class="small"'),'</svg>']
out=begin('Measured latency vs. number of questions')
x=lambda n:85+n*8.1
y=lambda ms:485-ms*0.07
for n in range(0,6000,1000):
    out += [f'<line x1="80" y1="{y(n)}" x2="920" y2="{y(n)}" stroke="#ddd"/>',txt(68,y(n)+5,f'{n:,}','text-anchor="end"')]
for n in [1,12,24,50,75,100]:out.append(txt(x(n),508,n,'text-anchor="middle"'))
out += [txt(500,539,'Number of questions','text-anchor="middle"'),'<text transform="translate(22 290) rotate(-90)" text-anchor="middle">Latency (ms)</text>']
for i,(label,color,rows) in enumerate(series):
    rows=sorted(rows,key=lambda r:r['questions'])
    points=' '.join(f'{x(r["questions"])},{y(r["p50_ms"])}' for r in rows)
    out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    for r in rows:
        out.append(f'<circle cx="{x(r["questions"])}" cy="{y(r["p50_ms"])}" r="4" fill="{color}"><title>{escape(label)}: {r["questions"]} questions, {r["p50_ms"]} ms; {r["correct"]}/{r["total"]} correct</title></circle>')
    out.append(txt(235,82+i*24,label))
    out.append(f'<line x1="195" y1="{77+i*24}" x2="225" y2="{77+i*24}" stroke="{color}" stroke-width="3"/>')
out+=footer();(ROOT/'latency-vs-questions.svg').write_text('\n'.join(out))
out=begin('100 questions: latency and observed correctness')
for i,(label,color,rows) in enumerate(series):
    r=next(r for r in rows if r['questions']==100)
    yy=90+i*110; width=r['p50_ms']/5500*700
    out += [txt(80,yy,label),f'<rect x="80" y="{yy+15}" width="{width}" height="38" fill="{color}"/>',
            txt(80+width+12,yy+40,f'{r["p50_ms"]:,.0f} ms'),txt(80,yy+78,f'{r["correct"]:,}/{r["total"]:,} answers correct')]
out+=footer();(ROOT/'hundred-questions.svg').write_text('\n'.join(out))
