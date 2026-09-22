// Rebuild the report chart from the original measured comparison, not the repeat.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = path.dirname(fileURLToPath(import.meta.url));
const series = [
  {file:'comparison-nvfp4-base-lora-engine.json', name:'NVFP4 baseline', color:'#2563eb', offset:-10},
  {file:'comparison-nvfp4-trained-lora.json', name:'NVFP4 + trained LoRA', color:'#16803c', offset:10},
];
const svg = [`<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520" role="img" aria-labelledby="title desc">
<title id="title">Combination experiment: latency versus question count</title>
<desc id="desc">HTTP median latency at 1 and 100 questions. Baseline: 190.06 and 778.66 milliseconds. Trained LoRA: 214.98 and 837.22 milliseconds. Whiskers extend to p95. Intermediate counts were not measured.</desc>
<rect width="960" height="520" fill="#ffffff"/>
<g font-family="Arial, Helvetica, sans-serif" fill="#172033">
<text x="60" y="38" font-size="23" font-weight="700">Fast NVFP4 Lightning + fine-tune</text>
<text x="60" y="64" font-size="14" fill="#526174">Same original prompts · 1,000 state tokens · H100 · 10 trials per point</text>`];
const y = value => 410-value*0.29;
for (let value=0; value<=1000; value+=200) {
  svg.push(`<line x1="105" x2="850" y1="${y(value)}" y2="${y(value)}" stroke="#e2e8f0"/>
<text x="90" y="${y(value)+5}" text-anchor="end" font-size="13" fill="#526174">${value}</text>`);
}
svg.push('<text x="29" y="270" transform="rotate(-90 29 270)" text-anchor="middle" font-size="14">HTTP latency (ms)</text>');
for (let j=0;j<series.length;j++) {
  const s=series[j];
  const rows=JSON.parse(fs.readFileSync(path.join(root,s.file),'utf8')).scenarios;
  if (rows.length!==2 || rows.some(r=>![1,100].includes(r.questions)||r.successes!==10)) throw Error('Unexpected measurement set');
  svg.push(`<circle cx="${105+j*270}" cy="91" r="5" fill="${s.color}"/><text x="${118+j*270}" y="96" font-size="14">${s.name}</text>`);
  for (const row of rows) {
    const x=(row.questions===1?150:805)+s.offset;
    svg.push(`<line x1="${x}" x2="${x}" y1="${y(row.p50_ms)}" y2="${y(row.p95_ms)}" stroke="${s.color}" stroke-width="2"/>
<line x1="${x-5}" x2="${x+5}" y1="${y(row.p95_ms)}" y2="${y(row.p95_ms)}" stroke="${s.color}" stroke-width="2"/>
<circle cx="${x}" cy="${y(row.p50_ms)}" r="6" fill="${s.color}"/>
<text x="${x+(j===0?-13:13)}" y="${y(row.p50_ms)+6}" text-anchor="${j===0?'end':'start'}" font-size="15" font-weight="700" fill="${s.color}">${row.p50_ms.toFixed(0)} ms</text>`);
  }
}
svg.push(`<text x="150" y="434" text-anchor="middle" font-size="14">1</text>
<text x="805" y="434" text-anchor="middle" font-size="14">100</text>
<text x="480" y="460" text-anchor="middle" font-size="15">Questions per System One request</text>
<text x="60" y="490" font-size="13" fill="#526174">Dots: median. Whiskers: p95. Only 1 and 100 questions measured; no interpolated curve.</text>
</g></svg>`);
fs.writeFileSync(path.join(root,'latency-vs-questions.svg'),svg.join('\n')+'\n');
