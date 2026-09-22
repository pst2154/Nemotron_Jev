"""Render matched isolated-request HTTP measurements as a dependency-free SVG."""
import json
import math
import statistics
from pathlib import Path


def main():
    root = Path(__file__).with_name('optimized-results')
    series = [
        ('Optimized diffusion BF16', 'vllm-http-isolated.jsonl', '#2563eb'),
        ('Lightning NVFP4 + LoRA', 'lightning-http.jsonl', '#b45309'),
    ]
    data = [(label, [json.loads(line) for line in (root / filename).read_text().splitlines()], color)
            for label, filename, color in series]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="420" viewBox="0 0 1200 420">',
             '<rect width="1200" height="420" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#172033">',
             '<text x="24" y="30" font-size="21">H100: isolated-request HTTP latency</text>',
             '<text x="24" y="53" font-size="13">Median of 3 requests; shared state; within-request prefix reuse; lower is faster</text>']
    for panel, length in enumerate([1000, 4000, 12000]):
        left, top, width, height = 62 + panel * 396, 110, 315, 230
        curves = []
        for label, rows, color in data:
            selected = [r for r in rows if r.get('state_tokens') == length]
            points = []
            for count in [1, 4, 8, 16, 32, 64, 100]:
                values = [r['ms'] for r in selected if r['questions'] == count]
                if len(values) != 3:
                    raise ValueError(f'Expected 3 repeats for {label}, {length}, {count}')
                points.append((count, statistics.median(values)))
            curves.append((color, points))
        ymax = math.ceil(max(ms for _, points in curves for _, ms in points) / 500) * 500
        parts.append(f'<text x="{left}" y="92" font-size="17">{length:,} state tokens</text>')
        for tick in range(5):
            value = ymax * tick / 4
            y = top + height * (1 - tick / 4)
            parts.append(f'<path d="M{left},{y}h{width}" stroke="#e2e8f0"/>')
            parts.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11">{value:g}</text>')
        for count in [1, 20, 40, 60, 80, 100]:
            x = left + width * count / 100
            parts.append(f'<text x="{x}" y="360" text-anchor="middle" font-size="11">{count}</text>')
        for color, points in curves:
            coords = [(left + width * count / 100, top + height * (1 - ms / ymax)) for count, ms in points]
            path = ' '.join(f'{x:.2f},{y:.2f}' for x, y in coords)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            parts.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>' for x, y in coords)
        parts.append(f'<text x="{left+width/2}" y="382" text-anchor="middle" font-size="12">Questions per request</text>')
    for i, (label, _, color) in enumerate(series):
        x = 300 + i * 320
        parts.append(f'<path d="M{x},406h25" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+33}" y="410" font-size="13">{label}</text>')
    parts.append('<text x="15" y="235" transform="rotate(-90 15 235)" font-size="12">Milliseconds</text>')
    parts.append('</g></svg>')
    (root / 'http-latency.svg').write_text('\n'.join(parts) + '\n')


if __name__ == '__main__':
    main()
