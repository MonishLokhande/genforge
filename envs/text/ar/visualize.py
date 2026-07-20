"""Render a self-contained HTML 'sampler visual' from a trained AR checkpoint.

Rebuilds the model from a self-contained checkpoint (Invariant 5), generates BPE story samples and a
per-token CONFIDENCE heatmap (the raw temp-1 softmax probability the model gave each token it
emitted — where it's sure vs guessing), then writes a theme-aware, Artifact-ready page (title +
style + content, no html/head/body wrappers).

    python -m envs.text.ar.visualize [checkpoint.pt] [out.html]

This is a POST-HOC tool, not the in-loop `env_render` visualizer: the heatmap needs generation-time
confidences, which the render hook (final samples only) never sees.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import envs.text.ar  # noqa: F401  — register AR components so from_checkpoint can rebuild
from forge.core.builder import build
from forge.runners.training import TrainingRunner
from forge.utils.torch_utils import model_device


@torch.no_grad()
def generate(model, env, n, temp, top_k, seed, capture=False):
    """Autoregressive generation seeded from EOS (opens a fresh document), stopping at EOS. Returns
    (text, pairs); `pairs[i] = (token_glyph, raw_confidence)` when capture=True — the raw temp-1
    softmax prob of the emitted token, its true confidence, separate from the temp/top_k knobs."""
    dev = model_device(model)
    block = int(getattr(model, "length", n))
    eos = env.eos_id
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.full((1, 1), eos, dtype=torch.long, device=dev)
    glyphs, confs = [], []
    for _ in range(n):
        logits = model(idx[:, -block:], None, None)[:, -1, :]
        if capture:
            raw = F.softmax(logits, dim=-1)
        s = logits / temp
        if top_k:
            v, _ = torch.topk(s, min(top_k, s.size(-1)))
            s[s < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(s, dim=-1), 1, generator=g)
        ci = int(nxt)
        if ci == eos:                                    # story finished
            if capture:
                glyphs.append("¶")
                confs.append(float(raw[0, ci]))
            break
        glyphs.append(env.enc.decode([ci]))
        if capture:
            confs.append(float(raw[0, ci]))
        idx = torch.cat([idx, nxt], dim=1)
    return "".join(glyphs), list(zip(glyphs, confs))


# confidence p -> bucket (b0 = very sure/calm ... b5 = very unsure/hot)
_EDGES = [0.85, 0.60, 0.40, 0.25, 0.12]


def _bucket(p: float) -> int:
    for i, e in enumerate(_EDGES):
        if p >= e:
            return i
    return len(_EDGES)


def heatmap_html(pairs) -> str:
    return "".join(
        f'<span class="b{_bucket(p)}" data-p="{p:.2f}">{html.escape(g)}</span>' for g, p in pairs
    )


def build_page(model, env, metric=None) -> str:
    _, ann_pairs = generate(model, env, 400, temp=0.85, top_k=40, seed=42, capture=True)
    avg = sum(p for _, p in ann_pairs) / max(1, len(ann_pairs))
    samples = [generate(model, env, 200, 0.8, 40, s)[0].strip() for s in (11, 23, 37)]
    temps = [(t, generate(model, env, 160, t, 0, 7)[0].strip()) for t in (0.4, 0.8, 1.2)]

    n_params = sum(p.numel() for p in model.parameters())
    ppl = "—"
    if metric is not None:
        try:
            ppl = f"{float(metric().get('val_perplexity', float('nan'))):.2f}"
        except Exception:
            ppl = "—"

    sample_cards = "\n".join(
        f'<article class="card"><span class="tag">sample {i+1}</span>'
        f'<p class="story">{html.escape(s)}</p></article>'
        for i, s in enumerate(samples)
    )
    temp_cards = "\n".join(
        f'<article class="card"><span class="tag">T = {t:g}</span>'
        f'<p class="story">{html.escape(s)}</p></article>'
        for t, s in temps
    )
    return (_TITLE + _STYLE
            + _BODY.replace("%%PARAMS%%", f"{n_params/1e6:.1f}M")
                   .replace("%%PPL%%", ppl)
                   .replace("%%VOCAB%%", f"{env.vocab_size:,}")
                   .replace("%%AVG%%", f"{avg*100:.0f}")
                   .replace("%%HEATMAP%%", heatmap_html(ann_pairs))
                   .replace("%%SAMPLES%%", sample_cards)
                   .replace("%%TEMPS%%", temp_cards))


def main(argv) -> None:
    ckpt = argv[0] if argv else "checkpoints/text/ar/base.pt"
    out = Path(argv[1]) if len(argv) > 1 else Path(ckpt).with_name("sampler.html")
    runner = TrainingRunner.from_checkpoint(ckpt, build_fn=build)
    runner.model.eval()
    page = build_page(runner.model, runner.environment, getattr(runner, "metric", None))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out}  (avg confidence in the annotated sample; open or publish as an Artifact)")


_TITLE = "<title>nanoGPT sampler — autoregressive BPE LM</title>\n"

_STYLE = """<style>
:root{
  --paper:#f7f5f1; --surface:#fffefb; --ink:#23202b; --muted:#6f6a78; --accent:#4f46b8;
  --rule:#e8e3da; --card-shadow:0 1px 2px rgba(30,24,45,.05),0 8px 24px -12px rgba(30,24,45,.12);
  --b1:#fdf0dc; --b2:#fbdfb0; --b3:#f7c877; --b4:#f0aa3f; --b5:#e2891c; --heat-ink:#2a2331;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#151319; --surface:#1c1a22; --ink:#ece7f2; --muted:#9a94a6; --accent:#a99cff;
  --rule:#2c2934; --card-shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px -14px rgba(0,0,0,.6);
  --b1:#2b2417; --b2:#43371b; --b3:#63491e; --b4:#8d6522; --b5:#b9862a; --heat-ink:#f3eefa;
}}
:root[data-theme="light"]{
  --paper:#f7f5f1; --surface:#fffefb; --ink:#23202b; --muted:#6f6a78; --accent:#4f46b8;
  --rule:#e8e3da; --b1:#fdf0dc; --b2:#fbdfb0; --b3:#f7c877; --b4:#f0aa3f; --b5:#e2891c; --heat-ink:#2a2331;
}
:root[data-theme="dark"]{
  --paper:#151319; --surface:#1c1a22; --ink:#ece7f2; --muted:#9a94a6; --accent:#a99cff;
  --rule:#2c2934; --b1:#2b2417; --b2:#43371b; --b3:#63491e; --b4:#8d6522; --b5:#b9862a; --heat-ink:#f3eefa;
}
*{box-sizing:border-box}
body{margin:0}
.wrap{
  --serif:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace;
  background:var(--paper); color:var(--ink); font-family:var(--sans);
  min-height:100vh; padding:clamp(24px,5vw,64px); line-height:1.5;
}
.inner{max-width:820px; margin:0 auto}
header{border-bottom:1px solid var(--rule); padding-bottom:26px; margin-bottom:34px}
.eyebrow{font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); font-weight:600; margin:0 0 10px}
h1{font-family:var(--serif); font-weight:600; font-size:clamp(1.9rem,4.5vw,2.9rem); line-height:1.08;
   margin:0 0 8px; text-wrap:balance; letter-spacing:-.01em}
.sub{color:var(--muted); margin:0 0 22px; max-width:56ch}
.chips{display:flex; flex-wrap:wrap; gap:10px}
.chip{display:flex; flex-direction:column; gap:2px; padding:8px 14px; background:var(--surface);
      border:1px solid var(--rule); border-radius:10px; box-shadow:var(--card-shadow)}
.chip b{font-family:var(--mono); font-size:1.05rem; font-variant-numeric:tabular-nums; letter-spacing:-.01em}
.chip span{font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
section{margin:40px 0}
h2{font-family:var(--serif); font-weight:600; font-size:1.4rem; margin:0 0 4px}
.note{color:var(--muted); font-size:.92rem; margin:0 0 18px; max-width:64ch}
.heat{font-family:var(--mono); font-size:.94rem; line-height:2.05; white-space:pre-wrap; word-break:break-word;
  color:var(--heat-ink); background:var(--surface); border:1px solid var(--rule); border-radius:14px;
  padding:22px 24px; box-shadow:var(--card-shadow); overflow-x:auto}
.heat span{border-radius:3px; padding:.06em 0}
.heat .b1{background:var(--b1)} .heat .b2{background:var(--b2)} .heat .b3{background:var(--b3)}
.heat .b4{background:var(--b4)} .heat .b5{background:var(--b5)}
.heat .b3,.heat .b4,.heat .b5{padding:.06em .04em}
.legend{display:flex; align-items:center; gap:12px; margin:16px 2px 0; font-size:.78rem; color:var(--muted)}
.ramp{display:flex; height:12px; flex:0 0 220px; border-radius:6px; overflow:hidden; border:1px solid var(--rule)}
.ramp i{flex:1}
.ramp .b0{background:var(--surface)} .ramp .b1{background:var(--b1)} .ramp .b2{background:var(--b2)}
.ramp .b3{background:var(--b3)} .ramp .b4{background:var(--b4)} .ramp .b5{background:var(--b5)}
.grid{display:grid; grid-template-columns:1fr; gap:16px}
@media(min-width:680px){.grid.three{grid-template-columns:repeat(3,1fr)}}
.card{background:var(--surface); border:1px solid var(--rule); border-radius:14px; padding:18px 20px;
  box-shadow:var(--card-shadow)}
.tag{display:inline-block; font-family:var(--mono); font-size:.7rem; letter-spacing:.06em; color:var(--accent);
  border:1px solid var(--rule); border-radius:999px; padding:3px 10px; margin-bottom:12px; text-transform:uppercase}
.story{font-family:var(--mono); font-size:.86rem; line-height:1.75; white-space:pre-wrap; margin:0; color:var(--ink)}
footer{margin-top:44px; padding-top:20px; border-top:1px solid var(--rule); color:var(--muted); font-size:.82rem}
footer code{font-family:var(--mono); background:var(--surface); padding:1px 6px; border-radius:5px; border:1px solid var(--rule)}
</style>
"""

_BODY = """<div class="wrap"><div class="inner">
<header>
  <p class="eyebrow">Autoregressive BPE sampler</p>
  <h1>What the little model actually writes</h1>
  <p class="sub">A nanoGPT-style transformer, trained from scratch one GPT-2-BPE token at a time,
  generating a complete, <em>self-terminating</em> document — shown with the model's own confidence
  in every token it picks.</p>
  <div class="chips">
    <div class="chip"><b>%%PARAMS%%</b><span>parameters</span></div>
    <div class="chip"><b>%%PPL%%</b><span>held-out ppl</span></div>
    <div class="chip"><b>%%VOCAB%%</b><span>BPE vocab</span></div>
    <div class="chip"><b>%%AVG%%%</b><span>avg confidence</span></div>
  </div>
</header>

<section>
  <h2>Confidence heatmap</h2>
  <p class="note">Each token is tinted by the raw probability the model assigned it. Calm = the model
  was sure (the skeleton of English: spaces, common words); warm = it was genuinely choosing (a name,
  the next noun) — exactly where the text is being invented. The final <b>¶</b> is the end-of-text
  token the model emits to stop; its tint is how sure it was the document was over.</p>
  <div class="heat">%%HEATMAP%%</div>
  <div class="legend"><span>confident</span><div class="ramp"><i class="b0"></i><i class="b1"></i><i class="b2"></i><i class="b3"></i><i class="b4"></i><i class="b5"></i></div><span>uncertain</span></div>
</section>

<section>
  <h2>Fresh samples</h2>
  <p class="note">Three independent generations, temperature 0.8, top-k 40 — each seeded only with the end-of-text token.</p>
  <div class="grid three">%%SAMPLES%%</div>
</section>

<section>
  <h2>Turning the temperature dial</h2>
  <p class="note">Same seed, no top-k. Low temperature plays it safe and repetitive; high temperature
  gets adventurous and starts to fall apart — the coherence/diversity trade-off, visible.</p>
  <div class="grid three">%%TEMPS%%</div>
</section>

<footer>
  Generated by <code>python -m envs.text.ar.visualize</code> from a self-contained forge checkpoint.
  Confidence = softmax(logits) at temperature 1 for the emitted token.
</footer>
</div></div>
"""


if __name__ == "__main__":
    main(sys.argv[1:])
