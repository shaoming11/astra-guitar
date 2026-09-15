"""Human-readable output: ASCII tab and an event table."""

from __future__ import annotations

from typing import List, Optional, Sequence

from .config import PipelineConfig
from .types import ArrangedGroup, NOTE_NAMES


def render_tab(
    arranged: Sequence[ArrangedGroup],
    cfg: PipelineConfig,
    line_width: int = 72,
    show_bars: bool = True,
) -> str:
    tuning = cfg.guitar.tuning
    n = tuning.n_strings
    if not arranged:
        return "(no notes detected)"

    columns: List[dict] = []
    last_measure: Optional[int] = None
    for g in arranged:
        if show_bars and g.measure is not None and g.measure != last_measure:
            columns.append({"bar": True})
            last_measure = g.measure
        cell = {p.string: str(p.fret) for p in g.shape.placements}
        width = max((len(v) for v in cell.values()), default=1)
        columns.append({"bar": False, "cell": cell, "width": width + 1})

    open_names = _string_names(cfg)
    label_w = max(len(s) for s in open_names) + 1

    lines: List[str] = []
    chunk: List[dict] = []
    used = 0
    for col in columns:
        w = 1 if col["bar"] else col["width"]
        if used + w > line_width and chunk:
            lines.extend(_emit_block(chunk, n, open_names, label_w))
            lines.append("")
            chunk, used = [], 0
        chunk.append(col)
        used += w
    if chunk:
        lines.extend(_emit_block(chunk, n, open_names, label_w))

    return "\n".join(lines).rstrip()


def _string_names(cfg: PipelineConfig) -> List[str]:
    from .types import NOTE_NAMES
    names = []
    for m in cfg.guitar.tuning.open_midi:
        names.append(NOTE_NAMES[m % 12])
    return names


def _emit_block(columns: Sequence[dict], n_strings: int, names: Sequence[str], label_w: int) -> List[str]:
    rows = []
    for s in range(n_strings - 1, -1, -1):          # highest string printed first
        label = names[s].ljust(label_w)
        buf = [label, "|"]
        for col in columns:
            if col["bar"]:
                buf.append("|")
                continue
            w = col["width"]
            val = col["cell"].get(s)
            buf.append(val.ljust(w, "-") if val else "-" * w)
        buf.append("|")
        rows.append("".join(buf))
    return rows


def string_letters(open_midi: Sequence[int]) -> List[str]:
    """One letter per string, low to high, e.g. ['E','A','D','G','B','e'].

    Standard tuning repeats E on the lowest and highest string, which is
    ambiguous in a bare STRING-FRET token. The convention this follows is
    the one already printed on paper tabs: the high string is lower-cased
    whenever it shares a letter with the lowest string.
    """
    letters = [NOTE_NAMES[m % 12] for m in open_midi]
    if len(letters) > 1 and letters[-1] == letters[0]:
        letters[-1] = letters[-1].lower()
    return letters


def render_string_tab(doc: dict) -> str:
    """Compact timing + STRING-FRET format for one-note playback.

    One note per line, time-ordered, whitespace-delimited, comments prefixed
    with '#' so a minimal line-oriented parser on the controller side can
    skip a header without knowing the schema:

        # tunefinder stringtab v1
        # tuning=standard strings=EADGBe tempo_bpm=101.33 source=microphone
        # onset_s duration_s note
        0.000 0.550 E-0
        0.610 0.550 E-0
        1.220 0.550 e-7

    Each note token is STRING-FRET, e.g. 'A-6', 'B-0', 'e-5', 'E-4' -- the
    string letter (lower-cased for the high string when it duplicates the
    low string's letter) and the fret, open strings written as fret 0.
    """
    inst = doc.get("instrument", {})
    letters = string_letters(inst.get("open_midi", []))
    header = [
        "# tunefinder stringtab v1",
        f"# tuning={inst.get('tuning', '')} strings={''.join(letters)} "
        f"tempo_bpm={doc.get('timing', {}).get('tempo_bpm', '')} "
        f"source={doc.get('source', {}).get('file', '')}",
        "# onset_s duration_s note",
    ]
    lines = list(header)
    for n in sorted(doc.get("notes", []), key=lambda r: (r["onset_s"], r["string"])):
        letter = letters[n["string"]] if n["string"] < len(letters) else "?"
        lines.append(f"{n['onset_s']:.3f} {n['duration_s']:.3f} {letter}-{n['fret']}")
    return "\n".join(lines)


def render_events(arranged: Sequence[ArrangedGroup], cfg: PipelineConfig, limit: int = 0) -> str:
    header = f"{'t(s)':>7}  {'beat':>6}  {'dur':>5}  {'pitch':<5}  {'str':>3}  {'fret':>4}  {'fing':>4}"
    lines = [header, "-" * len(header)]
    shown = 0
    for g in arranged:
        for p in sorted(g.shape.placements, key=lambda q: -q.string):
            finger = g.shape.fingers.get((p.string, p.fret), 0)
            from .types import midi_to_name
            lines.append(
                f"{g.onset:7.3f}  {(g.beat or 0):6.2f}  {g.duration:5.2f}  "
                f"{midi_to_name(p.midi):<5}  {cfg.guitar.tuning.string_label(p.string):>3}  "
                f"{p.fret:>4}  {finger if finger else '-':>4}"
            )
            shown += 1
            if limit and shown >= limit:
                lines.append(f"... ({sum(len(x.shape.placements) for x in arranged) - shown} more)")
                return "\n".join(lines)
    return "\n".join(lines)
