"""Compile an arrangement into a timed command stream for two robot arms.

Arm `fret` owns finger placement and hand position along the neck.
Arm `pick` owns plucking and strumming.

Every command carries an absolute time in seconds from the start of the take,
so a controller can run the whole thing off one clock. Musical time (beat
number) rides along for controllers that prefer to follow a metronome.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PipelineConfig
from .types import ArrangedGroup, midi_to_hz, midi_to_name


def _note_id(group_idx: int, string: int) -> str:
    return f"n{group_idx:04d}s{string}"


def _strum_order(group: ArrangedGroup, direction: str) -> List:
    placements = sorted(group.shape.placements, key=lambda p: p.string)
    return placements if direction == "down" else list(reversed(placements))


def _pick_direction(group: ArrangedGroup, prev_dir: str, cfg: PipelineConfig) -> str:
    if len(group.shape.placements) > 1:
        # strum down on the beat, up between beats
        frac = (group.beat or 0.0) % 1.0
        return "down" if frac < 0.25 or frac > 0.75 else "up"
    if not cfg.robot.alternate_picking:
        return "down"
    return "up" if prev_dir == "down" else "down"


def build_commands(
    arranged: Sequence[ArrangedGroup], cfg: PipelineConfig
) -> Tuple[List[dict], List[dict], List[str]]:
    """Return (note_records, commands, warnings)."""
    r = cfg.robot
    warnings: List[str] = []
    notes: List[dict] = []
    commands: List[dict] = []

    # ---- 1. resolve pluck times and note records -------------------------
    prev_dir = "up"
    per_string: Dict[int, List[dict]] = defaultdict(list)

    for g in arranged:
        direction = _pick_direction(g, prev_dir, cfg)
        prev_dir = direction
        order = _strum_order(g, direction)
        is_strum = len(order) > 1
        by_midi = {n.midi: n for n in g.notes}

        for i, p in enumerate(order):
            pluck_t = g.onset + (i * r.strum_stagger if is_strum else 0.0)
            src = by_midi.get(p.midi)
            duration = src.duration if src else g.duration
            rec = {
                "id": _note_id(g.index, p.string),
                "group": g.index,
                "onset_s": round(pluck_t, 5),
                "duration_s": round(max(duration, 0.02), 5),
                "beat": round(g.beat, 4) if g.beat is not None else None,
                "measure": g.measure,
                "midi": p.midi,
                "pitch": midi_to_name(p.midi),
                "freq_hz": round(midi_to_hz(p.midi), 3),
                "string": p.string,
                "string_label": cfg.guitar.tuning.string_label(p.string),
                "fret": p.fret,
                "finger": g.shape.fingers.get((p.string, p.fret), 0),
                "open": p.is_open,
                "velocity": round(src.velocity if src else 0.7, 3),
                "confidence": round(src.confidence if src else 0.5, 3),
                "strum_direction": direction if is_strum else None,
                "pick_direction": direction,
                "hand_position": g.shape.hand_position,
                "barre": (
                    list(g.shape.barre)
                    if g.shape.barre
                    and p.fret == g.shape.barre[0]
                    and g.shape.barre[1] <= p.string <= g.shape.barre[2]
                    else None
                ),
            }
            notes.append(rec)
            per_string[p.string].append(rec)

    # ---- 2. fretting hand: press / hold / release ------------------------
    for string, seq in per_string.items():
        seq.sort(key=lambda x: x["onset_s"])
        for i, rec in enumerate(seq):
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            natural_end = rec["onset_s"] + rec["duration_s"]
            if nxt is not None:
                if nxt["fret"] > 0:
                    # the next finger has to land press_lead before its pluck,
                    # so this one must be clear of the fret before that
                    handoff = nxt["onset_s"] - r.press_lead
                    limit = handoff - r.min_press_gap - r.release_lag
                else:
                    limit = nxt["onset_s"] - r.min_press_gap
                if limit < rec["onset_s"] + 0.03:
                    warnings.append(
                        f"string {rec['string_label']} handoff at "
                        f"t={nxt['onset_s']:.3f}s leaves under 30ms of ring time"
                    )
                natural_end = min(natural_end, limit)
            rec["_end"] = max(natural_end, rec["onset_s"] + 0.03)
            rec["_hold_next"] = bool(
                nxt is not None
                and nxt["fret"] == rec["fret"]
                and nxt["fret"] > 0
                and nxt["onset_s"] - rec["_end"] <= 0.30
            )
            rec["_held_from_prev"] = False

        for i, rec in enumerate(seq):
            if i > 0 and seq[i - 1].get("_hold_next"):
                rec["_held_from_prev"] = True

        prev_release = -1e9
        for rec in seq:
            if rec["fret"] == 0:
                continue
            if not rec["_held_from_prev"]:
                press_t = rec["onset_s"] - r.press_lead
                press_t = max(press_t, prev_release + r.min_press_gap, 0.0)
                if rec["onset_s"] - press_t < r.press_lead * 0.5:
                    warnings.append(
                        f"tight press at t={rec['onset_s']:.3f}s "
                        f"string {rec['string_label']} fret {rec['fret']}"
                    )
                commands.append({
                    "t": round(press_t, 5),
                    "arm": "fret",
                    "action": "press",
                    "string": rec["string"],
                    "fret": rec["fret"],
                    "finger": rec["finger"],
                    "note_id": rec["id"],
                    "barre": rec["barre"],
                })
            if not rec["_hold_next"]:
                rel_t = rec["_end"] + r.release_lag
                commands.append({
                    "t": round(rel_t, 5),
                    "arm": "fret",
                    "action": "release",
                    "string": rec["string"],
                    "fret": rec["fret"],
                    "finger": rec["finger"],
                    "note_id": rec["id"],
                })
                prev_release = rel_t

    # ---- 3. picking hand --------------------------------------------------
    for rec in notes:
        commands.append({
            "t": round(rec["onset_s"], 5),
            "arm": "pick",
            "action": "strum" if rec["strum_direction"] else "pluck",
            "string": rec["string"],
            "direction": rec["pick_direction"],
            "velocity": rec["velocity"],
            "note_id": rec["id"],
        })

    # ---- 4. hand position moves ------------------------------------------
    prev_pos: Optional[int] = None
    prev_time = 0.0
    for g in arranged:
        pos = g.shape.hand_position
        if pos == 0:
            continue
        if prev_pos is None or pos != prev_pos:
            distance = abs(pos - (prev_pos or 1))
            travel = distance / max(r.fret_travel_speed, 1e-6)
            move_t = max(g.onset - r.press_lead - travel, prev_time, 0.0)
            initial = prev_pos is None
            available = g.onset - r.press_lead - prev_time
            feasible = True if initial else available >= travel
            if not feasible:
                warnings.append(
                    f"hand shift of {distance} frets at t={g.onset:.3f}s needs "
                    f"{travel*1000:.0f}ms, has {max(available,0)*1000:.0f}ms"
                )
            commands.append({
                "t": round(move_t, 5),
                "arm": "fret",
                "action": "move",
                "position": pos,
                "from_position": prev_pos,
                "travel_s": round(travel, 5),
                "feasible": feasible,
                "pre_take": initial,
            })
            prev_pos = pos
        prev_time = g.onset

    if notes:
        end = max(rec["onset_s"] + rec["duration_s"] for rec in notes) + 0.25
        commands.append({"t": round(end, 5), "arm": "fret", "action": "release_all"})
        commands.append({"t": round(end, 5), "arm": "pick", "action": "rest"})

    commands.sort(key=lambda c: (c["t"], 0 if c["arm"] == "fret" else 1))
    for i, c in enumerate(commands):
        c["seq"] = i

    for rec in notes:
        for k in ("_end", "_hold_next", "_held_from_prev"):
            rec.pop(k, None)

    return notes, commands, warnings


def build_document(
    arranged: Sequence[ArrangedGroup],
    cfg: PipelineConfig,
    transcription,
    stats: dict,
    tab_text: str = "",
    quantized: Optional[int] = None,
) -> dict:
    notes, commands, warnings = build_commands(arranged, cfg)
    return {
        "format": "robotab/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "file": transcription.source,
            "duration_s": round(transcription.duration, 3),
            "sample_rate": transcription.sr,
            "mode": "polyphonic" if transcription.polyphonic else "monophonic",
        },
        "instrument": {
            "type": "guitar",
            "tuning": cfg.guitar.tuning.name,
            "open_midi": list(cfg.guitar.tuning.open_midi),
            "string_index": "0 = lowest pitched string",
            "max_fret": cfg.guitar.max_fret,
            "capo": cfg.guitar.capo,
        },
        "timing": {
            "tempo_bpm": round(transcription.tempo_bpm, 2),
            "time_signature": "4/4",
            "quantized_subdivision": quantized,
            "beat_times": [round(b, 4) for b in transcription.beat_times],
            "clock": "seconds from start of take",
        },
        "robot": {
            "press_lead_s": cfg.robot.press_lead,
            "release_lag_s": cfg.robot.release_lag,
            "strum_stagger_s": cfg.robot.strum_stagger,
            "arms": ["fret", "pick"],
        },
        "stats": stats,
        "notes": notes,
        "commands": commands,
        "warnings": warnings,
        "tab": tab_text,
    }
