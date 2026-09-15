# tunefinder

Listen to a tune. Get back a guitar tab and a timed command stream two robot arms can execute.

The hard part is not pitch detection. It is that the fretboard is ambiguous. Middle C sits in four places on a standard guitar, and a tab that picks each note independently produces something no hand and no machine can physically play. tunefinder picks the whole fretting path at once.

## Install

```bash
pip install librosa soundfile numpy scipy
pip install sounddevice      # optional, only for microphone input
```

## Use

```bash
# audio file to robot JSON plus ASCII tab
python -m tunefinder transcribe riff.wav -o riff.json --tab riff.tab

# hear what it decided to play, rendered with a plucked string model
python -m tunefinder transcribe riff.wav --preview check.wav

# chords instead of a single line
python -m tunefinder transcribe rhythm.wav --poly

# no audio file handy
python -m tunefinder demo
python -m tunefinder demo --chords

# record eight seconds from the default input
python -m tunefinder listen --seconds 8 -o take.json

# check a document is ordered, consistent and physically executable
python -m tunefinder validate riff.json

# start/stop recording window: hum, press stop, get files back
python -m tunefinder ui
```

As a library:

```python
from tunefinder import run, PipelineConfig, GuitarSpec, Tuning

cfg = PipelineConfig(guitar=GuitarSpec(tuning=Tuning.get("drop_d"), max_fret=12))
doc, arranged, transcription = run("riff.wav", cfg)

doc["commands"]   # the executable timeline
doc["notes"]      # one record per sounded note
doc["tab"]        # ASCII tab
```

Useful flags: `--tuning`, `--capo`, `--max-fret`, `--max-span`, `--no-open-strings`, `--quantize N`, `--prefer-low-frets`, `--minimise-shifts`, `--events`.

## Recording UI

`python -m tunefinder ui` opens a small window with two buttons: **Start Recording** and **Stop Recording**. Press start, hum a melody into the microphone, press stop. The take runs through the full pipeline immediately and three files land in `--outdir` (default `recordings/`), timestamped `hum-YYYYMMDD-HHMMSS.*`:

- `.stab` -- the compact STRING-FRET timing file, see below
- `.json` -- the full `robotab/1.0` document
- `.tab` -- the ASCII tab

The window also prints a summary (note count, the ASCII tab, the `.stab` contents) so you can sanity check a take without opening a file. Flags: `--tuning`, `--capo`, `--max-fret`, `--outdir`, `--quantize`.

## Output

Three output formats, from richest to leanest:

**`robotab/1.0` JSON** (`-o` / `--out`, schema in `schema/robotab.schema.json`) is the full executable command stream for two robot arms. Every time is seconds from the start of the take, so a controller runs the whole piece off one clock. Beat numbers ride along for controllers that prefer a metronome. String index 0 is the lowest pitched string. Finger 0 is an open string, finger 1 is the index finger.

```json
{"t": 1.103, "arm": "fret", "action": "move",  "position": 1, "from_position": 2, "travel_s": 0.022, "feasible": true},
{"t": 1.126, "arm": "fret", "action": "press", "string": 4, "fret": 3, "finger": 3, "note_id": "n0001s4"},
{"t": 1.186, "arm": "pick", "action": "strum", "string": 0, "direction": "down", "velocity": 0.79}
```

Actions are `press`, `release`, `release_all` and `move` for the fretting arm, `pluck`, `strum` and `rest` for the picking arm.

**ASCII tab** (`--tab`) is the human-readable six-line tab shown in every example above.

**STRING-FRET timing file** (`--strtab`, also written by the UI as `.stab`) is a minimal, line-oriented protocol for a controller that just wants "when" and "where" and doesn't need press/release/strum choreography -- one note per line, comments prefixed `#`:

```
# tunefinder stringtab v1
# tuning=standard strings=EADGBe tempo_bpm=101.33 source=microphone
# onset_s duration_s note
0.000 0.550 E-0
0.610 0.550 E-0
1.220 0.550 e-7
```

Each note token is `STRING-FRET`: the string letter and the fret, open strings written as fret `0`. Letters follow standard tab convention, low to high -- `E A D G B e` -- with the high string lower-cased whenever it duplicates the low string's letter (standard and drop-D tuning both do; most alternate tunings don't, and every letter stays uppercase). `A-6` is the A string, 6th fret. `e-5` is the high E string, 5th fret -- lower case distinguishes it from the low `E` string. Onset and duration are seconds from the start of the take, matching the JSON format's clock, so both can be read off the same recording. Parse it as whitespace-separated fields per line; anything starting with `#` is a comment.

Four things the scheduler handles that a naive dump of note events does not:

- **Press lead.** A finger lands 60 ms before its pluck. Configurable.
- **Handoffs.** When the next note uses the same string, the current finger lifts early enough that the new one is down before the pluck. A note stops ringing rather than the arm arriving late.
- **Holds.** A finger staying on the same fret across consecutive notes is not lifted and re-pressed.
- **Reachability.** Every hand shift is checked against the arm's traverse speed and the time available. Shifts that do not fit are flagged in `warnings` and marked `"feasible": false`.

## How the fretting path is chosen

Each note or chord becomes a set of candidate hand shapes. A shape is valid when no two notes share a string, the fret span fits the hand, and the fingers required do not exceed four. Barres are used when three or more strings share the low fret, or when the fingers would otherwise run out.

A Viterbi search then runs over the whole piece. Emission cost is shape difficulty: fret height, stretch, number of fingers, with a bonus for open strings and a penalty for barres. Transition cost is hand travel: distance moved along the neck, strings the picking arm crosses, with a reward for fingers that stay planted.

The weights live in `CostWeights` and are worth tuning to a specific machine. A robot with a slow neck traverse wants `hand_shift` raised. One with a fast fretting hand and a slow picking arm wants `string_travel` raised.

Given an unlabelled recording of Em, C, G and D, the arranger recovers 022000, x32010, 320003 and xx0232 with no chord dictionary.

## Transcription

Monophonic mode tracks the fundamental with pYIN, then segments on pitch changes and re-attacks. Three things it does beyond a textbook implementation, each of which fixed a real failure:

- Confidence is pitch stability, the fraction of frames agreeing on the note. pYIN's own voicing probability is unreliable on plucked strings and discards correct notes.
- Onsets are refined against a short window energy derivative. The pitch analysis window is 93 ms long and straddles the attack, which puts raw segment starts about 30 ms early.
- Pitch excursions with no envelope rise are absorbed into the note before them. A dip to the neighbouring semitone inside a held note is a tracker artifact, and so is the semitone passed through while one note decays into the next. Set `strip_continuations=False` for legato playing where hammer-ons arrive without a fresh attack.

Polyphonic mode picks peaks from a constant-Q transform at each onset and suppresses harmonics. It is coarser than the monophonic path. It handles open chords and double stops well and dense voicings less well.

Measured on a synthetic plucked riff with overlapping ring and added noise: all twelve pitches correct, median onset error under 10 ms.

## Limits

- One note at a time per string. Bends, slides, vibrato and palm muting are not modelled.
- Polyphonic mode assumes notes start on detected onsets. Arpeggios inside a sustained chord get grouped.
- Tempo comes from `librosa.beat.beat_track` and can land on half or double time. Quantisation only snaps a note when the grid is within a third of a beat, so a bad tempo degrades timing rather than destroying it. Pass `--quantize 0` for raw timing.
- Onset-poor material -- a hummed melody has no percussive attack for the beat tracker to lock onto -- can make `beat_track` report 0 bpm. That falls back to an assumed 120 bpm grid for quantisation rather than collapsing every note onto the same instant; `timing.tempo_bpm` in the output is then a placeholder, not a measurement, so don't read it as the performer's actual tempo.
- Notes outside the instrument range are folded by octaves into it rather than dropped.

## Layout

```
tunefinder/
  config.py       tunings, instrument limits, cost weights, robot envelope
  types.py        NoteEvent, Placement, Shape, ArrangedGroup, Transcription
  transcribe.py   pitch tracking, onset segmentation, polyphonic mode, tempo, StreamRecorder
  timing.py       beat mapping, quantisation, octave folding
  fretboard.py    candidate positions, shape enumeration, fingering, barres
  arrange.py      Viterbi search over the fretting path
  robot.py        press, hold, release and pluck scheduling
  tabtext.py      ASCII tab, event table, STRING-FRET stringtab
  synth.py        test signals and plucked string playback
  cli.py          command line
  ui.py           start/stop recording window
schema/robotab.schema.json
examples/run_on_robot.py
tests/test_pipeline.py
```

## Tests

```bash
python tests/test_pipeline.py      # or: python -m pytest tests/ -q
```

Nineteen tests covering fretboard geometry, barre rules, finger limits, arranger behaviour, command ordering, the press-before-pluck invariant, chord recognition and end to end pitch and onset accuracy.
