# tunefinder

Listen to a tune. Get back a timed, single-note melody transcription with every tab note arranged on the D string.

The hard part is not pitch detection. It is that the fretboard is ambiguous. Middle C sits in four places on a standard guitar, and a tab that picks each note independently produces something no hand and no machine can physically play. tunefinder picks the whole fretting path at once.

## Install

```bash
pip install librosa soundfile numpy scipy
pip install sounddevice      # optional, only for microphone input
pip install yt-dlp           # YouTube URL input
# FFmpeg is required for video files and URLs: brew install ffmpeg
```

## Use

```bash
# audio file to timed melody JSON plus ASCII tab
python -m tunefinder transcribe riff.wav -o riff.json --tab riff.tab

# local video (MP4/MOV/WebM/etc.) or a YouTube URL
python -m tunefinder transcribe performance.mp4 -o performance.json
python -m tunefinder transcribe 'https://www.youtube.com/watch?v=...' -o melody.json

# hear what it decided to play, rendered with a plucked string model
python -m tunefinder transcribe riff.wav --preview check.wav

# no audio file handy
python -m tunefinder demo

# record eight seconds from the default input
python -m tunefinder listen --seconds 8 -o take.json

# check a document is ordered and internally consistent
python -m tunefinder validate riff.json

# start/stop recording window: hum, press stop, get files back
python -m tunefinder ui
```

As a library:

```python
from tunefinder import run, PipelineConfig, GuitarSpec, Tuning

cfg = PipelineConfig(guitar=GuitarSpec(tuning=Tuning.get("drop_d"), max_fret=12))
doc, arranged, transcription = run("riff.wav", cfg)

doc["notes"]      # one record per sounded note
doc["tab"]        # ASCII tab
```

Useful flags: `--tuning`, `--capo`, `--max-fret`, `--max-span`, `--no-open-strings`, `--quantize N`, `--prefer-low-frets`, `--minimise-shifts`, `--events`.

## Recording UI

`python -m tunefinder ui` opens a small window with two buttons: **Start Recording** and **Stop Recording**. Press start, hum a melody into the microphone, press stop. The take runs through the full pipeline immediately and three transcription files land in `--outdir` (default `recordings/`), timestamped `hum-YYYYMMDD-HHMMSS.*`. A cleaned source WAV is saved too, which makes humming failures easy to inspect:

- `.stab` -- the compact STRING-FRET timing file, see below
- `.json` -- the full `tunefinder/1.0` document
- `.tab` -- the ASCII tab

The window also prints a summary (note count, the ASCII tab, the `.stab` contents) so you can sanity check a take without opening a file. Flags: `--tuning`, `--capo`, `--max-fret`, `--outdir`, `--quantize`.

The same window can also choose a local video/audio file or accept a YouTube URL. Every UI source is decoded to mono audio and simplified to a single monophonic melody line, so the result is one note at a time. The window also exposes tuning/capo/fretboard options, quantisation, confidence and note thresholds, voice-cleanup controls, a demo melody, JSON validation, event-table output, and optional WAV preview/source export. The CLI `transcribe` command accepts the same local paths and URLs. YouTube needs `yt-dlp`; video and direct URL sources need FFmpeg.

Microphone recordings are cleaned before pitch tracking with a voice-band filter, adaptive spectral noise reduction, and a soft foreground gate. This suppresses room noise and quieter competing voices. A single mono microphone cannot guarantee speaker identity; for the best separation, keep the mic close to the user and use headphones for playback. Use `--raw-mic` only when diagnosing the cleanup.

## Output

The output contains a timed JSON transcription, ASCII tab, and compact STRING-FRET timing text. The default arrangement uses only the D string (physical string 4, internal index 2), so every note token is `D-fret` in standard tuning.

**`tunefinder/1.0` JSON** (`-o` / `--out`) contains one timed note record per detected melody note. Each record includes pitch, onset, duration, guitar string, fret, finger, and a `string_fret` token such as `e-5`. Beat numbers are included when available. String index 0 is the lowest pitched string. Finger 0 is an open string, finger 1 is the index finger.

```json
{"note_id": "n0000", "onset_s": 1.103, "duration_s": 0.42, "pitch": "E4", "midi": 64,
 "string": 5, "string_label": "1", "fret": 0, "string_fret": "e-0", "finger": 0}
```

**ASCII tab** (`--tab`) is the human-readable six-line tab shown in every example above.

**STRING-FRET timing text** (`--strtab`, also written by the UI as `.stab`) is a minimal, line-oriented file with one note per line and comments prefixed `#`:

```
# tunefinder stringtab v1
# tuning=standard strings=EADGBe tempo_bpm=101.33 source=microphone
# onset_s duration_s note
0.000 0.550 E-0
0.610 0.550 E-0
1.220 0.550 e-7
```

Each note token is `STRING-FRET`: the string letter and the fret, open strings written as fret `0`. Letters follow standard tab convention, low to high -- `E A D G B e` -- with the high string lower-cased whenever it duplicates the low string's letter (standard and drop-D tuning both do; most alternate tunings don't, and every letter stays uppercase). `A-6` is the A string, 6th fret. `e-5` is the high E string, 5th fret -- lower case distinguishes it from the low `E` string. Onset and duration are seconds from the start of the take, matching the JSON format's clock, so both can be read off the same recording. Parse it as whitespace-separated fields per line; anything starting with `#` is a comment.

## How the fretting path is chosen

Each note becomes a set of candidate fretboard positions. The arranger chooses a continuous path that respects the fret span, finger limit, tuning, capo, and open-string settings.

A Viterbi search then runs over the whole piece. Emission cost is shape difficulty: fret height, stretch, number of fingers, with a bonus for open strings and a penalty for barres. Transition cost is hand travel: distance moved along the neck, strings the picking arm crosses, with a reward for fingers that stay planted.

The weights live in `CostWeights` and can be tuned for a preferred playing style. Raising `hand_shift` favours fewer position changes; raising `string_travel` favours staying on nearby strings.

## Transcription

Monophonic mode tracks the fundamental with pYIN, then segments on pitch changes and re-attacks. Three things it does beyond a textbook implementation, each of which fixed a real failure:

- Confidence is pitch stability, the fraction of frames agreeing on the note. pYIN's own voicing probability is unreliable on plucked strings and discards correct notes.
- Onsets are refined against a short window energy derivative. The pitch analysis window is 93 ms long and straddles the attack, which puts raw segment starts about 30 ms early.
- Pitch excursions with no envelope rise are absorbed into the note before them. A dip to the neighbouring semitone inside a held note is a tracker artifact, and so is the semitone passed through while one note decays into the next. Set `strip_continuations=False` for legato playing where hammer-ons arrive without a fresh attack.

The application always uses the monophonic path. If several notes overlap in the source, the strongest/confident pitch is retained and overlaps are trimmed so every output note can be played one at a time.

## Limits

- One note at a time per string. Bends, slides, vibrato and palm muting are not modelled.
- Tempo comes from `librosa.beat.beat_track` and can land on half or double time. Quantisation only snaps a note when the grid is within a third of a beat, so a bad tempo degrades timing rather than destroying it. Pass `--quantize 0` for raw timing.
- Onset-poor material -- a hummed melody has no percussive attack for the beat tracker to lock onto -- can make `beat_track` report 0 bpm. That falls back to an assumed 120 bpm grid for quantisation rather than collapsing every note onto the same instant; `timing.tempo_bpm` in the output is then a placeholder, not a measurement, so don't read it as the performer's actual tempo.
- Notes outside the instrument range are folded by octaves into it rather than dropped.

## Layout

```
tunefinder/
  config.py       tunings, instrument limits, and fretting-path cost weights
  types.py        NoteEvent, Placement, Shape, ArrangedGroup, Transcription
  media.py       local video/audio and YouTube URL decoding
  transcribe.py   pitch tracking, humming cleanup, onset segmentation, tempo, StreamRecorder
  timing.py       beat mapping, quantisation, octave folding
  fretboard.py    candidate positions, shape enumeration, fingering, barres
  arrange.py      Viterbi search over the fretting path
  document.py     timed note JSON and per-note string/fret data
  tabtext.py      ASCII tab, event table, STRING-FRET stringtab
  synth.py        test signals and plucked string playback
  cli.py          command line
  ui.py           start/stop recording window
schema/tunefinder.schema.json
tests/test_pipeline.py
```

## Tests

```bash
python tests/test_pipeline.py      # or: python -m pytest tests/ -q
```

Tests cover fretboard geometry, monophonic simplification, media decoding, humming cleanup, arrangement behaviour, and end-to-end pitch and onset accuracy.
