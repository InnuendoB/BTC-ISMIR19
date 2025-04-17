'''
File: datasetGen.py
Brief: 
Created Date: Wednesday April 9th 2025
Author: Innuendo
Initials: WCF
-----
Last Modified: Wednesday April 9th 2025 10:16:09 am
Modified By: the developer formerly known as Innuendo at <wangcefeng@hotone.com>
-----
Copyright (c) 2025 Changsha Hotone Audio Co.,LTD 
'''

import random
import mido
from mido import MidiFile, MidiTrack, Message, bpm2tempo
import os

# === 可配置参数 ===
BPM = 100
NUM_BARS = 64
TICKS_PER_BEAT = 480
CHORD_DURATION_BEATS = 4
USE_SEVENTHS_RATIO = 0.6
NUM_SAMPLES = 20  # 批量数量
OUTPUT_DIR = 'output_chords'

# 和弦配置
ROOT_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ROOT_TO_SEMITONE = {n: i for i, n in enumerate(ROOT_LIST)}
CHORD_INTERVALS = {
    'maj':      [0, 4, 7],
    'min':      [0, 3, 7],
    'dim':      [0, 3, 6],
    'aug':      [0, 4, 8],
    'sus2':     [0, 2, 7],
    'sus4':     [0, 5, 7],
    'maj6':     [0, 4, 7, 9],
    'min6':     [0, 3, 7, 9],
    '7':        [0, 4, 7, 10],
    'maj7':     [0, 4, 7, 11],
    'min7':     [0, 3, 7, 10],
    'minmaj7':  [0, 3, 7, 11],
    'dim7':     [0, 3, 6, 9],
    'hdim7':    [0, 3, 6, 10],
}

# 调式结构
MAJOR_MODE = {
    'triads':  ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim'],
    'sevenths': ['maj7', 'min7', 'min7', 'maj7', '7', 'min7', 'hdim7'],
}
MINOR_MODE = {
    'triads':  ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj'],
    'sevenths': ['min7', 'hdim7', 'maj7', 'min7', 'min7', 'maj7', '7'],
}

def build_chord(root: str, quality: str, base_octave=4):
    root_note = 12 * (base_octave + 1) + ROOT_TO_SEMITONE[root]
    intervals = CHORD_INTERVALS[quality]
    return [root_note + i for i in intervals]

def get_scale_degrees(key: str, mode: str):
    tonic_index = ROOT_TO_SEMITONE[key]
    scale = [(tonic_index + i) % 12 for i in [0, 2, 4, 5, 7, 9, 11]]
    return [ROOT_LIST[i] for i in scale]

def generate_chord_sequence(key: str, mode: str, num_bars: int, use_sevenths_ratio=0.6):
    mode_data = MAJOR_MODE if mode == 'major' else MINOR_MODE
    scale_roots = get_scale_degrees(key, mode)
    chords = []
    for i in range(num_bars):
        degree = random.randint(0, 6)
        root = scale_roots[degree]
        quality_list = mode_data['sevenths'] if random.random() < use_sevenths_ratio else mode_data['triads']
        quality = quality_list[degree]
        chords.append((root, quality))
    return chords

def add_chord(track, notes, duration_ticks, velocity=80, arpeggio=False):
    if arpeggio:
        # 每拍 = 1/4，小节 4 拍 = 8 个八分音符，每个音 = 1/8 拍
        step = duration_ticks // 8
        if len(notes) == 3:
            # 1 3 5 3 1 1^ 3 5
            seq = [0, 1, 2, 1, 0, 'octave', 1, 2]
        elif len(notes) == 4:
            # 1 3 5 7 3 5 1 3
            seq = [0, 1, 2, 3, 1, 2, 0, 1]
        else:
            raise ValueError(f"Unsupported chord size: {len(notes)}")

        for i, s in enumerate(seq):
            if s == 'octave':
                note = notes[0] + 12
            else:
                note = notes[s]
            track.append(Message('note_on', note=note, velocity=velocity, time=0 if i > 0 else 0))
            track.append(Message('note_off', note=note, velocity=0, time=step))
    else:
        # Block chord: 所有 note_on 同时，note_off 一起在 duration 之后
        for i, note in enumerate(notes):
            track.append(Message('note_on', note=note, velocity=velocity, time=0 if i > 0 else 0))
        for i, note in enumerate(notes):
            track.append(Message('note_off', note=note, velocity=0, time=duration_ticks if i == 0 else 0))

def format_chord_label(root, quality):
    return f"{root}{quality}"

def generate_one_sample(index: int):
    key = random.choice(ROOT_LIST)
    mode = random.choice(['major', 'minor'])
    chords = generate_chord_sequence(key, mode, NUM_BARS, USE_SEVENTHS_RATIO)

    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=bpm2tempo(BPM)))

    ticks_per_chord = CHORD_DURATION_BEATS * TICKS_PER_BEAT

    midi_path = os.path.join(OUTPUT_DIR, f'chords_{index:03d}.mid')
    txt_path = os.path.join(OUTPUT_DIR, f'chords_{index:03d}.txt')

    with open(txt_path, 'w') as f:
        for i, (root, quality) in enumerate(chords):
            notes = build_chord(root, quality)
            arpeggio = random.random() < 0.2
            add_chord(track, notes, ticks_per_chord, arpeggio=arpeggio)
            label = format_chord_label(root, quality)
            f.write(f"{i+1}: {label}\n")

    mid.save(midi_path)
    print(f"Saved: {midi_path} / {txt_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(NUM_SAMPLES):
        generate_one_sample(i)

if __name__ == '__main__':
    main()
