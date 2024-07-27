import json
import time
import math
import librosa
import zipfile
import os
import essentia.standard as es

def detect_bpm(audio_file):
    audio = es.MonoLoader(filename=audio_file)()
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    return round(bpm)

def seconds_to_beat(seconds, bpm):
    alpha = 60 / bpm
    beat = seconds / alpha
    measure = math.floor(beat)
    split = 8
    sub_beat = (seconds * split) / alpha - measure * split
    
    return [measure, math.floor(sub_beat), math.floor(split)]

def offset_to_mc_beat(offset):
    measure = int(offset)
    beat_fraction = offset - measure
    beat = int(beat_fraction * 4)
    split = 4  # We'll keep using a fixed split of 4
    return [measure, beat, split]

def get_speed_multiplier(offset, sorted_speeds):
    # Find the first speed that has an offset less than or equal to the given offset
    for speed in sorted_speeds:
        if speed.get('offset', 0) <= offset:
            return speed.get('multiplier', 1.0)
    
    # If no matching speed found, return the default multiplier (1.0)
    return 1.0

def convert_beatstar_to_mc(input_json, input_audio, output_mcz):
    # Read BeatStar JSON
    with open(input_json, 'r') as f:
        beatstar_data = json.load(f)

    # Detect BPM
    bpm = detect_bpm(input_audio)

    # Create basic .mc structure
    mc_data = {
        "meta": {
            "creator": "nladuo",
            "version": "Generated with Artificial Intelligence",
            "mode": 0,
            "time": int(time.time()),
            "song": {
                "title": output_mcz,
                "artist": "Void",
            },
            "mode_ext": {
                "column": 4
            }
        },
        "time": [
            {
                "beat": [0, 0, 1],
                "bpm": bpm
            },
        ],
        "extra": {
            "test": {
                "divide": 4,
                "speed": 100,
                "save": 0,
                "lock": 0,
                "edit_mode": 0
            }
        }
    }

    last_note = {
        "beat": [0, 0, 1],
        "sound": input_audio,
        "vol": 100,
        "offset": 0,
        "type": 1
    }

    notes = []

    # Convert notes
    for note in beatstar_data["notes"]:
        if "single" in note:
            offset = note["single"]["note"]["offset"]
            lane = note["single"]["note"]["lane"]
        elif "long" in note:
            offset = note["long"]["note"][0]["offset"]
            lane = note["long"]["note"][0]["lane"]
        elif "switchHold" in note["note"]:
            offset = note["note"]["switchHold"][0]["offset"]
            lane = note["note"]["switchHold"][0]["lane"]
        else:
            print("Unknown note type:", note)
            continue  # Skip unknown note types

        # offset *= get_speed_multiplier(offset, sorted_speeds)
        beat = offset_to_mc_beat(offset)
        column = lane - 1  # Convert lane to column (0-2)

        mc_note = {
            "beat": beat,
            "column": column
        }

        # Handle long notes
        if note["note_type"] in [2, 5]:
            if "long" in note:
                end_offset = note["long"]["note"][1]["offset"]
            elif "switchHold" in note["note"]:
                end_offset = note["note"]["switchHold"][-1]["offset"]
            else:
                end_offset = offset  # Fallback to single note if end not found

            # end_offset *= get_speed_multiplier(end_offset, sorted_speeds)
            end_beat = offset_to_mc_beat(end_offset)
            mc_note["endbeat"] = end_beat

        notes.append(mc_note)

    notes.append(last_note)

    mc_data["note"] = notes

    # Write .mc file
    mc_filename = f"{input_json}.mc"
    with open(mc_filename, 'w') as f:
        json.dump(mc_data, f)

    # Create .mcz file
    with zipfile.ZipFile(output_mcz, 'w') as zipf:
        zipf.write(mc_filename)
        zipf.write(input_audio)

# Usage
input_json = "1127_ImagineD-01102_Enemy-Hard.json"
input_audio = "1127_ImagineD-01102_Enemy-Hard.mp3"
output_mcz = "01102_Enemy3.mcz"

convert_beatstar_to_mc(input_json, input_audio, output_mcz)