import essentia.standard as es

audio = es.MonoLoader(filename='1127_ImagineD-01102_Enemy-Hard.mp3')()

rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

print("BPM:", bpm)
print("Beat positions (sec.):", beats)
print("Beat estimation confidence:", beats_confidence)