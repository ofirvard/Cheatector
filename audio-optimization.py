from pydub import AudioSegment
import os

datadir = "./Raw/"
exportdir = "./Sounds/"

def detect_leading_silence(sound, silence_threshold=-32.5, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def trim_silence(sound):
    duration = len(sound)  
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())  
    return sound[start_trim:duration-end_trim]
    
def to_mono(sound):
    monoSound = sound.set_channels(1)
    return monoSound

if not os.path.isdir(datadir):
	os.makedirs(datadir)
if not os.path.isdir(exportdir):
	os.makedirs(exportdir)
for i, dirname in enumerate(os.listdir(datadir)):
	dirpath = datadir + dirname
	for filename in os.listdir(datadir + dirname):
		filepath = dirpath + "/" + filename
		sound = AudioSegment.from_file(filepath, format="wav")
		sound = trim_silence(sound)
		sound = to_mono(sound)
		exportpath = exportdir + dirname
		if not os.path.isdir(exportpath):
			os.makedirs(exportpath)
		sound.export(exportpath + "/" + filename, format="wav")
