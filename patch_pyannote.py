import pathlib
io = pathlib.Path('/app/.venv/lib/python3.11/site-packages/pyannote/audio/core/io.py')
txt = io.read_text()
txt = txt.replace('import torchaudio', 'try:\n    import torchaudio\nexcept Exception:\n    torchaudio = None')
txt = txt.replace(') -> torchaudio.AudioMetaData:', ') -> object:')
txt = txt.replace('torchaudio.list_audio_backends()', 'getattr(torchaudio, "list_audio_backends", lambda: [])()')
io.write_text(txt)
print("Patched pyannote io.py successfully")
