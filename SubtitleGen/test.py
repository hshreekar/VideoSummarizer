import speech_recognition as sr

r=sr.Recognizer()

audioFile = sr.AudioFile('harvard.wav')
with audioFile as source:
    audio = r.record(source)
WIT_API_KEY="RNK7AMKUTDNBMBRC3OKYHHY3DKHPWCWP"
result=r.recognize_wit(audio,WIT_API_KEY)

print(result)