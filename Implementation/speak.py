import pyttsx3
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

text = "There is a person 2 meters in front of you"
speak(text)