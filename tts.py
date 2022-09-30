from gtts import gTTS
from playsound import playsound

mytext = 'Welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?'
# mytext = "Welkom bij het UU restaurantsysteem! U kunt naar restaurants vragen op gebied, prijsklasse of voedseltype. Hoe kan ik u helpen?"
language = 'it'
# language = "nl"

myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("./audio/welcome.mp3")

playsound('./audio/welcome.mp3')
