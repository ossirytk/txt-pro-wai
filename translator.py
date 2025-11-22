from deep_translator import GoogleTranslator
from deep_translator import MyMemoryTranslator

text = "In accordance with the third explanation(above), we mourn because of the continual conflict between our flesh and spirit[which frustrates our desire to be restored to the dignity of the first man], yet it is impossible for the soul not to be wounded at least by venial sins due to the flesh."
#
translated_google = GoogleTranslator(source='en', target='fi').translate(text=text)
translated_my_memory = MyMemoryTranslator(source='en-GB', target='fi-FI').translate(text=text)

print(text)
print(translated_google)
print(translated_my_memory)
