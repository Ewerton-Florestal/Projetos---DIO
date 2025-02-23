import pyttsx3
import os
import speech_recognition as sr
import webbrowser
import requests
import json

# Módulo de Texto para Fala (Text to Speech) com pyttsx3
def text_to_speech(text, lang='pt-br'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Controle de velocidade
    engine.setProperty('volume', 1)  # Volume (0.0 a 1.0)
    engine.say(text)
    engine.runAndWait()

# Módulo de Fala para Texto (Speech to Text)
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    # Carregar o arquivo de áudio
    with sr.AudioFile(audio_file) as source:
        try:
            audio = recognizer.record(source)  # Captura o áudio do arquivo
            print("Reconhecendo...")
            text = recognizer.recognize_google(audio, language="pt-BR")
            print(f"Você disse: {text}")
            return text
        except sr.UnknownValueError:
            print("Desculpe, não entendi o que você disse.")
        except sr.RequestError as e:
            print(f"Erro ao se conectar ao serviço de reconhecimento de fala; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return None

# Módulo de Comandos de Voz
def execute_command(command):
    if "wikipedia" in command:
        webbrowser.open("https://pt.wikipedia.org")
        return "Abrindo Wikipedia."
    elif "youtube" in command:
        webbrowser.open("https://www.youtube.com")
        return "Abrindo YouTube."
    else:
        return "Desculpe, não entendi o comando."

# Função principal da assistência virtual
def virtual_assistant():
    while True:
        print("Aguardando comando...")
        # Caminho para o arquivo de áudio gravado
        audio_file = "audio_comando.wav"  # Insira o caminho do seu arquivo de áudio
        command = speech_to_text(audio_file)  # Captura o comando de voz a partir do arquivo
        if command:
            response = execute_command(command.lower())  # Executa o comando
            text_to_speech(response)  # Responde com áudio

# Iniciar a assistência virtual
if __name__ == "__main__":
    virtual_assistant()