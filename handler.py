import json
import requests

class TextToSpeech:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_audio(self, text_chunk):
        response = requests.post(
            'https://api.runpod.io/generate',
            headers={'Authorization': f'Bearer {self.api_key}'},
            json={'text': text_chunk}
        )
        return response.json()

def handler(event, context):
    api_key = event['api_key']
    text_chunks = event['text_chunks']
    tts = TextToSpeech(api_key)

    audio_files = []
    for chunk in text_chunks:
        audio_response = tts.generate_audio(chunk)
        audio_files.append(audio_response)

    return {
        'statusCode': 200,
        'body': json.dumps(audio_files)
    }