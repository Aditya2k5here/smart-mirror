from voice.conversation_service import conversation_service
from voice.speech_service import SpeechService


def init_openai():
    conversation_service._init_openai_client()


def get_gpt_response(prompt, memory_context):
    return conversation_service.respond(prompt, "Unknown", memory_context)


def listen_in_background(on_command_callback):
    service = SpeechService(on_command_callback)
    service.start()
    return service

