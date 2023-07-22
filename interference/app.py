import time
import json
import random
import sys
from flask import Flask, request
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
from waitress import serve
from concurrent.futures import ThreadPoolExecutor

from g4f import ChatCompletion, Provider

app = Flask(__name__)
CORS(app)

model = SentenceTransformer('all-mpnet-base-v2')
executor = ThreadPoolExecutor(max_workers=4)

def get_embeddings(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

@app.route("/embeddings", methods=['POST'])
def handle_embeddings_request():
    data = request.get_json()
    input_text = data.get('input', '')
    model_name = data.get('model')

    future = executor.submit(get_embeddings, input_text)
    embeddings = future.result()

    return {
        'data': [
            {
                'embedding': embeddings.tolist(),
                'index': 0,
                'object': 'embedding'
            }
        ],
        'model': model_name,
        'object': 'list',
        'usage': {
            'prompt_tokens': 5,
            'total_tokens': 5
        }
    }

@app.route("/v1/models", methods=['GET'])
def models():
    return {
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "owned_by": "openai",
                "permission": ""
            },
            {
                "id": "gpt-4",
                "object": "model",
                "owned_by": "openai",
                "permission": ""
            }
        ],
        "object": "list"
    }

def make_chat_completion(model, streaming, messages):
    return ChatCompletion.create(model=model, provider=Provider.Bing, stream=streaming, messages=messages)


def create_response_object(response, model):
    completion_data = generate_completion_data(response)

    return {
        'id': f'chatcmpl-{completion_data["id"]}',
        'object': 'chat.completion',
        'created': completion_data["timestamp"],
        'model': model,
        'usage': {
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None
        },
        'choices': [{
            'message': {
                'role': 'assistant',
                'content': response
            },
            'finish_reason': 'stop',
            'index': 0
        }]
    }


def stream_generator(response):
    for token in response:
        completion_data = generate_completion_data(token)

        yield 'data: %s\n\n' % json.dumps(completion_data, separators=(',', ':'))
        time.sleep(0.1)

    completion_data = generate_completion_data(None, finish_reason="stop")

    yield 'data: %s\n\n' % json.dumps(completion_data, separators=(',', ':'))

def generate_completion_data(content, finish_reason=None):
    completion_timestamp = int(time.time())
    completion_id = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=28))

    completion_data = {
        'id': completion_id,
        'timestamp': completion_timestamp,
        'model': 'gpt-3.5-turbo-0301',
        'choices': [
            {
                'delta': {
                    'content': content
                },
                'index': 0,
                'finish_reason': finish_reason
            }
        ]
    }

    return completion_data

@app.route("/chat/completions", methods=['POST'])
@app.route("/v1/chat/completions", methods=['POST'])
def chat_completions():
    streaming = request.json.get('stream', False)
    model = request.json.get('model', 'gpt-3.5-turbo')
    messages = request.json.get('messages')
    
    try:
        response = make_chat_completion(model, streaming, messages)
    except Exception as e:
        print("Error during ChatCompletion.create:", str(e))
        return "Internal Server Error", 500
    
    if not streaming:
        while 'curl_cffi.requests.errors.RequestsError' in response:
            try:
                response = make_chat_completion(model, streaming, messages)
            except Exception as e:
                print("Error during ChatCompletion.create:", str(e))
                return "Internal Server Error", 500

        return create_response_object(response, model)

    return app.response_class(stream_generator(response), mimetype='text/event-stream')

if __name__ == '__main__':
    if "--prod" in sys.argv:
        serve(app, host='0.0.0.0', port=1337)
    else:
        app.run(host='0.0.0.0', port=1337, debug=True)
