from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import tensorflow as tf
import gensim.downloader as api

model = tf.keras.models.load_model('ner_service/model.h5')

embedding_model = api.load('glove-twitter-50')

def vectorize_tokens(tokens, embedding_model):
    token_vectors = []
    for token in tokens:
        if token in embedding_model:
            token_vectors.append(embedding_model[token])
        else:
            token_vectors.append(np.zeros(embedding_model.vector_size))
    return token_vectors

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        tokens = data.get('tokens', [])

        if not tokens:
            return JsonResponse({'error': 'No tokens provided'}, status=400)

        tokens = [token.lower() for token in tokens]  # Convert to lower case
        token_vectors = vectorize_tokens(tokens, embedding_model)
        token_vectors = np.asarray(token_vectors).reshape(1, -1)

        predictions_prob = model.predict(token_vectors)
        predictions = np.argmax(predictions_prob, axis=1)

        # Map predictions to tags
        ner_tag_encoding = {1: "B-O", 2: "B-AC", 3: "B-LF", 4: "I-LF"}
        decode_ner_tag = {v: k for k, v in ner_tag_encoding.items()}
        predicted_tags = [decode_ner_tag[tag] for tag in predictions]

        return JsonResponse({'tokens': tokens, 'tags': predicted_tags})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)