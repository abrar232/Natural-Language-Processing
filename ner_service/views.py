import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import logging
from datetime import datetime
import tensorflow as tf

def load_glove_vectors(glove_file_path):
    glove_vectors = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_vectors[word] = vector
    return glove_vectors

# Path to the GloVe file
glove_file_path = r'C:\Users\abrar\nlp_deploymentV2\ner_service\glove.6B.50d.txt'

# Load GloVe vectors
glove_vectors = load_glove_vectors(glove_file_path)

def vectorize_tokens(tokens, glove_vectors):
    token_vectors = []
    for token in tokens:
        if token in glove_vectors:
            token_vectors.append(glove_vectors[token])
        else:
            token_vectors.append(np.zeros(50))  # Assuming 50-dimensional GloVe vectors
    return token_vectors

# Load your trained model
model = tf.keras.models.load_model(r'C:\Users\abrar\nlp_deploymentV2\ner_service\model.h5')

# Set up logging
logger = logging.getLogger('django')

def log_interaction(user_input, model_prediction):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - Input: {user_input} - Prediction: {model_prediction}"
    logger.debug(log_message)

@csrf_exempt
def predict(request):
    if request.method == "POST":
        data = json.loads(request.body)
        tokens = data.get('tokens')

        # Transform tokens into vectors
        token_vectors = np.array(vectorize_tokens(tokens, glove_vectors))

        # Make predictions
        predictions_prob = model.predict(token_vectors)
        predictions = np.argmax(predictions_prob, axis=1)

        # Map predictions to tags
        ner_tag_encoding = {"B-O": 1, "B-AC": 2, "B-LF": 3, "I-LF": 4}
        decode_ner_tag = {v: k for k, v in ner_tag_encoding.items()}

        # Print statements for debugging
        print("Predictions:", predictions)
        print("NER Tag Encoding:", ner_tag_encoding)
        print("Decode NER Tag:", decode_ner_tag)

        try:
            predicted_tags = [decode_ner_tag.get(tag, "Unknown") for tag in predictions]
        except KeyError as e:
            print("KeyError:", e)
            return JsonResponse({'error': f'KeyError: {e}'}, status=400)

        # Log the interaction
        log_interaction(tokens, predicted_tags)

        return JsonResponse({'tokens': tokens, 'predictions': predicted_tags})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)