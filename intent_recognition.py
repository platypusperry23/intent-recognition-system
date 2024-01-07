import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import spacy
from textblob import TextBlob as tb


nlp = spacy.load("en_core_web_sm")
remove_from_stop = ['in', 'IN', 'AM', 'PM', 'am', 'pm']
for text in remove_from_stop:
    if text in nlp.Defaults.stop_words:
        nlp.Defaults.stop_words.remove(text)

def clean(text):
    if text == "hi how are you":
        return text
    doc = nlp(text)
    text = re.sub(r'\bcheck(\s+in)?\b', 'check-in', text)
    text = re.sub(r'\bcheck(\s+out)?\b', 'check-out', text)
    filtered_tokens = [token.text for token in doc if not (token.is_punct or token.is_stop)]
    if not filtered_tokens:
        return text
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text

# loading dataset

with open('/Users/vatsal.rai/Projects/intent/intents.json', 'r') as f:
    data = json.load(f)
print('Learning')
intents = []
unique_intents = []
text_input = []
for intent in data['intents']:
    if intent['intent'] not in unique_intents:
        unique_intents.append(intent['intent'])
    for text in intent['text']:
        text_input.append(clean(text))
        intents.append(intent['intent'])

intent_to_index = {}
categorical_target = []
index = 0

for intent in intents:
    if intent not in intent_to_index:
        intent_to_index[intent] = index
        index += 1
    categorical_target.append(intent_to_index[intent])

num_classes = len(intent_to_index)

# Convert intent_to_index to index_to_intent
index_to_intent = {index: intent for intent, index in intent_to_index.items()}

tokenizer = Tokenizer(filters='', oov_token='<unk>')
tokenizer.fit_on_texts(text_input)
sequences = tokenizer.texts_to_sequences(text_input)
padded_sequences = pad_sequences(sequences, padding='pre')

categorical_vec = tf.keras.utils.to_categorical(categorical_target,
                                                num_classes=num_classes, dtype='int32')
epochs = 100
embed_dim = 300
lstm_num = 50
output_dim = categorical_vec.shape[1]
input_dim = len(unique_intents)
print("Building Model")
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, embed_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_num, dropout=0.1)),
    tf.keras.layers.Dense(lstm_num, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(output_dim, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(padded_sequences, categorical_vec, epochs=epochs, verbose=0)

test_text_inputs = ["arriving at 9 am", "I just want to contact hotel", 
                    "ariving by 9 am", "Can I come in after noon?", "how are you",
                    "running late need more time","Plz confirm booking"
                    ]

test_intents = ["early check-in", 
                "contact hotel", 
                "early check-in",
                "late check-in",
                "greeting",
                "late check-in",
                "booking-confirmation",
                ]

test_sequences = tokenizer.texts_to_sequences(test_text_inputs)
test_padded_sequences = pad_sequences(test_sequences, padding='pre')
test_labels = np.array([unique_intents.index(intent) for intent in test_intents])
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
loss, accuracy = model.evaluate(test_padded_sequences, test_labels)
print(f'Bot Accurracy {accuracy}, loss {loss}')

def i_response(sentence):
    try:
        test_sequence = tokenizer.texts_to_sequences([sentence])
        test_padded_sequence = pad_sequences(test_sequence, padding='pre')
        threshold = 0.5
        preds = model.predict(test_padded_sequence)
        print("predictions:", preds)
        if np.max(preds) > threshold:  
            predicted_intent = index_to_intent[np.argmax(preds)]
            return predicted_intent
        else:
            return 'Not Recognizable'
    except Exception as e:
        print(e)

print("Note: Model Ready----Enter 'quit' to break the loop.")
while True:
    input_str = input('You: ')
    if input_str.lower() == 'quit':
        break
    input_str = str(tb(input_str.lower()).correct())
    input_str = ' '.join(input_str.split())
    print(input_str)
    if input_str:
        typs = i_response(input_str)
        print('Predicted Intent {}'.format(typs))
    else:
        print('Not Recognizable')
    print()
