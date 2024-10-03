import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy import array

# Load data from JSONL file
def load_data(filepath):
    pairs = []
    with open(filepath, 'r') as file:
        for line in file:
            entry = json.loads(line)
            pairs.append((entry['acronym'], entry['definition']))
    return pairs

# Create tokenizer  
def create_tokenizer(pairs):
    tokenizer = Tokenizer(filters='', oov_token='<OOV>')
    tokenizer.fit_on_texts([item for pair in pairs for item in pair])
    return tokenizer

# Prepare data for the model
def preprocess_data(pairs, tokenizer, max_length):
    input_texts = [pair[0] for pair in pairs]
    target_texts = ['\t' + pair[1] + '\n' for pair in pairs]  # Start and end tokens
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)
    input_padded = pad_sequences(input_sequences, maxlen=max_length, padding='post')
    target_padded = pad_sequences(target_sequences, maxlen=max_length, padding='post')
    return input_padded, target_padded

# Define the NMT model
def define_model(tokenizer, max_length):
    num_tokens = len(tokenizer.word_index) + 1
    embed_size = 50

    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(num_tokens, embed_size)(encoder_inputs)
    encoder_lstm = LSTM(256, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(num_tokens, embed_size)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(num_tokens, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main execution block
if __name__ == '__main__':
    filepath = 'C:/Users/mic/Documents/perbinusian/semester 4/RM/processed_acronyms.jsonl'
    # Please make sure that the filepath is according to your folder location, use processed_acronyms.jsonl
    pairs = load_data(filepath)
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    tokenizer = create_tokenizer(pairs)
    max_length = max(len(pair[0]) + len(pair[1]) + 2 for pair in pairs)

    input_train, target_train = preprocess_data(train_pairs, tokenizer, max_length)
    input_test, target_test = preprocess_data(test_pairs, tokenizer, max_length)

    model = define_model(tokenizer, max_length)
    model.summary()

    # Train model
    model.fit([input_train, target_train[:, :-1]], target_train[:, 1:, None],
              batch_size=64, epochs=30, validation_split=0.2)

    # Evaluate model
    loss, accuracy = model.evaluate([input_test, target_test[:, :-1]], target_test[:, 1:, None])
    print(f'Test Accuracy: {accuracy:.5f}')