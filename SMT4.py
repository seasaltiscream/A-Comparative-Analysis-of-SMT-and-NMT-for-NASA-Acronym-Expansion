import nltk
import json
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def load_data(filepath):
    with open(filepath, 'r') as file:
        pairs = [json.loads(line) for line in file]
    return [(entry['acronym'], entry['definition']) for entry in pairs]

def generate_acronym_from_definition(definition):
    words = nltk.word_tokenize(definition)
    return ''.join(word[0].upper() for word in words if word.isalpha())

def acronym_match_rate(test_pairs):
    correct = 0
    for acronym, definition in test_pairs:
        generated_acronym = generate_acronym_from_definition(definition)
        if generated_acronym == acronym:
            correct += 1
    return correct / len(test_pairs) if test_pairs else 0

def main(filepath):
    pairs = load_data(filepath)
    _, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    match_rate = acronym_match_rate(test_pairs)
    print(f'Acronym Match Rate: {match_rate:.5f}')

if __name__ == "__main__":
    filepath = 'C:/Users/mic/Documents/perbinusian/semester 4/RM/processed_acronyms.jsonl'
     # Please make sure that the filepath is according to your folder location, use processed_acronyms.jsonl
    main(filepath)