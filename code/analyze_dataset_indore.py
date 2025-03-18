import json
import statistics
import re

def count_data_points_and_analyze(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    all_doc_text_word_counts = []
    all_lexical_distances = []
    
    for dataset in ['train', 'validation', 'test']:
        dataset_points = data.get(dataset, [])
        print(f"{dataset.capitalize()} set: {len(dataset_points)} data points")
        
        for point in dataset_points:
            doc_text = point.get('doc_text', '')
            word_count = len(doc_text.split())
            all_doc_text_word_counts.append(word_count)
            
            orig_sent = point.get('orig_sent', '')
            lexical_distance = calculate_lexical_distance(orig_sent)
            all_lexical_distances.append(lexical_distance)
    
    total_points = len(all_doc_text_word_counts)
    
    print(f"Total number of data points: {total_points}")
    
    if all_doc_text_word_counts:
        mean_words = statistics.mean(all_doc_text_word_counts)
        median_words = statistics.median(all_doc_text_word_counts)
        print(f"Mean number of words in 'doc_text': {mean_words:.2f}")
        print(f"Median number of words in 'doc_text': {median_words:.2f}")
    else:
        print("No 'doc_text' data found in the dataset.")
    
    if all_lexical_distances:
        mean_distance = statistics.mean(all_lexical_distances)
        median_distance = statistics.median(all_lexical_distances)
        print(f"Mean lexical distance: {mean_distance:.2f}")
        print(f"Median lexical distance: {median_distance:.2f}")
    else:
        print("No lexical distances could be calculated.")

def calculate_lexical_distance(sentence):
    e1_start = sentence.find('<e1>')
    e1_end = sentence.find('</e1>')
    e2_start = sentence.find('<e2>')
    e2_end = sentence.find('</e2>')
    
    if e1_start == -1 or e1_end == -1 or e2_start == -1 or e2_end == -1:
        return None
    
    # Determine which entity comes first
    if e1_start < e2_start:
        start = e1_end + 5  # +5 to skip '</e1>'
        end = e2_start
    else:
        start = e2_end + 5  # +5 to skip '</e2>'
        end = e1_start
    
    between_text = sentence[start:end]
    words_between = len(re.findall(r'\S+', between_text))
    return words_between

# Usage
for lang in ['en', 'hi', 'te']:
    for parse in ['stanza']:
        print(f"Analyzing dataset for {lang} with parse {parse}")
        json_file_path = f'../data/indore/{lang}_prompt_{parse}.json'
        count_data_points_and_analyze(json_file_path)
