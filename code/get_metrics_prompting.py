import json
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Get list of json files
json_files = os.listdir('../prompting_predictions/')

for file in json_files:
    true_labels = []
    predicted_labels = []
    
    with open(f'../prompting_predictions/{file}', 'r') as f:
        data = json.load(f)
        
        dataset = file.split('-')[0]
        relation_data_file = f"../data/{dataset}/relation_dict.json"
        with open(relation_data_file, 'r') as f:
            relation_data = json.load(f)
            labels = list(relation_data.values())
        
        for data_point in data:
            filtered_sentence = ' '.join(re.findall(r'\b(?:' + '|'.join(re.escape(item) for item in labels) + r')\b', data_point['gen_text']))
            
            true_label = data_point['true_label']
            true_labels.append(true_label)
            
            if filtered_sentence:
                predicted_labels.append(filtered_sentence)  # Add prediction if found
            else:
                predicted_labels.append('')  # Empty prediction for false negatives

    # print(true_labels[0:10])
    # print(predicted_labels[0:10])

    y_true_binary = [1] * len(true_labels)  # True labels are all 1 (because they are ground truth)
    y_pred_binary = [1 if true in pred else 0 for true, pred in zip(true_labels, predicted_labels)]

    # Calculate metrics using sklearn
    precision = precision_score(y_true_binary, y_pred_binary, average='macro')
    recall = recall_score(y_true_binary, y_pred_binary, average='macro')
    f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
    
    # Accuracy calculation
    correct = sum([1 for true, pred in zip(true_labels, predicted_labels) if true in pred])
    accuracy = correct / len(true_labels)


    
    print("Filename: ", file)
    print("Accuracy: ", accuracy)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
