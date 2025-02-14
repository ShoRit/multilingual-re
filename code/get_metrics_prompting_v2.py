import numpy as np
import json
import os
import csv
from collections import defaultdict as ddict

# Get list of json files
json_files = sorted(os.listdir('../prompting_predictions/'))

# Prepare CSV header
csv_file = 'metrics_output.csv'
csv_columns = ['Filename', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Macro F1 Score']

# Open CSV file to write metrics
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)  # Write the header
    
    for file in json_files:
        true_labels = []
        predicted_labels = []

        with open(f'../prompting_predictions/{file}', 'r') as f:
            data = json.load(f)

            if(len(data) != 0):
                dataset = file.split('-')[0]
                relation_data_file = f"../data/{dataset}/relation_dict.json"
                with open(relation_data_file, 'r') as f:
                    relation_data = json.load(f)
                    labels = list(relation_data.values())

                metrics = ddict(lambda: ddict(int))
                global_scores = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

                for idx, row in enumerate(data):
                    # y_pred, y_true, text, answer, label
                    gold_lbl = row['true_label']

                    for lbl in labels:
                        pred_ans = 0
                        gold_ans = 0

                        if lbl in row['gen_text']:
                            pred_ans = 1
                        if lbl == gold_lbl:
                            gold_ans = 1

                        if pred_ans == 0 and gold_ans == 0:
                            metrics[gold_lbl]['TN'] += 1
                            global_scores['TN'] += 1
                            continue
                        else:
                            if pred_ans == 1 and gold_ans == 1:
                                metrics[gold_lbl]['TP'] += 1
                                global_scores['TP'] += 1
                            elif pred_ans == 1 and gold_ans == 0:
                                metrics[gold_lbl]['FP'] += 1
                                global_scores['FP'] += 1
                            elif pred_ans == 0 and gold_ans == 1:
                                metrics[gold_lbl]['FN'] += 1
                                global_scores['FN'] += 1

                macro_F1 = []

                for lbl in metrics:
                    try:
                        prec = metrics[lbl]['TP'] / (metrics[lbl]['TP'] + metrics[lbl]['FP'])
                    except Exception as e:
                        prec = 0.0

                    try:
                        rec = metrics[lbl]['TP'] / (metrics[lbl]['TP'] + metrics[lbl]['FN'])
                    except Exception as e:
                        rec = 0.0

                    try:
                        f1 = (2 * prec * rec) / (prec + rec)
                    except Exception as e:
                        f1 = 0.0

                    macro_F1.append(f1)

                try:
                    micro_rec = global_scores['TP'] / (global_scores['TP'] + global_scores['FN'])
                except Exception as e:
                    micro_rec = 0.0

                try:
                    micro_prec = global_scores['TP'] / (global_scores['TP'] + global_scores['FP'])
                except Exception as e:
                    micro_prec = 0.0

                try:
                    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                except Exception as e:
                    micro_f1 = 0.0

                try:
                    acc = (global_scores['TP'] + global_scores['TN']) / (
                        global_scores['TP'] + global_scores['TN'] + global_scores['FP'] + global_scores['FN'])
                except:
                    acc = 0.0
                    print(file)
                macro_f1 = np.mean(macro_F1)

                # Write the results for each file to the CSV
                writer.writerow([file, acc, micro_prec, micro_rec, micro_f1, macro_f1])

        # Print results (optional)
        # print("Filename: ", file)
        # print("Accuracy: ", acc)
        # print(f"Precision: {micro_prec:.4f}")
        # print(f"Recall: {micro_rec:.4f}")
        # print(f"F1 Score: {micro_f1:.4f}")
        # print(f"Macro F1 Score: {macro_f1:.4f}")
