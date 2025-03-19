import numpy as np
import json
import os
import csv
from collections import defaultdict as ddict

# Get list of json files
json_files = sorted(os.listdir('../prompting_predictions_improve_dag/'))

# Prepare CSV header
csv_file = 'metrics_output.csv'
csv_columns = ['Filename', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Macro F1 Score']


def filter_text(generated_text, dataset_name):
	filtered_text = generated_text

	if dataset_name == 'redfm':
		filtered_text = filtered_text.replace('0: country\n1: place of birth \n2: spouse\n3: country of citizenship\n4: instance of\n5: capital\n6: child\n7: shares border with\n8: author\n9: director\n10: occupation\n11: founded by\n12: league\n13: owned by\n14: genre\n15: named after\n16: follows\n17: headquarters location\n18: cast member\n19: manufacturer\n20: located in or next to body of water\n21: location\n22: part of\n23: mouth of the watercourse\n24: member of\n25: sport \n26: characters\n27: participant\n28: notable work\n29: replaces\n30: sibling\n31: inception\n','')

	if dataset_name == 'indore':
		filtered_text = filtered_text.replace('0: manner_of_death\n1: position_held\n2: child\n3: producer\n4: contains_administrative_territorial_entity\n5: ethnic_group\n6: member_of_political_party\n7: member_of_sports_team\n8: founded_by\n9: mother\n10: employer\n11: educated_at\n12: present_in_work\n13: participant\n14: tributary\n15: place_of_death\n16: publisher\n17: creator\n18: member_of\n19: subsidiary\n20: owned_by\n21: parent_organization\n22: material_used\n23: writing_system\n24: named_after\n25: genre\n26: performer\n27: composer\n28: continent\n29: color\n30: discoverer_or_inventor\n31: director\n32: screenwriter\n33: shares_border_with\n34: place_of_birth\n35: award_received\n36: nominated_for\n37: production_company\n38: located_in_or_next_to_body_of_water\n39: occupation\n40: student_of\n41: capital_of\n42: spouse\n43: country_of_citizenship\n44: father\n45: capital\n46: winner\n47: league\n48: sibling\n49: original_language_of_film_or_TV_show\n50: sport\n', '')

	return filtered_text



# Open CSV file to write metrics
with open(csv_file, 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(csv_columns)  # Write the header

	for file in json_files:
		true_labels = []
		predicted_labels = []
		try:
			with open(f'../prompting_predictions_improve_dag/{file}', 'r') as f:
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

							#print(row['gen_text'])

							filtered_gen_text = filter_text(row['gen_text'], dataset)


							if lbl in filtered_gen_text:
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
		except:
			pass
        # Print results (optional)
        # print("Filename: ", file)
        # print("Accuracy: ", acc)
        # print(f"Precision: {micro_prec:.4f}")
        # print(f"Recall: {micro_rec:.4f}")
        # print(f"F1 Score: {micro_f1:.4f}")
        # print(f"Macro F1 Score: {macro_f1:.4f}")
