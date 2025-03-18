import json

no_prompt="/media/rtml/ext_drive/home/rtml/shounak_research/nlp/src/ritam_repo/multilingual-re/prompting_predictions/redfm-en-mistral_zshot_prompt-None-test.json"
with_prompt="/media/rtml/ext_drive/home/rtml/shounak_research/nlp/src/ritam_repo/multilingual-re/prompting_predictions/redfm-en-mistral_zshot_prompt-trankit-test-better_prompt.json"

with open(no_prompt, "r") as f:
    no_prompt_data = json.load(f)
with open(with_prompt, "r") as f:
    with_prompt_data = json.load(f)

print(len(no_prompt_data))
print(len(with_prompt_data))

for i, data in enumerate(with_prompt_data):
    if data["true_label"] in data["gen_text"] and data["true_label"] not in no_prompt_data[i]["gen_text"]:
        print(data["true_label"])
        print(data["gen_text"])
        print(no_prompt_data[i]["gen_text"])

