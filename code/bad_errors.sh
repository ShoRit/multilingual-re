#!/bin/bash



python3 relextract.py --src_lang it --tgt_lang en --model_name mbert-base --dep_model stanza --dep 1 --mode predict
python3 relextract.py --src_lang it --tgt_lang ar --model_name mbert-base --dep_model stanza --dep 1 --mode predict
python3 relextract.py --src_lang it --tgt_lang it --model_name mbert-base --dep_model stanza --dep 1 --mode predict
python3 relextract.py --src_lang it --tgt_lang de --model_name mbert-base --dep_model stanza --dep 1 --mode predict
python3 relextract.py --src_lang it --tgt_lang es --model_name mbert-base --dep_model stanza --dep 1 --mode predict
python3 relextract.py --src_lang it --tgt_lang fr --model_name mbert-base --dep_model stanza --dep 1 --mode predict
python3 relextract.py --src_lang it --tgt_lang zh --model_name mbert-base --dep_model stanza --dep 1 --mode predict


