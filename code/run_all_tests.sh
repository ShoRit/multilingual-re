# python run_fshot_prompting.py --src_lang en
# python run_fshot_prompting.py --src_lang hi
# python run_fshot_prompting.py --src_lang te

python run_fshot_prompting.py --src_lang en --dep_parser stanza
python run_fshot_prompting.py --src_lang hi --dep_parser stanza
python run_fshot_prompting.py --src_lang te --dep_parser stanza

python run_fshot_prompting.py --src_lang en --dep_parser trankit
python run_fshot_prompting.py --src_lang hi --dep_parser trankit
python run_fshot_prompting.py --src_lang te --dep_parser trankit


# python run_fshot_prompting.py --src_lang ar --dataset redfm
# python run_fshot_prompting.py --src_lang de --dataset redfm
# python run_fshot_prompting.py --src_lang en --dataset redfm
# python run_fshot_prompting.py --src_lang es --dataset redfm
# python run_fshot_prompting.py --src_lang fr --dataset redfm
# python run_fshot_prompting.py --src_lang it --dataset redfm
# python run_fshot_prompting.py --src_lang zh --dataset redfm


# python run_fshot_prompting.py --src_lang en --model_name gemma2
# python run_fshot_prompting.py --src_lang hi --model_name gemma2
# python run_fshot_prompting.py --src_lang te --model_name gemma2

# python run_fshot_prompting.py --src_lang ar --dataset redfm --model_name gemma2
# python run_fshot_prompting.py --src_lang de --dataset redfm --model_name gemma2
# python run_fshot_prompting.py --src_lang en --dataset redfm --model_name gemma2
# python run_fshot_prompting.py --src_lang es --dataset redfm --model_name gemma2
# python run_fshot_prompting.py --src_lang fr --dataset redfm --model_name gemma2
# python run_fshot_prompting.py --src_lang it --dataset redfm --model_name gemma2
# python run_fshot_prompting.py --src_lang zh --dataset redfm --model_name gemma2
