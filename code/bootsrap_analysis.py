import pickle
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import random
import numpy as np
import pandas as pd
from collections import defaultdict as ddict
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from collections import Counter
from helper import *
import dill
# from dataloader import *
from igraph import *






# def compute_f1(baseline_data, struct_data, idxs):

#     baseline_golds, baseline_preds, struct_golds, struct_preds = [baseline_data[idx]['true_rel'] for idx in idxs], \
#                                                                 [baseline_data[idx]['pred_rel'] for idx in idxs],\
#                                                                 [struct_data[idx]['true_rel'] for idx in idxs], \
#                                                                 [struct_data[idx]['pred_rel'] for idx in idxs],

#     baseline_f1 = f1_score(baseline_golds, baseline_preds, average='macro')
#     struct_f1   = f1_score(struct_golds, struct_preds, average='macro')

#     if struct_f1 > baseline_f1:
#         return 1
#     else:
#         return 0


def compute_bootstrap_f1(dep_df):


            
    golds, baseline_preds, struct_preds = list(dep_df['true_rel']), list(dep_df['bl_pred_rel']), list(dep_df['dep_pred_rel'])
                                                
    baseline_f1     = f1_score(golds, baseline_preds, average='macro')
    struct_f1       = f1_score(golds, struct_preds, average='macro')
    
    sys_f1          = struct_f1 - baseline_f1
    
    flag            = 1
    if sys_f1 < 0:
        flag        = -1
        sys_f1      = -sys_f1
        
    scores          = []
    
    for iter_ in range(0, 100):
        k        =  min(int(0.5*len(dep_df)), 100)
        idxs     =  random.choices(range(len(dep_df)), k=k)                    
        
        golds, baseline_preds, struct_preds = [dep_df.iloc[idx]['true_rel'] for idx in idxs], [dep_df.iloc[idx]['bl_pred_rel'] for idx in idxs], [dep_df.iloc[idx]['dep_pred_rel'] for idx in idxs],

        curr_baseline_f1    = f1_score(golds, baseline_preds, average='macro')
        curr_struct_f1      = f1_score(golds, struct_preds, average='macro')
        
        curr_f1_diff        = (curr_struct_f1 - curr_baseline_f1)*flag
        
        if curr_f1_diff > 2*sys_f1:
            scores.append(1)
        else:
            scores.append(0)

    scores_dict = {
        'pval': np.mean(scores),
        'struct_f1': struct_f1,
        'baseline_f1': baseline_f1,
        'size_k': k,
        'better': 'dep' if struct_f1 > baseline_f1 else 'baseline'
    }
    
    return scores_dict
    
    
    
def do_sigtest():
    sigtest     = ddict(list)

    if args.dataset == 'redfm':
        src_langs   = ['en', 'es', 'fr', 'it', 'de']
        tgt_langs   = ['en', 'es', 'fr', 'it', 'de', 'ar', 'zh']
    
    if args.dataset == 'indore':
        src_langs   = ['en', 'hi', 'te']
        tgt_langs   = ['en', 'hi', 'te']

    for src_lang in src_langs:
        for tgt_lang in tgt_langs:

            for connection in ['residual']:

                for dep_model in ['stanza', 'trankit']:

                    for ml_model in ['mbert-base', 'xlmr-base']:

                        dep_arr     = []
                        baseline_arr    = []

                        for seed in [11737, 11747, 98105, 98109, 15232]:
                            
                            dep_df                      =  pd.read_csv(f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{ml_model}-parser_{dep_model}-connection_{connection}-dep_1-gnn-depth_2-seed_{seed}.csv')

                            dep_df['seed']              = seed
                            dep_df['dep_pred_rel']      = dep_df['pred_rel']

                            baseline_df                 =  pd.read_csv(f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{ml_model}-parser_stanza-connection_{connection}-dep_0-gnn-depth_2-seed_{seed}.csv')

                            baseline_df['seed']         = seed
                            baseline_df['bl_pred_rel']  = baseline_df['pred_rel']    

                            # merge the dataframes into a single dataframe
                            dep_df = dep_df.merge(baseline_df, on=['doc_text', 'true_rel', 'seed', 'id'], how='inner')
                            dep_arr.append(dep_df)
                        

                        dep_df                                                  = pd.concat(dep_arr)
                        scores_dict                                             = compute_bootstrap_f1(dep_df)
                        
                        sigtest['src'].append(src_lang)
                        sigtest['tgt'].append(tgt_lang)
                        sigtest['dep_model'].append(dep_model)
                        sigtest['ml_model'].append(ml_model)
                        sigtest['connection'].append(connection)
                        sigtest['struct_f1'].append(scores_dict['struct_f1'])
                        sigtest['baseline_f1'].append(scores_dict['baseline_f1'])
                        sigtest['pvalue'].append(scores_dict['pval'])
                        sigtest['better'].append(scores_dict['better'])
                        sigtest['k'].append(scores_dict['size_k'])
                        
                        print(f'{src_lang}\t{tgt_lang}\t{dep_model}\t{ml_model}\t{scores_dict["struct_f1"]}\t{scores_dict["baseline_f1"]}\t{scores_dict["pval"]}\t{scores_dict["better"]}')


    df = pd.DataFrame(sigtest)
    df.to_csv(f'../results/{args.dataset}_sigtest_predictions.csv', index=False) 



def obtain_mapper(val, bins):
    if val > bins[0] and val <= bins[1]:
        return 'Low'
    elif val > bins[1] and val <= bins[2]:
        return 'Medium'
    elif val > bins[2] and val <= bins[3]:
        return 'High'

def obtain_numerical_mapper(val):
    if val > 2:
        return '3+'
    else:
        return str(val)


def create_df():

    ###### Sentence Length ######
    stats_dict      = ddict(list)
    
    if args.dataset == 'redfm':
        src_langs   = ['en', 'es', 'fr', 'it', 'de']
        tgt_langs   = ['en', 'es', 'fr', 'it', 'de', 'ar', 'zh']

    if args.dataset == 'indore':
        src_langs   = ['en', 'hi', 'te']
        tgt_langs   = ['en', 'hi', 'te']


    for src_lang in src_langs:
        for tgt_lang in tgt_langs:

            for connection in ['residual']:

                for dep_model in ['stanza', 'trankit']:

                    for ml_model in ['mbert-base', 'xlmr-base']:

                        dep_arr     = []
                        baseline_arr    = []

                        for seed in [11737, 11747, 98105, 98109, 15232]:
                            
                            dep_df                      =  pd.read_csv(f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{ml_model}-parser_{dep_model}-connection_{connection}-dep_1-gnn-depth_2-seed_{seed}.csv')

                            dep_df['seed']              = seed
                            dep_df['dep_pred_rel']      = dep_df['pred_rel']

                            baseline_df                 =  pd.read_csv(f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{ml_model}-parser_stanza-connection_{connection}-dep_0-gnn-depth_2-seed_{seed}.csv')

                            baseline_df['seed']         = seed
                            baseline_df['bl_pred_rel']  = baseline_df['pred_rel']    

                            # merge the dataframes into a single dataframe
                            dep_df = dep_df.merge(baseline_df, on=['doc_text', 'true_rel', 'seed', 'id', 'sent_len', 'lex_dist', 'dep_path'], how='inner')
                            dep_arr.append(dep_df)
                        
                        dep_df                                                  = pd.concat(dep_arr)

                        print(f'Done for {src_lang} - {tgt_lang} - {dep_model} - {ml_model} - {connection}', end='\r')

                        for idx in range(len(dep_df)):
                            stats_dict['src'].append(src_lang)
                            stats_dict['tgt'].append(tgt_lang)
                            stats_dict['sent_len'].append(dep_df.iloc[idx]['sent_len'])
                            stats_dict['dep_len'].append(dep_df.iloc[idx]['dep_path'])
                            stats_dict['lex_len'].append(dep_df.iloc[idx]['lex_dist'])
                            # stats_dict['true_rel'].append(dep_df.iloc[idx]['true_rel'])
                            # stats_dict['dep_pred_rel'].append(dep_df.iloc[idx]['dep_pred_rel'])
                            # stats_dict['bl_pred_rel'].append(dep_df.iloc[idx]['bl_pred_rel'])
                            stats_dict['ml_model'].append(ml_model)
                            stats_dict['dep_model'].append(dep_model)
                            stats_dict['connection'].append(connection)
                        
    stats_df    = pd.DataFrame(stats_dict)

    bins        = ddict(list)

    for src_lang in src_langs:
        for tgt_lang in tgt_langs:

            for connection in ['residual']:

                for dep_model in ['stanza', 'trankit']:

                    for ml_model in ['mbert-base', 'xlmr-base']:

                        curr_df         = stats_df[(stats_df['src'] == src_lang) & (stats_df['tgt'] == tgt_lang) & (stats_df['connection'] == connection) & (stats_df['dep_model'] == dep_model) & (stats_df['ml_model'] == ml_model)]
                        
                        sent_len_list    =  curr_df['sent_len'].tolist()
                        lex_len_list     =  curr_df['lex_len'].tolist()
                        dep_len_list     =  curr_df['dep_len'].tolist()

                        sent_q1, sent_q2 = np.percentile(sent_len_list, 25), np.percentile(sent_len_list, 50)
                        lex_q1, lex_q2   = np.percentile(lex_len_list, 25), np.percentile(lex_len_list, 50)
                        dep_q1, dep_q2   = np.percentile(dep_len_list, 25), np.percentile(dep_len_list, 50)
                        
                        bins['src_lang'].append(src_lang)
                        bins['tgt_lang'].append(tgt_lang)
                        bins['sent_len'].append([-np.inf, sent_q1, sent_q2, np.inf])
                        bins['lex_len'].append([-np.inf, lex_q1, lex_q2, np.inf])
                        bins['dep_len'].append([-np.inf, dep_q1, dep_q2, np.inf])
                        bins['dep_model'].append(dep_model)
                        bins['ml_model'].append(ml_model)
                        bins['connection'].append(connection)
 
    ### Create a dataframe for each src_tgt_pair ####

    f1_dict             = ddict(list)
    bins_df             = pd.DataFrame(bins)

    for src_lang in src_langs:
        for tgt_lang in tgt_langs:

             for connection in ['residual']:

                for dep_model in ['stanza', 'trankit']:

                    for ml_model in ['mbert-base', 'xlmr-base']:

                        # find the bins for the current src_tgt pair

                        curr_bins_df = bins_df[(bins_df['src_lang'] == src_lang) & (bins_df['tgt_lang'] == tgt_lang) & (bins_df['connection'] == connection) & (bins_df['dep_model'] == dep_model) & (bins_df['ml_model'] == ml_model)]

                        sent_len_range =  curr_bins_df['sent_len'].values[0]
                        lex_len_range  =  curr_bins_df['lex_len'].values[0]
                        dep_len_range  =  curr_bins_df['dep_len'].values[0]

                        for seed in [11737, 11747, 98105, 98109, 15232]:
                            
                            dep_df                      =  pd.read_csv(f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{ml_model}-parser_{dep_model}-connection_{connection}-dep_1-gnn-depth_2-seed_{seed}.csv')

                            dep_df['seed']              = seed
                            dep_df['dep_pred_rel']      = dep_df['pred_rel']

                            baseline_df                 =  pd.read_csv(f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{ml_model}-parser_stanza-connection_{connection}-dep_0-gnn-depth_2-seed_{seed}.csv')

                            baseline_df['seed']         = seed
                            baseline_df['bl_pred_rel']  = baseline_df['pred_rel']    

                            # merge the dataframes into a single dataframe
                            dep_df = dep_df.merge(baseline_df, on=['doc_text', 'true_rel', 'seed', 'id', 'sent_len', 'lex_dist', 'dep_path'], how='inner')

                            dep_arr.append(dep_df)
                        
                        print(f'Done for {src_lang} - {tgt_lang} - {dep_model} - {ml_model} - {connection}', end='\r')

                        dep_df                                                  = pd.concat(dep_arr)

                        dimensions_dict                                         = ddict(lambda: ddict(lambda: ddict(list)))
                        for idx in range(len(dep_df)):

                            sent_len                                = dep_df.iloc[idx]['sent_len']
                            lex_len                                 = dep_df.iloc[idx]['lex_dist']
                            dep_path                                = dep_df.iloc[idx]['dep_path']

                            baseline_gold                           = dep_df.iloc[idx]['true_rel']
                            baseline_pred                           = dep_df.iloc[idx]['bl_pred_rel']
                            dep_pred                                = dep_df.iloc[idx]['dep_pred_rel']

                            dimensions_dict['sent_len'][obtain_mapper(sent_len, sent_len_range)]['baseline_pred'].append(baseline_pred)
                            dimensions_dict['sent_len'][obtain_mapper(sent_len, sent_len_range)]['gold'].append(baseline_gold)
                            dimensions_dict['sent_len'][obtain_mapper(sent_len, sent_len_range)]['dep_pred'].append(dep_pred)

                            dimensions_dict['lex_len'][obtain_mapper(lex_len, lex_len_range)]['baseline_pred'].append(baseline_pred)
                            dimensions_dict['lex_len'][obtain_mapper(lex_len, lex_len_range)]['gold'].append(baseline_gold)
                            dimensions_dict['lex_len'][obtain_mapper(lex_len, lex_len_range)]['dep_pred'].append(dep_pred)

                            dimensions_dict['dep_len'][obtain_mapper(dep_path, dep_len_range)]['baseline_pred'].append(baseline_pred)
                            dimensions_dict['dep_len'][obtain_mapper(dep_path, dep_len_range)]['gold'].append(baseline_gold)
                            dimensions_dict['dep_len'][obtain_mapper(dep_path, dep_len_range)]['dep_pred'].append(dep_pred)


            

                        for dimension in dimensions_dict:
                            for val in dimensions_dict[dimension]:

                                f1_dict['feature'].append(dimension)
                                f1_dict['src_lang'].append(src_lang)
                                f1_dict['tgt_lang'].append(tgt_lang)
                                f1_dict['connection'].append(connection)
                                f1_dict['dep_model'].append(dep_model)
                                f1_dict['ml_model'].append(ml_model)
                                f1_dict['val'].append(val)

                                baseline_f1 = f1_score(dimensions_dict[dimension][val]['gold'], dimensions_dict[dimension][val]['baseline_pred'], average='macro')
                                dep_f1      = f1_score(dimensions_dict[dimension][val]['gold'], dimensions_dict[dimension][val]['dep_pred'], average='macro')

                                f1_dict['Dep Parse'].append(dep_f1 - baseline_f1)

                                

    f1_df               = pd.DataFrame(f1_dict)
    f1_df.to_csv(f'../results/{args.dataset}_f1_stats.csv', index=False)




def legend_mapper(dim):
    dim_map ={'sent_len': 'Sentence Length', 'lex_len': 'Entity Distance', 'dep_len': 'Dependency Path Length'}
    return dim_map[dim]




def generate_images():

    f1_df = pd.read_csv(f'../results/{args.dataset}_f1_stats.csv')

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=2.0)


    for dimension in ['sent_len', 'lex_len', 'dep_len']:

        for ml_model in ['mbert-base', 'xlmr-base']:

            hue_orders = ['Low','Medium','High']

            dimension_df = f1_df[(f1_df['feature'] == dimension) & (f1_df['ml_model'] == ml_model)]
            dimension_df[legend_mapper(dimension)] = dimension_df['val']

            dimension_df['score']    = dimension_df['Dep Parse']*100

            # draw a barplot with FacetGrid 
            g = sns.catplot(x="dep_model", y= 'score', hue=legend_mapper(dimension), data=dimension_df, row= 'src_lang',
                            col = 'tgt_lang', height=6, kind="bar", palette="muted", hue_order=hue_orders)
            
            sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
            g.set(ylabel= f'Diff in F1 score for Dep Parse')
            # savefig g
            g.savefig(f"../results/{args.dataset}_{dimension}_{ml_model}.png")
            plt.show()
            plt.clf()
            plt.close()


def get_overall_results():
    stats_dict      = ddict(list)
    
    if args.dataset == 'redfm':
        src_langs   = ['en', 'es', 'fr', 'it', 'de']
        tgt_langs   = ['en', 'es', 'fr', 'it', 'de', 'ar', 'zh']

    if args.dataset == 'indore':
        src_langs   = ['en', 'hi', 'te']
        tgt_langs   = ['en', 'hi', 'te']

    config_dict = ddict(list)
    cnt = 1

    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            # if src_lang != tgt_lang: continue
            for connection in ['residual']:
                for model_name in ['mbert-base', 'xlmr-base']:
                    for seed in [11737, 15123, 98105, 98109, 15232]:

                        dep             = '1'
                        for dep_model in ['stanza', 'trankit']:
                            for gnn_model in ['rgcn', 'rgat']:

                                try:
                                    dep_df                      =  pd.read_csv(f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{model_name}-parser_{dep_model}-gnn_{gnn_model}-connection_{connection}-dep_{dep}-gnn-depth_2-seed_{seed}.csv')

                                except Exception as e:

                                    print(f'Absent File: ../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{model_name}-parser_{dep_model}-gnn_{gnn_model}-connection_{connection}-dep_{dep}-gnn-depth_2-seed_{seed}.csv')
                                    
                                    config_dict['ArrayTaskID'].append(cnt)
                                    config_dict['DATASET'].append(args.dataset)
                                    config_dict['MLM'].append(model_name)
                                    config_dict['DEP_MODEL'].append(dep_model)
                                    config_dict['SRC_LANG'].append(src_lang)
                                    config_dict['TGT_LANG'].append(tgt_lang)
                                    config_dict['DEP'].append('1')
                                    config_dict['SEED'].append(seed)
                                    config_dict['GNN_MODEL'].append(gnn_model)
                                    
                                    cnt +=1
                                    continue

                                stats_dict['src'].append(src_lang)
                                stats_dict['tgt'].append(tgt_lang)
                                stats_dict['dep_model'].append(dep_model)
                                stats_dict['gnn_model'].append(gnn_model)
                                stats_dict['ml_model'].append(model_name)
                                stats_dict['seed'].append(seed)
                                test_f1 =  f1_score(dep_df['true_rel'], dep_df['pred_rel'], average='macro')
                                stats_dict['f1'].append(test_f1)

                        try:
                            dep_df                      =  pd.read_csv(f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{model_name}-parser_stanza-gnn_rgcn-connection_{connection}-dep_0-gnn-depth_2-seed_{seed}.csv')
                        except Exception as e:

                            print(f"Absent File: ../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{model_name}-parser_stanza-gnn_rgcn-connection_{connection}-dep_0-gnn-depth_2-seed_{seed}.csv")

                            config_dict['ArrayTaskID'].append(cnt)
                            config_dict['DATASET'].append(args.dataset)
                            config_dict['MLM'].append(model_name)
                            config_dict['DEP_MODEL'].append('stanza')
                            config_dict['SRC_LANG'].append(src_lang)
                            config_dict['TGT_LANG'].append(tgt_lang)
                            config_dict['DEP'].append('0')
                            config_dict['SEED'].append(seed)
                            config_dict['GNN_MODEL'].append('rgcn')
                            
                            cnt +=1
                            continue
                            

                        stats_dict['src'].append(src_lang)
                        stats_dict['tgt'].append(tgt_lang)
                        stats_dict['dep_model'].append('None')
                        stats_dict['gnn_model'].append('None')
                        stats_dict['ml_model'].append(model_name)
                        stats_dict['seed'].append(seed)
                        test_f1 =  f1_score(dep_df['true_rel'], dep_df['pred_rel'], average='macro')
                        stats_dict['f1'].append(test_f1)

    stats_df    = pd.DataFrame(stats_dict)
    stats_df.to_csv(f'../results/{args.dataset}_overall_results.csv', index=False)
                                
                                
    # /data/shire/projects/multilingual_re/predictions/redfm/de_ar-model_mbert-base-parser_stanza-gnn_rgat-connection_residual-dep_1-gnn-depth_2-seed_11737.csv                            
        
    config_df = pd.DataFrame(config_dict)
    config_df.to_csv(f'../configs/{args.dataset}_eval_missing_config.csv',  index=False, sep=' ')



def agg_results():

    stats_df = pd.read_csv(f'../results/{args.dataset}_overall_results.csv')

    # identify the common set of source and target languages

    aggregated_dict = ddict(list)

    if args.dataset == 'redfm':
        src_langs   = ['en', 'es', 'fr', 'it', 'de']
        tgt_langs   = ['en', 'es', 'fr', 'it', 'de', 'ar', 'zh']

    if args.dataset == 'indore':
        src_langs   = ['en', 'hi', 'te']
        tgt_langs   = ['en', 'hi', 'te']

    for src_lang in src_langs:
        for tgt_lang in tgt_langs:

            for ml_model in ['mbert-base', 'xlmr-base']:        

                for dep_model in ['stanza', 'trankit']:
                    for gnn_model in ['rgcn', 'rgat']:

                        dep_dict    = {}
                        
                        for seed in [11737, 98105, 98109, 15232, 15123]:


                            dep_f1 = stats_df[(stats_df['src'] == src_lang) & (stats_df['tgt'] == tgt_lang) & (stats_df['dep_model'] == dep_model) & (stats_df['gnn_model'] == gnn_model) & (stats_df['ml_model'] == ml_model) & (stats_df['seed'] == seed)]['f1'].values[0]

                            dep_dict[seed] = dep_f1
                        
                        # select the top 3 seeds for each src_tgt pair

                        top_seeds  = sorted(dep_dict, key=dep_dict.get, reverse=True)[:3]

                        top_f1s    = [dep_dict[seed] for seed in top_seeds]
                        all_f1s    = [dep_dict[seed] for seed in dep_dict]

                        aggregated_dict['src'].append(src_lang)
                        aggregated_dict['tgt'].append(tgt_lang)
                        aggregated_dict['dep_model'].append(dep_model)
                        aggregated_dict['gnn_model'].append(gnn_model)
                        aggregated_dict['ml_model'].append(ml_model)
                        aggregated_dict['top_seed'].append(top_seeds)
                        aggregated_dict['top_f1_mean'].append(round(100*np.mean(top_f1s),2))
                        aggregated_dict['top_f1_std'].append(round(100*np.std(top_f1s),2))
                        aggregated_dict['all_f1_mean'].append(round(100*np.mean(all_f1s),2))
                        aggregated_dict['all_f1_std'].append(round(100*np.std(all_f1s),2))

                
                dep_dict    = {}
                for seed in [11737, 98105, 98109, 15232, 15123]:

                    dep_f1 = stats_df[(stats_df['src'] == src_lang) & (stats_df['tgt'] == tgt_lang) & (stats_df['dep_model'] == 'None') & (stats_df['gnn_model'] == 'None') & (stats_df['ml_model'] == ml_model) & (stats_df['seed'] == seed)]['f1'].values[0]

                    dep_dict[seed] = dep_f1
                
                # select the top 3 seeds for each src_tgt pair

                top_seeds  = sorted(dep_dict, key=dep_dict.get, reverse=True)[:3]

                top_f1s    = [dep_dict[seed] for seed in top_seeds]
                all_f1s    = [dep_dict[seed] for seed in dep_dict]

                aggregated_dict['src'].append(src_lang)
                aggregated_dict['tgt'].append(tgt_lang)
                aggregated_dict['dep_model'].append("None")
                aggregated_dict['gnn_model'].append('None')
                aggregated_dict['ml_model'].append(ml_model)
                aggregated_dict['top_seed'].append(top_seeds)
                aggregated_dict['top_f1_mean'].append(round(100*np.mean(top_f1s),2))
                aggregated_dict['top_f1_std'].append(round(100*np.std(top_f1s),2))
                aggregated_dict['all_f1_mean'].append(round(100*np.mean(all_f1s),2))
                aggregated_dict['all_f1_std'].append(round(100*np.std(all_f1s),2))


    aggregated_df = pd.DataFrame(aggregated_dict)
    aggregated_df.to_csv(f'../results/{args.dataset}_aggregated_results.csv', index=False)




    
def get_args():

    parser  = argparse.ArgumentParser()
    parser.add_argument("--dataset",        help="dataset of choice",               type=str, default='redfm')
    parser.add_argument("--step",           help="what process to follow",          type=str, required=True)
    args    = parser.parse_args()
    return args


if __name__ =='__main__':	

    args                            =   get_args()

    if args.step                    == 'create_df':
        create_df()
    elif args.step                  == 'sigtest':
        do_sigtest()

    elif args.step                  == 'gen_img':
        generate_images()

    elif args.step                  == 'all_results':
        get_overall_results()
    
    elif args.step                  == 'agg_results':
        agg_results()

