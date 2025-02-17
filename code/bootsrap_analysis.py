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
from dataloader import *
import networkx as nx


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


def create_dataset_df():
    eps = 1e-10
    ###### Sentence Length ######
    stats_dict      = ddict(list)
    
    if args.dataset == 'redfm':
        src_langs   = ['en', 'es', 'fr', 'it', 'de']
        tgt_langs   = ['en', 'es', 'fr', 'it', 'de', 'ar', 'zh']

    if args.dataset == 'indore':
        src_langs   = ['en', 'hi', 'te']
        tgt_langs   = ['en', 'hi', 'te']


    dimensions      = ['sent_len', 'lex_len', 'dep_len']


    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            for connection in ['residual']:

                for ml_model in ['mbert-base', 'xlmr-base']:
                    
                    baseline_dict           = {} 
                    baseline_arr            = []
                    
                    for seed in [11737, 15123, 98105, 98109, 15232]:
                        baseline_file                    = f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{ml_model}-parser_stanza-gnn_rgcn-connection_{connection}-dep_0-gnn-depth_2-seed_{seed}.csv'
                        baseline_df                      =  pd.read_csv(baseline_file)
                        baseline_f1                      = f1_score(baseline_df['true_rel'], baseline_df['pred_rel'], average='macro')
                        baseline_dict[seed]              = baseline_f1
                        baseline_df['seed']              = [seed for _ in range(len(baseline_df))]
                        baseline_df['sidx']              = [idx for idx in range(len(baseline_df))]
                        baseline_arr.append(baseline_df)

                    baseline_df = pd.concat(baseline_arr)

                    top_seeds        = sorted(baseline_dict, key=baseline_dict.get, reverse=True)[:3]
                    baseline_df      = baseline_df[baseline_df['seed'].isin(top_seeds)]

                    # replace the seed values with 1,2,3 for consistency
                    for idx, seed in enumerate(top_seeds):
                        baseline_df.loc[baseline_df['seed'] == seed, 'seed'] = idx
                            


                    for gnn in ['rgcn', 'rgat']:
                        for dep_model in ['stanza', 'trankit']:

                            dep_dict        = {}
                            dep_arr         = []

                            len_stats_df                        = pd.read_csv(f'../stats/{args.dataset}/{tgt_lang}_{ml_model}_{dep_model}_test.csv')
                            len_stats_df['sidx']                = [idx for idx in range(len(len_stats_df))]

                            for seed in [11737, 15123, 98105, 98109, 15232]:

                                try:
                                    dep_file                    = f'../predictions/{args.dataset}/{src_lang}_{tgt_lang}-model_{ml_model}-parser_{dep_model}-gnn_{gnn}-connection_{connection}-dep_1-gnn-depth_2-seed_{seed}.csv'
                                    dep_df                      =  pd.read_csv(dep_file)
                                    dep_f1                      = f1_score(dep_df['true_rel'], dep_df['pred_rel'], average='macro')
                                    dep_dict[seed]              = dep_f1
                                    dep_df['seed']              = [seed for _ in range(len(dep_df))]
                                    dep_df['sidx']              = [idx for idx in range(len(dep_df))]
                                    dep_arr.append(dep_df)
                                except Exception as e:
                                    print(f'Absent File: {dep_file}')
                                    continue

                            all_dep_df        = pd.concat(dep_arr)
                            # select the top 3 seeds for each src_tgt pair
                            top_seeds         = sorted(dep_dict, key=dep_dict.get, reverse=True)[:3]
                            final_dep_df      = all_dep_df[all_dep_df['seed'].isin(top_seeds)]

                            ############# Substitute the seeds values with 1,2,3 for consistency #####################

                            for idx, seed in enumerate(top_seeds):
                                final_dep_df.loc[final_dep_df['seed'] == seed, 'seed'] = idx
                            ###########################################################################################

                            final_dep_df       = final_dep_df.merge(len_stats_df, on=['sidx', 'doc_text'], how='inner')
                            final_baseline_df  = baseline_df.merge(len_stats_df, on=['sidx', 'doc_text'], how='inner')
                            
                            final_dep_df['dep_pred_rel']      = final_dep_df['pred_rel']
                            final_baseline_df['bl_pred_rel']  = final_baseline_df['pred_rel']

                            # merge the dataframes into a single dataframe

                            results_df = final_dep_df.merge(final_baseline_df, on=['doc_text', 'true_rel', 'seed', 'sidx', 'sent_len', 'dep_len', 'lex_len', 'sent_len_val', 'lex_len_val', 'dep_len_val'], how='inner')
                            
                            for dim in dimensions:

                                for bin in ['Low', 'Medium', 'High']:

                                    curr_df = results_df[results_df[dim] == bin]
                                    if len(curr_df) == 0: continue

                                    dep_f1 = f1_score(curr_df['true_rel'], curr_df['dep_pred_rel'], average='macro')
                                    baseline_f1 = f1_score(curr_df['true_rel'], curr_df['bl_pred_rel'], average='macro')

                                    f1_diff  = 100*(dep_f1 - baseline_f1)/(baseline_f1 + eps)


                                    stats_dict['src'].append(src_lang)
                                    stats_dict['tgt'].append(tgt_lang)
                                    stats_dict['dep_model'].append(dep_model)
                                    stats_dict['ml_model'].append(ml_model)
                                    stats_dict['dimension'].append(dim)
                                    stats_dict['bin'].append(bin)
                                    stats_dict['gnn'].append(gnn)
                                    stats_dict['f1_diff'].append(f1_diff)        
                        

                            # print(f'{src_lang} - {tgt_lang} - {dep_model} - {ml_model} - {connection}')

    ############# Create a dataframe for the test dataset ####################

    stats_df    = pd.DataFrame(stats_dict)
    stats_df.to_csv(f'../results/{args.dataset}_stats.csv', index=False)





def legend_mapper(dim):
    dim_map ={'sent_len': 'Sentence Length', 'lex_len': 'Entity Distance', 'dep_len': 'Dependency Path Length'}
    return dim_map[dim]




def generate_images():

    for dataset in ['indore', 'redfm']:

        f1_df = pd.read_csv(f'../results/{dataset}_stats.csv')

        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=2.0)


        for dimension in ['sent_len', 'lex_len', 'dep_len']:

            for ml_model in ['mbert-base', 'xlmr-base']:

                hue_orders = ['Low','Medium','High']

                dimension_df = f1_df[(f1_df['dimension'] == dimension) & (f1_df['ml_model'] == ml_model)]
                dimension_df[legend_mapper(dimension)] = dimension_df['bin']

                dimension_df['score']    = dimension_df['f1_diff']

                # remove error bars 
                # draw a barplot with FacetGrid 
                g = sns.catplot(x="dep_model", y= 'score', hue=legend_mapper(dimension), data=dimension_df, row= 'src',
                                col = 'tgt', height=6, kind="bar", palette="muted", hue_order=hue_orders, ci=None)
                
                sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
                g.set(ylabel= f'Diff in F1 score for F1 Diff')
                # savefig g
                g.savefig(f"../results/{dataset}_{dimension}_{ml_model}.png")
                plt.show()
                plt.clf()
                plt.close()



def generate_images_overall():

    for dataset in ['indore', 'redfm']:

        f1_df = pd.read_csv(f'../results/{dataset}_stats.csv')

        modes = []
        for idx, row in f1_df.iterrows():
            if row['src'] == row['tgt']:
                modes.append('ID')
            else:
                modes.append('ZS')

        f1_df['mode'] = modes

        f1_df = f1_df[f1_df['ml_model'] == 'mbert-base']

        print(f1_df)

        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=2.0)        

        hue_orders = ['Low','Medium','High']

        # draw a barplot with FacetGrid 
        g = sns.catplot(x="ml_model", y= 'f1_diff', hue='bin', data=f1_df, height=6, kind="bar", palette="muted", hue_order=hue_orders, col = 'mode', row= 'dimension', ci=None)

        sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
        g.set(ylabel= f'Diff in F1 score for Dep Parse')
        # savefig g
        g.savefig(f"../results/{dataset}_overall.png")
        plt.show()
        plt.clf()
        plt.close()




def compute_dep_path_len(dep_data):

    n1_nodes, n2_nodes = np.array(dep_data['n1_mask']), np.array(dep_data['n2_mask'])

    edges   = np.array(dep_data['edge_index'])

    G       = nx.Graph()

    for e1, e2 in zip(edges[0], edges[1]):
        G.add_edge(e1, e2)
    
    n1_nodes = np.where(n1_nodes == 1)[0]
    n2_nodes = np.where(n2_nodes == 1)[0]

    min_path = np.inf

    # find the shortest path between all pairs of n1 and n2 nodes

    path_len   = []
    for n1 in n1_nodes:
        for n2 in n2_nodes:
            try:
                path = nx.shortest_path(G, source=n1, target=n2)
                # print(path)
                path_len.append(len(path))
            except Exception as e:
                continue
    
    try:
        max_min_path = max(path_len)
        assert max_min_path >= 0
    except Exception as e:
        max_min_path = np.inf

    return max_min_path






def compute_dataset_stats():

    for dataset in ['indore', 'redfm']:

        if dataset == 'redfm':
            src_langs   = ['en', 'es', 'fr', 'it', 'de']
            tgt_langs   = ['en', 'es', 'fr', 'it', 'de', 'ar', 'zh']
        if dataset == 'indore':
            src_langs   = ['en', 'hi', 'te']
            tgt_langs   = ['en', 'hi', 'te']

        for model_name in ['mbert-base', 'xlmr-base']:

            for dep_model in ['stanza', 'trankit']:

                for lang in tgt_langs:

                    test_dataset_stats = ddict(list)

                    with open(f'../data/{dataset}/{lang}_{model_name}_{dep_model}.dill', 'rb') as f:
                        loaded_dataset = dill.load(f)
                    
                    test_dataset = loaded_dataset['test']

                    sent_len_list    = []
                    lex_len_list     = []
                    dep_len_list     = []

                    for idx in range(len(test_dataset)):

                        sent_len                        = len(test_dataset[idx]['tokens'])
                        e1_ids, e2_ids                  = test_dataset[idx]['e1_ids'], test_dataset[idx]['e2_ids']

                        try:
                            assert 1 in e1_ids and 1 in e2_ids
                        
                            try:
                                indices                 = [i for i, x in enumerate(e1_ids) if x == 1]
                                first_e1, last_e1       = indices[0], indices[-1]
                            except Exception as e:
                                first_e1, last_e1       = indices[0], indices[0]

                            try:
                                indices                 = [i for i, x in enumerate(e2_ids) if x == 1]
                                first_e2, last_e2       = indices[0], indices[-1]
                            except Exception as e:
                                first_e2, last_e2       = indices[0], indices[0]


                            lex_dist                    = max(last_e1 - first_e2, last_e2 - first_e1)

                            assert lex_dist >= 0
                        
                        except Exception as e:
                            lex_dist = np.inf

                        sent_len_list.append(sent_len)
                        lex_len_list.append(lex_dist)                        
                        
                        dep_data    = test_dataset[idx]['dep_data']

                        dep_len     = compute_dep_path_len(dep_data)
                        dep_len_list.append(dep_len)

                    ############## Create a dataframe for the test dataset ####################
                    sent_q1, sent_q2  = np.percentile(sent_len_list, 25), np.percentile(sent_len_list, 75)
                    lex_q1, lex_q2    = np.percentile(lex_len_list, 25), np.percentile(lex_len_list, 75)
                    dep_q1, dep_q2    = np.percentile(dep_len_list, 25), np.percentile(dep_len_list, 75)

                    for idx in range(len(test_dataset)):

                        lex_len                 = lex_len_list[idx]
                        sent_len                = sent_len_list[idx]
                        dep_len                 = dep_len_list[idx]

                        lex_category            = obtain_mapper(lex_len,    [-np.inf, lex_q1,  lex_q2, np.inf])
                        sent_category           = obtain_mapper(sent_len,   [-np.inf, sent_q1, sent_q2, np.inf])
                        dep_category            = obtain_mapper(dep_len,    [-np.inf, dep_q1,  dep_q2, np.inf])


                        test_dataset_stats['idx'].append(idx)
                        test_dataset_stats['sent_len'].append(sent_category)
                        test_dataset_stats['lex_len'].append(lex_category)
                        test_dataset_stats['sent_len_val'].append(sent_len)
                        test_dataset_stats['lex_len_val'].append(lex_len)
                        test_dataset_stats['dep_len'].append(dep_category)
                        test_dataset_stats['dep_len_val'].append(dep_len)
                        test_dataset_stats['doc_text'].append(test_dataset[idx]['doc_text'])


                    print(f'{dataset} - {lang} - {model_name} - {dep_model} Sent Len: {sent_q1} - {sent_q2}')
                    print(f'{dataset} - {lang} - {model_name} - {dep_model} Lex Len:  {lex_q1}  - {lex_q2}')
                    print(f'{dataset} - {lang} - {model_name} - {dep_model} Dep Len:  {dep_q1}  - {dep_q2}')

                    test_dataset_stats_df = pd.DataFrame(test_dataset_stats)

                    test_dataset_stats_df.to_csv(f'../stats/{dataset}/{lang}_{model_name}_{dep_model}_test.csv')


                    # print(f'{dataset} - {lang} - {model_name} - Dep Len: {dep_q1} - {dep_q2}')

'''
Part of the code to consolidate all the results across the different seeds
'''


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



def agg_results_old():

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


    ### create a simple csv file for the indomain and cross domain results

    indomain_dict = ddict(list)

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
                        f1s    = [dep_dict[seed] for seed in top_seeds]
                        

                        mean_f1 = np.mean(f1s)
                        std_f1  = np.std(f1s) 

                        aggregated_dict['src'].append(src_lang)
                        aggregated_dict['tgt'].append(tgt_lang)
                        aggregated_dict['DEP'].append(dep_model)
                        aggregated_dict['GNN'].append(gnn_model)
                        aggregated_dict['ENC'].append(ml_model)
                        
                        result  = f'{round(100*mean_f1, 1)}\pm{round(100*std_f1, 1)}'

                        aggregated_dict['F1'].append(result)
                
                dep_dict    = {}
                for seed in [11737, 98105, 98109, 15232, 15123]:

                    try:
                        dep_f1 = stats_df[(stats_df['src'] == src_lang) & (stats_df['tgt'] == tgt_lang) & (stats_df['dep_model'] == '-') & (stats_df['gnn_model'] == '-') & (stats_df['ml_model'] == ml_model) & (stats_df['seed'] == seed)]['f1'].values[0]


                        dep_dict[seed] = dep_f1
                    except Exception as e:
                        import pdb; pdb.set_trace()
                        print(f"Absent File: {src_lang} - {tgt_lang} - None - None - {ml_model} - {seed}")
                
                top_seeds  = sorted(dep_dict, key=dep_dict.get, reverse=True)[:3]
                f1s    = [dep_dict[seed] for seed in top_seeds]

                mean_f1 = np.mean(f1s)
                std_f1  = np.std(f1s) 

                aggregated_dict['src'].append(src_lang)
                aggregated_dict['tgt'].append(tgt_lang)
                aggregated_dict['DEP'].append('-')
                aggregated_dict['GNN'].append('-')
                aggregated_dict['ENC'].append(ml_model)
                
                result  = f'{round(100*mean_f1, 1)}\pm{round(100*std_f1, 1)}'

                aggregated_dict['F1'].append(result)

    aggregated_df = pd.DataFrame(aggregated_dict)



    indomain_dict   = ddict(list)
    indomain_df     = aggregated_df[(aggregated_df['src'] == aggregated_df['tgt'])]

    
    for ml_model in ['mbert-base', 'xlmr-base']:        
        for dep_model in ['stanza', 'trankit']:
            for gnn_model in ['rgcn', 'rgat']:
                indomain_dict['DEP'].append(dep_model)
                indomain_dict['GNN'].append(gnn_model)
                indomain_dict['ENC'].append(ml_model)

                for lang in src_langs:
                    result = indomain_df[(indomain_df['src'] == lang) & (indomain_df['tgt'] == lang) & (indomain_df['DEP'] == dep_model) & (indomain_df['GNN'] == gnn_model) & (indomain_df['ENC'] == ml_model)]['F1'].values[0]
                    
                    indomain_dict[lang].append(result)

        indomain_dict['DEP'].append('-')
        indomain_dict['GNN'].append('-')
        indomain_dict['ENC'].append(ml_model)

        for lang in src_langs:
            result = indomain_df[(indomain_df['src'] == lang) & (indomain_df['tgt'] == lang) & (indomain_df['DEP'] == '-') & (indomain_df['GNN'] == '-') & (indomain_df['ENC'] == ml_model)]['F1'].values[0]
                                    
            indomain_dict[lang].append(result)                    
        
    # import pdb; pdb.set_trace()

    indomain_df = pd.DataFrame(indomain_dict)
    # get the indomain_results now
    indomain_df.to_csv(f'../results/{args.dataset}_indomain_results.csv', index=False)

    cross_domain_dict       = ddict(list)

    cross_domain_df         = aggregated_df
    # cross_domain_df         = aggregated_df[(aggregated_df['src'] != aggregated_df['tgt'])]

    for ml_model in ['mbert-base', 'xlmr-base']:
        for src_lang in src_langs:        
            for dep_model in ['stanza', 'trankit']:
                for gnn_model in ['rgcn', 'rgat']:
                    

                    cross_domain_dict['DEP'].append(dep_model)
                    cross_domain_dict['GNN'].append(gnn_model)
                    cross_domain_dict['ENC'].append(ml_model)
                    cross_domain_dict['Src'].append(src_lang)

                    for tgt_lang in tgt_langs:
                        if src_lang == tgt_lang:
                            cross_domain_dict[f'{tgt_lang}'].append('-')
                        else:
                            result = cross_domain_df[(cross_domain_df['src'] == src_lang) & (cross_domain_df['tgt'] == tgt_lang) & (cross_domain_df['DEP'] == dep_model) & (cross_domain_df['GNN'] == gnn_model) & (cross_domain_df['ENC'] == ml_model)]['F1'].values[0]                            
                            cross_domain_dict[f'{tgt_lang}'].append(result)

        
            cross_domain_dict['DEP'].append('-')
            cross_domain_dict['GNN'].append('-')
            cross_domain_dict['ENC'].append(ml_model)
            cross_domain_dict['Src'].append(src_lang)

            for tgt_lang in tgt_langs:
                if src_lang == tgt_lang: 
                    cross_domain_dict[f'{tgt_lang}'].append('-')
                else:
                    result = cross_domain_df[(cross_domain_df['src'] == src_lang) & (cross_domain_df['tgt'] == tgt_lang) & (cross_domain_df['DEP'] == '-') & (cross_domain_df['GNN'] == '-') & (cross_domain_df['ENC'] == ml_model)]['F1'].values[0]
                    cross_domain_dict[f'{tgt_lang}'].append(result)


    # get the cross domain results now
    cross_domain_df = pd.DataFrame(cross_domain_dict)
    cross_domain_df.to_csv(f'../results/{args.dataset}_cross_domain_results.csv', index=False)


    
def get_args():

    parser  = argparse.ArgumentParser()
    parser.add_argument("--dataset",        help="dataset of choice",               type=str, default='redfm')
    parser.add_argument("--step",           help="what process to follow",          type=str, required=True)
    args    = parser.parse_args()
    return args


if __name__ =='__main__':	

    args                            =   get_args()

    if args.step                    == 'create_df':
        create_dataset_df()

    elif args.step                  == 'sigtest':
        do_sigtest()

    elif args.step                  == 'gen_img':
        generate_images()

    elif args.step                  == 'all_results':
        get_overall_results()
    
    elif args.step                  == 'agg_results':
        agg_results()
    
    elif args.step                 == 'gen_img_overall':
        generate_images_overall()
    
    elif args.step                 == 'dataset_stats':
        compute_dataset_stats()

