import numpy as np
import pandas as pd
import random

seed = 2024
np.random.seed(seed)
random.seed(seed)


def combine_csv(tableqa_dev, text_to_sql_dev, dataset):

    df = text_to_sql_dev[['id', 'question', 'answer']]

    df.loc[:,['ans_text_to_sql']] = text_to_sql_dev['sql_result']
    df.loc[:,['ans_tableqa']] = tableqa_dev['predictions']

    df.loc[:,['acc_text_to_sql']] = text_to_sql_dev['acc'].astype('int16')
    df.loc[:,['acc_tableqa']] = tableqa_dev['acc'].astype('int16')

    df.loc[:,['log_prob_sum_text_to_sql']] = text_to_sql_dev['log_probs_sum']
    df.loc[:,['log_prob_sum_tableqa']] = tableqa_dev['log_probs_sum']

    df.loc[:,['log_prob_avg_text_to_sql']] = text_to_sql_dev['log_probs_avg']
    df.loc[:,['log_prob_avg_tableqa']] = tableqa_dev['log_probs_avg']

    df.loc[:,['truncated_text_to_sql']] = text_to_sql_dev['truncated'].astype('int16')
    df.loc[:,['truncated_tableqa']] = tableqa_dev['truncated'].astype('int16')

    # df.loc[:,['query_fuzzy']] = text_to_sql_dev['query_fuzzy']
    df.loc[:,['query_pred']] = text_to_sql_dev['sql_pred']

    df.loc[:,['labels']] = [ 0 if int(x)==1 else 1 for x in df['acc_text_to_sql'].to_list()]
    df.loc[:,['input_tokens']] = tableqa_dev['input_tokens']
    
    df.loc[:, ['oracle']] = df.apply(lambda x: 1 if x['acc_text_to_sql'] == 1 or x['acc_tableqa'] == 1 else 0, axis=1)

    return df



def load_df_test(dataset, test_split, downsize=None):

    dataset_suffix = f'{dataset}_plus'

    tableqa_test = pd.read_csv(f"./predict/{dataset}/{dataset_suffix}_tableqa_test{test_split}_noise1.csv")
    text_to_sql_test = pd.read_csv(f"./predict/{dataset}/{dataset_suffix}_text_to_sql_test{test_split}_noise1.csv")
    df_test = combine_csv(tableqa_test, text_to_sql_test, dataset)

    return df_test



if __name__=='__main__':
    dataset = 'tablebench'
    test_split = 10
    downsize = None

    df_test = load_df_test(dataset, test_split, downsize=downsize)

    desired_order_selected = ['id', 'question', 'answer', 'acc_tableqa', 'ans_tableqa', 'acc_text_to_sql', 'ans_text_to_sql', 'query_pred', 'labels']
    remaining_columns = [col for col in df_test.columns if col not in desired_order_selected]
    df_test = df_test[desired_order_selected + remaining_columns]
    df_test.to_csv(f'./predict/{dataset}_classifier_test{test_split}.csv', na_rep='',index=False)
