"""MANC cached-table grouping, padding and ranking."""
import numpy as np
import pandas as pd


def prepare_manc_heatmap(df, top_cells=10):
    for cell_id in df.bodyId_post.unique():
        if len(df[df['bodyId_post'] == cell_id]) < 2:
            temp_df = df[df['bodyId_post'] == cell_id].head(1)
            if df[df['bodyId_post'] == cell_id]['side'].values[0] == 'contra':
                ipsi_side = df[df['bodyId_post'] == cell_id]['post_side'].values[0]
                temp_df['pre_side'] = ipsi_side
                if ipsi_side == 'right':
                    temp_df['bodyId_pre'] = 10531
                else:
                    temp_df['bodyId_pre'] = 10506
                temp_df['side'] = 'ipsi'
                temp_df['weight'] = 0
            else:
                ipsi_side = df[df['bodyId_post'] == cell_id]['pre_side'].values[0]
                if ipsi_side == 'left':
                    contra_side = 'right'
                else:
                    contra_side = 'left'
                temp_df['pre_side'] = contra_side
                if contra_side == 'right':
                    temp_df['bodyId_pre'] = 10531
                else:
                    temp_df['bodyId_pre'] = 10506
                temp_df['side'] = 'contra'
                temp_df['weight'] = 0
            df = pd.concat([df, temp_df])
    df['cell_counts'] = np.zeros(len(df))
    for cell_type in df.type_post.unique():
        L_count = len(df[(df['type_post'] == cell_type) & (df['post_side'] == 'right')]['bodyId_post'].unique())
        R_count = len(df[(df['type_post'] == cell_type) & (df['post_side'] == 'left')]['bodyId_post'].unique())
        cell_count = int(np.ceil((R_count + L_count) / 2))
        df.loc[df['type_post'] == cell_type, 'cell_counts'] = cell_count
    df['pre_NT'] = 'acetylcholine'
    pivot_df = df.pivot_table(index=['type_post', 'pre_NT', 'cell_counts'], columns=['side'], values='weight', fill_value=0, aggfunc='mean')
    pivot_df['sum'] = pivot_df['contra'] + pivot_df['ipsi']
    sorted_df = pivot_df.sort_values(by='sum', ascending=False)
    heatmap_df = sorted_df.head(top_cells).drop(columns=['sum'])
    return heatmap_df[['ipsi', 'contra']].copy()
