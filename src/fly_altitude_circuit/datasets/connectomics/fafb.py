"""FAFB cached-table grouping, padding and ranking."""
import numpy as np
import pandas as pd


def prepare_fafb_heatmap(df, pre_super_class="visual_projection", top_cells=10):
    if pre_super_class != None:
        df = df[df['pre_super_class'] == pre_super_class]
        name = pre_super_class
    else:
        name = 'all_inputs'
    df['post_side'] = df['post'].apply(lambda x: 'left' if x == 720575940623468691 else 'right')
    df['post_cell_type'] = 'DNa10'
    df['side'] = df['pre_side'].eq(df['post_side']).map({True: 'ipsi', False: 'contra'})
    for cell_id in df.pre.unique():
        if len(df[df['pre'] == cell_id]) < 2:
            temp_df = df[df['pre'] == cell_id].head(1)
            if df[df['pre'] == cell_id]['side'].values[0] == 'contra':
                ipsi_side = df[df['pre'] == cell_id]['pre_side'].values[0]
                temp_df['post_side'] = ipsi_side
                if ipsi_side == 'right':
                    temp_df['post'] = 720575940627946261
                else:
                    temp_df['post'] = 720575940623468691
                temp_df['side'] = 'ipsi'
                temp_df['weight'] = 0
            else:
                ipsi_side = df[df['pre'] == cell_id]['pre_side'].values[0]
                if ipsi_side == 'left':
                    contra_side = 'right'
                else:
                    contra_side = 'left'
                temp_df['post_side'] = contra_side
                if contra_side == 'right':
                    temp_df['post'] = 720575940627946261
                else:
                    temp_df['post'] = 720575940623468691
                temp_df['side'] = 'contra'
                temp_df['weight'] = 0
            df = pd.concat([df, temp_df])
    df['cell_counts'] = np.zeros(len(df))
    for cell_type in df.pre_cell_type.unique():
        L_count = len(df[(df['pre_cell_type'] == cell_type) & (df['pre_side'] == 'right')]['pre'].unique())
        R_count = len(df[(df['pre_cell_type'] == cell_type) & (df['pre_side'] == 'left')]['pre'].unique())
        cell_count = int(np.ceil((R_count + L_count) / 2))
        df.loc[df['pre_cell_type'] == cell_type, 'cell_counts'] = cell_count
    pivot_df = df.pivot_table(index=['pre_cell_type', 'pre_NT', 'cell_counts'], columns=['side'], values='weight', fill_value=0, aggfunc='mean')
    pivot_df['sum'] = pivot_df['contra'] + pivot_df['ipsi']
    sorted_df = pivot_df.sort_values(by='sum', ascending=False)
    heatmap_df = sorted_df.head(top_cells).drop(columns=['sum'])
    return heatmap_df[['ipsi', 'contra']].copy()
