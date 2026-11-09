"""MANC data regeneration
"""
from .regeneration_paths import output_file, output_directory


def retrieve_downstream_connectivity_all_cells(cell_type='DNa10', synapse_thresh_val=2, write_to_csv=False, save_dir=None, *, client):
    """retrieve all cell outputs of specified cell type"""
    import neuprint
    from neuprint import merge_neuron_properties
    if write_to_csv and save_dir is not None:
        output_directory(save_dir)
    cell_criteria = neuprint.NeuronCriteria(type=f'.*{cell_type}.*', regex=True)
    neuron_df, conn_df = neuprint.queries.fetch_adjacencies(sources=cell_criteria, targets=None, client=client)
    merged_df = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
    merged_df['post_side'] = merged_df['instance_post'].str.split('_').str[-1].map({'L': 'left', 'R': 'right'})
    merged_df['pre_side'] = merged_df['instance_pre'].str.split('_').str[-1].map({'L': 'left', 'R': 'right'})
    grouped_df = merged_df.groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'post_side', 'pre_side']).sum().reset_index()
    df = grouped_df.drop(['roi', 'instance_pre', 'instance_post'], axis=1)
    df_filtered = df[df['weight'] >= synapse_thresh_val]
    df_filtered['side'] = df['pre_side'].eq(df['post_side']).map({True: 'ipsi', False: 'contra'})
    df_clean = df_filtered.sort_values(by='weight', ascending=False).reset_index().drop(columns=['index'])
    if write_to_csv and save_dir is not None:
        df_clean.to_csv(output_file(save_dir, f'{cell_type}_cell_outputs_VNC' + '.csv'))
    return df_clean
