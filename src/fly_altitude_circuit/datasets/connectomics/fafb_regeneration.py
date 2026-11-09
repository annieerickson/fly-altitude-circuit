"""Explicit FAFB regeneration using the paper's v3.0.0 annotation release.
"""
import pandas as pd
from .regeneration_paths import output_file, output_directory


def _flywire():
    from fafbseg import flywire
    flywire.set_default_dataset('production')
    return flywire


def get_all_input_ds(cell_type='DNa10', downstream_root_ids=None, upstream=True, downstream=False, save_df=False, save_dir=None, regex=False, materialization=783, annotation_version='v3.0.0'):
    """"
    get all inputs to a cell class (ie DNa10)
    materialization options: int(version), 'latest' or 'live'
    annotation_version pins cell-type annotations independently of materialization.
    """
    if save_df:
        output_directory(save_dir)
    flywire = _flywire()
    NC = flywire.NeuronCriteria
    if downstream_root_ids is not None:
        ds = flywire.get_connectivity(NC(root_id=downstream_root_ids, annotation_version=annotation_version), upstream=upstream, downstream=downstream, materialization=materialization)
        cell_type = flywire.search_annotations(downstream_root_ids[0], annotation_version=annotation_version)['cell_type'].values[0]
    else:
        ds = flywire.get_connectivity(NC(type=cell_type, regex=regex, annotation_version=annotation_version), upstream=upstream, downstream=downstream, materialization=materialization)
    if save_df == True:
        ds.to_pickle(output_file(save_dir, cell_type + f'_input_ds_v{materialization}.pkl'))
    return ds

def search_annotation_with_retry(neuron_id, max_retries=3, delay=2, materialization_type=783, annotation_version='v3.0.0'):
    """Search annotations with retries, keeping the selected release and materialization.
    """
    from requests.exceptions import HTTPError
    import time
    flywire = _flywire()
    for attempt in range(max_retries):
        try:
            if isinstance(materialization_type, int):
                return flywire.search_annotations(neuron_id, materialization=materialization_type, annotation_version=annotation_version)
            elif materialization_type == 'latest':
                return flywire.search_annotations(neuron_id, materialization='latest', annotation_version=annotation_version)
            else:
                return flywire.search_annotations(neuron_id, annotation_version=annotation_version)
        except HTTPError as e:
            if e.response.status_code == 502:
                print(f'Server busy, retry {attempt + 1}/{max_retries} for ID {neuron_id}')
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    print(f'Failed to get annotation after {max_retries} attempts')
                    return pd.DataFrame()
            else:
                raise
        except ValueError as e:
            print(f'ValueError for ID {neuron_id}: {e}')
            return pd.DataFrame()

def generate_df_with_all_inputs_and_types_with_NT(df_up, synapse_thresh=True, synapse_thresh_val=2, save_df=True, save_dir=None, cell_type_name='DNa10', materialization=783, annotation_version='v3.0.0'):
    """""
    generate a dataframe with all inputs to DNa10 and their cell types (or hemibrain type) and side 
    with the goal of being able to merge cell classes
    already have df with all upstream connections to DNa10
    synapse_thresh: only find annotations for synapses > synapse_thresh_val (decreased dataset size)
    *also add NT type
    Pin materialization 783 and annotation release v3.0.0.
    """
    from tqdm import tqdm
    if save_df and save_dir is not None:
        output_directory(save_dir)
    if synapse_thresh:
        df_up = df_up[df_up['weight'] >= synapse_thresh_val]
    result_columns = ['pre', 'post', 'weight', 'pre_cell_type', 'pre_side', 'pre_super_class', 'pre_NT']
    upstream_annotations_df = pd.DataFrame(columns=result_columns)
    data = []
    for _, row in tqdm(df_up.iterrows()):
        pre_id = row['pre']
        annotation = search_annotation_with_retry(pre_id, materialization_type=materialization, annotation_version=annotation_version)
        if annotation.empty:
            print(f'Annotation not found for ID: {pre_id}')
            continue
        else:
            post_id = row['post']
            weight = row['weight']
            cell_type = annotation['cell_type'].values[0]
            super_class = annotation['super_class'].values[0]
            if not isinstance(cell_type, str):
                cell_type = annotation['hemibrain_type'].values[0]
            side = annotation['side'].values[0]
            NT = annotation['top_nt'].values[0]
            data.append({'pre': pre_id, 'post': post_id, 'weight': weight, 'pre_cell_type': cell_type, 'pre_side': side, 'pre_super_class': super_class, 'pre_NT': NT})
    upstream_annotations_df = pd.DataFrame(data)
    if save_df and save_dir is not None:
        upstream_annotations_df.to_pickle(output_file(save_dir, cell_type_name + f'_upstream_annotations_df_with_NT_v{materialization}.pkl'))
    return upstream_annotations_df
