from multivae.data.datasets import MultimodalBaseDataset, IncompleteDataset, DatasetOutput


def split_inputs(inputs:MultimodalBaseDataset,n_split):
    
    n_data = len(list(inputs.data.values())[0])
    mini_batch_size = int(n_data/n_split)
    if mini_batch_size == 0:
        return [inputs]
    batches = []
    i_start = 0
    while i_start < n_data:
        print(i_start)
        i_end = min(i_start+mini_batch_size,n_data)
        mini_batch = DatasetOutput(
            data = {k : inputs.data[k][i_start:i_end].reshape((-1, *inputs.data[k].shape[1::]))for k in inputs.data},
            labels = inputs.labels[i_start:i_end] if inputs.labels is not None else None,
        )
        if hasattr(inputs, 'masks'):
            mini_batch['masks'] = {k : inputs.masks[k][i_start:i_end] for k in inputs.masks}
        
        batches.append(mini_batch)
        i_start += mini_batch_size
    
    return batches
    