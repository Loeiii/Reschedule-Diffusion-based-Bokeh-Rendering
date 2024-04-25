'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=False)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.Defocus_dataset import Defocus_dataset as D
    dataset = D(dataroot= dataset_opt['dataroot'],
                datatype= dataset_opt['datatype'],
                origin_fnumber= dataset_opt['origin_fnumber'],
                target_fnumber= dataset_opt['target_fnumber'],
                split= phase,
                data_len= dataset_opt['data_len']
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
