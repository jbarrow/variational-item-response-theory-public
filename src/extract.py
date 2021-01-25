import torch
import jsonlines

from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from src.datasets import load_dataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--dataset', type=str, default='squad')

    args = parser.parse_args()

    dataset = load_dataset(args.dataset, train = True)
    saved_model = torch.load(args.checkpoint / 'checkpoint.pth.tar')

    weights = saved_model['model_state_dict']

    models, items = [], []

    for i, model in tqdm(enumerate(dataset.ix_to_model)):
        models.append(
            {
                'submission_id': model,
                'ability_mu': weights['ability_mu_lookup.weight'][i].tolist(),
                'ability_logvar': weights['ability_logvar_lookup.weight'][i].tolist()
            }
        )

    for i, item in tqdm(enumerate(dataset.ix_to_id)):
        items.append(
            {
                'submission_id': item,
                'item_feat_mu': weights['item_mu_lookup.weight'][i].squeeze().tolist(),
                'item_feat_logvar': weights['item_logvar_lookup.weight'][i].squeeze().tolist()
            }
        )

    with jsonlines.open(args.checkpoint / 'models.jsonlines', 'w') as writer:
        writer.write_all(models)
    
    with jsonlines.open(args.checkpoint / 'items.jsonlines', 'w') as writer:
        writer.write_all(items)