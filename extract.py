""" Extracts high-priority tweets from a large collection of tweets available in path provided and writes
results in output folder."""
import json

import click
from model import predict as _predict
from tqdm.auto import tqdm


def predict(batch):
    texts = [d['text'] for d in batch]
    priority_lst = _predict(texts, 768, 'en', '../../resources/urgency_en.pt')
    return [d for d, p in zip(batch, priority_lst) if int(p) > 0]


@click.command()
@click.argument("ipt", type=click.Path(exists=True))
@click.argument("opt", type=click.Path(exists=False))
def cli(ipt, opt):
    batch = []
    with open(ipt, 'r', encoding='utf-8') as fp:
        idx = 0
        for line in tqdm(fp.readlines()):
            cols = line.split('\t')
            if len(cols) != 4:
                continue
            tweet_id, text, created_at, username = cols
            record = dict(tweet_id=tweet_id, text=text, created_at=created_at, username=username)
            batch.append(record)
            if len(batch) == 1000:
                predict(batch)
                json.dump(batch, open('{}/batch_{}.json'.format(opt, idx), 'w'))
                batch.clear()
                idx += 1
        # Ignore the remaining (< 1000)


if __name__ == '__main__':
    cli()
