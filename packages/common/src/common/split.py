import random

def split_dev_test(records: list, dev_frac: float = 0.6):
    records = records[:]
    random.shuffle(records)
    k = int(len(records) * dev_frac)
    return records[:k], records[k:]