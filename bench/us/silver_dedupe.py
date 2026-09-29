"""Keep one copy of near-identical reviewed paragraphs with the same labels."""

import collections
import re
import unicodedata


def words(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def signature(row):
    source = row["text"].encode()
    return tuple(sorted((span["kind"], words(source[span["start"]:span["end"]].decode()))
                        for span in row["entities"]))


def shingles(text):
    tokens = words(text)
    return {tokens[index:index + 7] for index in range(len(tokens) - 6)}


class NearDuplicateIndex:
    def __init__(self):
        self.by_labels = collections.defaultdict(list)

    def add(self, row):
        self.by_labels[signature(row)].append((row["id"], shingles(row["text"])))

    def prior(self, row):
        sample = shingles(row["text"])
        if len(sample) < 5:
            return None
        for source_id, previous in self.by_labels[signature(row)]:
            if len(previous) < 5:
                continue
            if len(sample & previous) / min(len(sample), len(previous)) >= 0.80:
                return source_id
        return None


class NearTextIndex:
    def __init__(self):
        self.rows = []
        self.by_shingle = collections.defaultdict(set)

    def add(self, source_id, text):
        sample = shingles(text)
        index = len(self.rows)
        self.rows.append((source_id, sample))
        for shingle in sample:
            self.by_shingle[shingle].add(index)

    def prior(self, text):
        sample = shingles(text)
        if len(sample) < 5:
            return None
        matches = collections.Counter(index for shingle in sample
                                      for index in self.by_shingle.get(shingle, ()))
        for index, count in matches.items():
            source_id, previous = self.rows[index]
            if len(previous) >= 5 and count / min(len(sample), len(previous)) >= 0.80:
                return source_id
        return None
