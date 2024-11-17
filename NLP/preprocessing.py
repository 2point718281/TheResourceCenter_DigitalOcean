import nltk
from dataclasses import dataclass
import types
from typing import Union
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from difflib import get_close_matches, SequenceMatcher
from collections import Counter
from functools import cache, lru_cache, wraps
from itertools import permutations
import json
import os
import time

try:
    nltk.data.find('corpora/brown.zip')  # Check for the Brown Corpus
except LookupError:
    nltk.download('brown')

try:
    nltk.data.find('corpora/wordnet.zip')  # Check for WordNet
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4.zip')  # Check for the Open Multilingual WordNet
except LookupError:
    nltk.download('omw-1.4')


lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
dir_ = os.path.dirname(__file__)

@dataclass
class Filter:
    '''Filter class to preprocess strings'''
    func: Union[types.FunctionType, types.BuiltinMethodType, str]
    desc: str
    name: str

    def __call__(self, obj):
        if isinstance(self.func, types.FunctionType):
            return self.func(obj)

        elif isinstance(self.func, types.BuiltinMethodType):
            return getattr(str, self.func.__name__)(obj)

        elif isinstance(self.func, str):
            return getattr(str, self.func)(obj)

    def __str__(self):
        return self.name + ': ' + self.desc

class FilterList:
    def __init__(self, filters, save_cache = False, try_load = False):
        self.filters = filters
        self.cache = save_cache
        if try_load:
            self.load_caches()

    def __call__(self, obj, print_ = False):
        if print_:
            print('Original: ' + str(obj))
            for x, filter_ in enumerate(self.filters):
                obj = filter_(obj)
                print(f'After Filter {x}, or {filter_.name}: ' + str(obj))
            # save_all_caches(*[filter_.func for filter_ in self.filters])
            return obj
        else:
            return [obj := filter_(obj) for filter_ in self.filters][-1]

    def __str__(self):
        return '\n'.join([str(number) + '. ' + str(filter_) for number, filter_ in enumerate(self.filters)])

    def __repr__(self):
        return '\n'.join([str(number) + '. ' + repr(filter_) for number, filter_ in enumerate(self.filters)])

    def summary(self):
        return str(self)

    def save_caches(self):
        save_all_caches(*[filter_.func for filter_ in self.filters])

    def load_caches(self):
        load_all_caches(*[filter_.func for filter_ in self.filters])

## Utility functions and definitions
words = list(wn.words())
keyboard_grid = '''`1234567890-=
qwertyuiop[]\\
asdfghjkl;'
zxcvbnm,./'''.split('\n')

other_keyboard_grid = '''~!@#$%^&*()_+
QWERTYUIOP{}|
ASDFGHJKL;'
ZXCVBNM,./'''.split('\n')

# Get all words in the Brown Corpus
words_ = [word.lower() for word in brown.words()]
word_frequencies = Counter(words_)

# Sort words by frequency in descending order
sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

# Get just the words in order of popularity
popular_words = [word for word, freq in sorted_words if word.isalpha()]

with open(os.path.join(dir_, 'popularity.json')) as f:
    p = json.loads(f.read())
    popular = [word for word in sorted(p, key = lambda x: p[x]) if word.isalpha()]

# Acronyms
with open(os.path.join(dir_, 'acronyms.json')) as f:
    p = json.loads(f.read())
    acronyms = {acronym.lower(): p[acronym].lower() for acronym in p}

# extra words go here
exceptions = []
words += list(acronyms) + exceptions
matcher = SequenceMatcher()

# Code from https://stackoverflow.com/questions/15585493/store-the-cache-to-a-file-functools-lru-cache-in-python-3-2
def cached(func):
    func.cache = {}
    @wraps(func)
    def wrapper(*args):
        try:
            return func.cache[hash(args)]
        except KeyError as e:
            func.cache[hash(args)] = result = func(*args)
            return result   
    return wrapper

def index_list(ls, v):
    try:
        return ls.index(v)

    except ValueError:
        return len(ls)

## TO FIX
def save_all_caches(*funcs):
    '''Makes cache files in a "cache" folder'''
    os.makedirs('cache', exist_ok = True)
    for x, func in enumerate(funcs):  
        try:
            if not isinstance(func, str):
                with open(os.path.join('cache', f'{x}.json'), 'w') as f:
                    f.write(json.dumps(func.cache))

        except (AttributeError, TypeError) as e:
            raise e
            continue

def load_all_caches(*funcs):
               
    for x, func in enumerate(funcs):
        try:
            if os.path.exists(os.path.join('cache', f'{x}.json')) and not isinstance(func, str):
                with open(os.path.join('cache', f'{x}.json')) as f:
                    if not hasattr(func, 'cache'):
                        func.cache = {}
                    print(json.loads(f.read()))
                    func.cache.update(json.loads(f.read()))
                        

        except (AttributeError, json.decoder.JSONDecodeError) as e:
            print(e)
            continue
## ##


@cached
def char_pos(l):
    '''Gets the position of a character on a standard QWERTY keyboard'''
    for rnum, row in enumerate(keyboard_grid):
        if l in row:
            return rnum, row.index(l)

    for rnum, row in enumerate(other_keyboard_grid):
        if l in row:
            return rnum, row.index(l)

    return float('nan'), float('nan')

@cached
def char_dist(char1, char2):
    pos1, pos2 = char_pos(char1), char_pos(char2)
    if pos1[0] == float('nan') or pos2[0] == float('nan'): # Check if either of the characters are not on the english keyboard
        return float('inf')  # Return infinity if they are

    else:
        return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 # Distance formula, no need to square root as the distance can\'t be less than one, and sqrt(x) < sqrt(y) iff x < y for all positive x and y

@cached
def rank_wordsim_on_counter(word1, word2):
    word1 = stemmer.stem(word1)
    c1, c2 = Counter(word1), Counter(word2)
    
    if not sum((c1 - c2).values()) + sum((c2 - c1).values()):
        return 0
    
    if matcher.b != word1:
        matcher.set_seq2(word1)
    matcher.set_seq1(word2)
    opcodes = [op for op in matcher.get_opcodes() if op[0] != 'equal']
    replaces = [op[1:] for op in opcodes if op[0] == 'replace']
    instructs = []
    extra = 0
    delete_insertw = 20
    for replace in replaces:
        instructs.extend([instruct for instruct in zip(word1[replace[0]:replace[1]], word2[replace[2]:replace[3]]) if instruct[0] != instruct[1]])
        if replace[3] - replace[2] != replace[1] - replace[0]:
            extra += delete_insertw * abs(replace[1] - replace[0] - (replace[3] - replace[2]))

    dist = sum([char_dist(char1, char2) for char1, char2 in instructs])
    others = delete_insertw * sum([max(op[2] - op[1], op[4] - op[3]) for op in opcodes if op[0] in ('insert', 'delete')])
    return dist + others + extra

@cached
def round_to_close_word(string):
    if string in words:
        return string
    matcher.set_seq2(string)
    close = get_close_matches(string, words, n = 50)
    if not close:   # We don't have any idea what to autocorrect it to if get_close_matches returns [], so we just drop the token
        return ''

    scores_func = lambda x: rank_wordsim_on_counter(string, x) # + 10 ** (-20) * index_list(popular_words, x)
    
    scores = [scores_func(word) for word in close]
    words_scores = [(word, score) for word, score in zip(close, scores)]
    # scores_func = lambda x: rank_wordsim_on_counter(string, x) + 10 ** (-20) * index_list(popular, x)
    # scores = [scores_func(word) for word in words_scores]
    # print(list(zip(words_scores, scores)))
    return sorted(words_scores, key = lambda x: x[1])[0][0]# min(zip(words_scores, scores), key = lambda x: x[1])[0]

@cached
def expandacronyms(string):
    return acronyms.get(string, string)  # Get the full form of "string". If string isn't in the acronym list, just return "string"

@cached
def expand_all_acronyms(ls):
    return sum(tuple([tokenize(expandacronyms(word)) for word in ls]), tuple())

@cached
def lemmatize_(string):
    orig = string
    types = ['n', 'v', 'a', 'r']
    level = 0
    while orig == string and level < len(types):
        string = lemmatizer.lemmatize(string, types[level])
        level += 1
    return string

@cached
def lemmatize_all(ls):
    return tuple([lemmatize_(word) for word in ls])

@cached
def round_to_closest_word_all(ls):
    return tuple([round_to_close_word(word) for word in ls])
        
## Predefined filters
lower = Filter('lower', 'Convert all letters in a text to lowercase', 'LOWER')
upper = Filter('upper', 'Convert all letters in a text to uppercase', 'UPPER')
tokenize = Filter(lambda x: tuple(nltk.tokenize.word_tokenize(x)), 'Tokenize a string', 'TOKENIZE')
remove_punctuation = Filter(cached(lambda x: ''.join([x := x.replace(p, ' ') for p in string.punctuation][-1])), 'Remove Punctuation', 'RMPUNCT')
remove_stopwords = Filter(cached(lambda x: tuple([word for word in x if word not in nltk.corpus.stopwords.words("english")])), 'Remove Stopwords', 'RMSTOP')
lemmatize = Filter(lemmatize_all, 'Lemmatize word', 'LEMWORD')
round_closest = Filter(round_to_closest_word_all, 'Round all words to their closest form in the dictionary', 'ROUNDWORD')
expandacros = Filter(expand_all_acronyms, 'Expand all acronyms to their full form', 'EXPANDACROS')

## Predefined FilterLists

def get_filter(lower_ = True, upper_ = False, rmpunct = True, tokenize_ = True, rmstop = True, expand = True, lemma = True, round_close = True):
    filters = []
    if lower_:
        filters.append(lower)

    if upper_:
        filters.append(upper)

    if rmpunct:
        filters.append(remove_punctuation)

    if tokenize_:
        filters.append(tokenize)

    if rmstop:
        filters.append(remove_stopwords)

    if expand:
        filters.append(expandacros)
        
    if round_close:
        filters.append(round_closest)
        
    if lemma:
        filters.append(lemmatize)
    
    return FilterList(filters)

default = get_filter()
noround = get_filter(round_close = False)
nolower = get_filter(lower_ = False)

## Time the various preprocessers
def timeit(func, iters = 1000, *args):
    start_time = time.time()
    for _ in range(iters - 1):
        [func(arg) for arg in args]
    out = [func(arg) for arg in args]
    
    end_time = time.time()
    return (end_time - start_time) / (len(args) * iters), out
