import collections
import math
import numbers
import pytest
import labeled_matrix
skip = pytest.mark.skip

def as_distribution(f):
    ''' Decorator. Convert the first argument into a P distribution. '''
    def transform(*args):
        return f(P(args[0]), *args[1:])
    return transform

def test_as_distribution():
    @as_distribution
    def identity(px):
        return px
    assert isinstance(identity([0.4, 0.6]), P), 'input got transformed'
    # TODO make it work for classes as well
    # class MyClass:
    #     @as_distribution
    #     def identity(self, px):
    #         return px
    # my_object = MyClass()
    # assert isinstance(my_object.identity([0.4, 0.6]), P), 'works on classes'

def positive(x):
    ''' True if x is a nonzero positive number. Good for probabilities. '''
    return x > 0

def test_positive():
    assert positive(0.5)
    assert not positive(0)
    assert not positive(-0.1)

def uniform(domain):
    n_elements = len(domain)
    return P(lambda _: 1 / n_elements, domain)

def test_uniform():
    assert uniform(['a', 'b'])[0] == 0.5

def tup_collect_values(tup, patterns):
    result = {}
    for i, pattern in enumerate(patterns):
        if isinstance(pattern, list) or isinstance(pattern, tuple):
            result.update(tup_collect_values(tup[i], pattern))
        else:
            result[pattern] = tup[i]
    return result

def test_tup_collect_values():
    assert tup_collect_values(('a'), [1]) == {1: 'a'}
    assert tup_collect_values(('a', 'b'), [1, 2]) == {1: 'a', 2: 'b'}
    assert tup_collect_values((('a',), 'b'), [(1,), 2]) == {1: 'a', 2: 'b'}

def tup_create_tuple(patterns, values):
    result = []
    for pattern in patterns:
        if isinstance(pattern, list) or isinstance(pattern, tuple):
            result.append(tup_create_tuple(pattern, values))
        else:
            result.append(values[pattern])
    return tuple(result)

def reorder_tup(tup, old_pattern, new_pattern):

    return tup_create_tuple(new_pattern, tup_collect_values(tup, old_pattern))

def test_reorder_tup():
    assert reorder_tup(('a', 'b'), (1, 2), (1, 2)) == ('a', 'b')
    assert reorder_tup(('a', 'b'), (1, 2), (2, 1)) == ('b', 'a')
    assert reorder_tup(('a', 'b', 'c'), (1, 2, 3), ((1, 2), 3)) == (('a', 'b'), 'c')
    assert reorder_tup((('a', 'b')), ((0, 1)), (0, 1)) == ('a', 'b')
    assert reorder_tup((('l', 't'), 's'), ((1, 2), 3), (3, 1, 2)) == ('s', 'l', 't')

def test_reorder_tup_with_letters():
    assert reorder_tup((1, 2), ('a', 'b'), ('b', 'a'))

class P:
    def __init__(self, px, domain=None):
        if isinstance(px, P): # If already P, just make a note of the dist
            self.distribution = px.distribution
            if callable(px.distribution):
                self._given_domain = px.domain
        else: # Otherwise, wrap the list or dict or etc...
            self.distribution = px
        if domain:
            self._given_domain = set(domain)
        if (not isinstance(pick_one(self.probs), numbers.Real) and
                not isinstance(pick_one(self.probs), P)):
            # If it's a conditional, then fix the nested Ps
            self.distribution = {event: P(p) for event, p in self}

    @property
    def codomain(self):
        accum = set()
        for _, pdist in self:
            for e, _ in pdist:
                accum.add(e)
        return accum

    def __contains__(self, e):
        if isinstance(self.distribution, dict):
            return e in self.distribution
        raise NotImplementedError('other fundamental types')

    def fill_cond(self):
        ''' Fill with zeroes any missing outputs. '''
        result = {cause: {} for cause, _ in self}
        codomain = self.codomain # precompute
        # FIXME google if codomain is ok name for outcome domain
        for cause, _ in self:
            for outcome in codomain:
                result[cause][outcome] = (self[cause][outcome] if outcome in
                    self[cause] else 0)
        return P(result)

    @property
    def matrix(self):
        domain = self.domain
        codomain = self.codomain
        full = self.fill_cond()
        return labeled_matrix.M(
            [[full[cause][effect] for effect in codomain]
                for cause in domain], domain, codomain)

    @property
    def domain(self):
        ''' Returns a set of all events in the domain of the distribution. '''
        if callable(self.distribution):
            return self._given_domain
        if isinstance(self.distribution, collections.Sequence):
            return set(range(len(self.distribution)))
        if isinstance(self.distribution, dict):
            return set(self.distribution.keys())
        else:
            raise TypeError('Distribution must be P, function, sequence or dict.')

    @property
    def probs(self):
        ''' Returns a set of all event probabilities of the distribution. '''
        return dict(self).values()

    def cond_items(self):
        ''' Allows you to conveniently iterate over all the items. '''
        for trigger, pdist in self:
            for outcome, p in pdist:
                yield (trigger, outcome, p)

    @property
    def is_cond(self):
        ''' Returns True if this is a conditional probability distribution '''
        return isinstance(pick_one(self.probs), P)

    def to_cond(self, given=0, to=1):
        ''' Returns P(Y|X) = P(X,Y)/P(X) given P(X,Y). '''
        # TODO now that we parametrised this function, stop using names x and y
        # TODO decide if a CondP class is necesary of we can abuse this one
        px = self.marginalize(keep=given)
        result = collections.defaultdict(lambda: collections.defaultdict(int))
        for tup in self.domain:
            x = tup[given] # TODO refactor x and y unpacking into another function
            if isinstance(to, int):
                y = tup[to]
            if isinstance(to, tuple):
                y = tuple([tup[i] for i in to])
            if px[x] > 0:
                result[x][y] = self[tup] / px[x]
        return P(result)

    def to_joint(self, px):
        ''' Returns P(X,Y) if self is P(Y|X) and the argument is P(X). '''
        px = P(px)
        return P({(trigger, outcome): p * px[trigger]
                  for trigger, outcome, p in self.cond_items()})

    def when(self, px):
        ''' Takes P(Y|X) and P(X) and returns P(Y) '''
        pxy = self.to_joint(px)
        py = pxy.marginalize(keep=1)
        return py

    def marginalize(self, keep=0):
        ''' Computes P(X) = Σ_y P(X,Y) '''
        result = collections.defaultdict(int)
        for event, p in self:
            result[event[keep]] += p
        return P(result)

    def normalize(self):
        computed_sum = sum(self.probs)
        return P({event: p / computed_sum for event, p in self})

    def reorder(self, old, new):
        return P({reorder_tup(tup, old, new): p for tup, p in self})

    def swap(self):
        ''' Swaps P(X,Y) to P(Y,X) '''
        return self.reorder((1, 2), (2, 1))
        # assert len(pick_one(self.domain)) == 2, 'can only swap 2 items'
        # return P({(y, x): p for (x, y), p in self})

    def __getitem__(self, key):
        if callable(self.distribution):
            return self.distribution(key)
        return self.distribution[key]

    def __iter__(self):
        for x in self.domain:
            yield x, self[x]

    def __eq__(self, other):
        return dict(self) == dict(other)

    def __str__(self):
        return 'P({})'.format(str(dict(self)))

    def __repr__(self):
        return self.__str__()

    def filter(self, byx=lambda _: True, byp=lambda _: True):
        return P({x: p for (x, p) in dict(self).items() if byx(x) and byp(p)})

    @property
    def row_key(self):
        ''' Creates a mapping to ints for all events in the input space. '''
        ordered_domain = sorted(self.domain)
        return {event: ordered_domain.index(event) for event in ordered_domain}

    def unsparse(self):
        raise Exception('remember to also write the test')

    @property
    def column_key(self):
        ''' Create a mapping to ints for all events in the target space. '''
        unsparsed = self.unsparse()
        target_space = unsparsed[pick_one(unsparsed.domain)].domain

    @staticmethod
    def from_map(deterministic):
        d = {}
        for state, action in deterministic.items():
            d[state] = {action: 1}
        return P(d)

class TestP:
    def test_in(self):
        assert 'a' in P({'a': 1})
        assert not 'b' in P({'a': 1})

    def test_from_list(self):
        assert P([0.1, 0.2, 0.7]).domain == {0, 1, 2}
        assert P([0.1, 0.2, 0.7])[1] == 0.2

    def test_equality(self):
        assert P([0.2, 0.8]) == P([0.2, 0.8])
        assert P([0.2, 0.8]) != P([0.8, 0.2]), 'order matters'
        assert P([0.2, 0.8]) != P([0.2, 0.8, 0]), 'zeroes matter'
        assert P([[0, 1], [1, 0]]) == P([[0, 1], [1, 0]]), 'works on a conditional'
        assert P([[0, 1], [0.5, 0.5]]) != P([[0, 1], [1, 0]]), 'works on a conditional'

    def test_probs(self):
        assert sorted(P({'a': 0.6, 'b': 0.4}).probs) == sorted([0.4, 0.6])
        assert list(P([1, 1]).probs) == [1, 1], 'duplicates work'

    def test_from_dict(self):
        assert P({'a': 0.2, 'b': 0.8}).domain == {'a', 'b'}
        assert P({'a': 0.2, 'b': 0.8})['b'] == 0.8

    def test_filter(self):
        actual = P({'a': 0.2, 'b': 0, 'c': 0.7, 'd': 0}).filter(byp=positive)
        expected = P({'a': 0.2, 'c': 0.7})
        assert actual == expected

    def test_dict_to_dict(self):
        original = {'a': 0.2, 'b': 0.8}
        assert dict(P(original)) == original

    def test_is_cond(self):
        assert not P([0.5, 0.5]).is_cond
        assert P({'a': {'b': 1}})

    def test_to_cond(self):
        assert P({
            (0, 0): 1, (1, 1): 0
        }).to_cond() == P({0: {0: 1}}), "shouldn't crash because of zeroes"
        assert P({
            (0, 0): 0.1, (0, 1): 0.25,
            (1, 0): 0.45, (1, 1): 0.2
        }).to_cond() == P({
            0: {0: 0.1 / (0.1 + 0.25), 1: 0.25 / (0.1 + 0.25)},
            1: {0: 0.45 / (0.45 + 0.2), 1: 0.2 / (0.45 + 0.2)}
        }), 'matches hand calculation'
        # TODO also test with different arguments

    def test_to_joint(self):
        assert (P({'red': {'tomato': 1}}).to_joint({'red': 1}) ==
            P({('red', 'tomato'): 1}))
        p_weather = {'rainy': 0.1, 'sunny': 0.9}
        cond = P({
            'rainy': P({'wet': 0.9, 'dry': 0.1}),
            'sunny': P({'wet': 0.2, 'dry': 0.8})})
        expected = P({
            ('rainy', 'wet'): 0.1 * 0.9,
            ('rainy', 'dry'): 0.1 * 0.1,
            ('sunny', 'wet'): 0.9 * 0.2,
            ('sunny', 'dry'): 0.9 * 0.8
        })
        assert cond.to_joint(p_weather) == expected, 'hand calculation'

    def test_when(self):
        assert P({'rains': {'wet': 1}}).when({'rains': 1}) == P({'wet': 1})

    def test_swap(self):
        assert P({('cat', 'dog'): 1}).swap() == P({('dog', 'cat'): 1})

    def test_marginalize(self):
        p = P({
            (0, 0): 0.1, (0, 1): 0.25,
            (1, 0): 0.45, (1, 1): 0.2
        })
        assert p.marginalize(keep=0) == P([p[0, 0] + p[0, 1], p[1, 0] + p[1, 1]])

    def test_normalize(self):
        assert P([1, 1]).normalize() == P([0.5, 0.5])

    def test_reorder(self):
        assert P({(0, 1): 1}).reorder((0, 1), (1, 0)) == P({(1, 0): 1})
    
    def test_cond_items(self):
        actual = P({'is_angry': {'mistake': 1}}).cond_items()
        assert list(actual) == [
            ('is_angry', 'mistake', 1)
        ]

    def test_codomain(self):
        example = P({'a': {'x': 0.4, 'y': 0.6}, 'x': {'z': 1}})
        assert example.codomain == set(['x', 'y', 'z'])

    def test_fill_cond(self):
        example = P({'a': {'x': 0.4, 'y': 0.6}, 'x': {'z': 1}})
        actual = example.fill_cond()
        expected = P({
            'a': {'x': 0.4, 'y': 0.6, 'z': 0},
            'x': {'x': 0, 'y': 0, 'z': 1}
        })
        assert actual == expected

    def test_matrix(self):
        example = P({'a': {'x': 0.4, 'y': 0.6}, 'x': {'z': 1}})
        assert set(example.matrix.row_symbols) == set('ax')
        assert set(example.matrix.column_symbols) == set('xyz')

    def test_from_map(self):
        assert P.from_map({0: 1, 1: 0}) == P({0: P({1: 1}), 1: P({0: 1})})

@as_distribution
def ev(px, f=lambda e, p: e):
    '''
    Computes the expected value E[x] given a probability distribution

    If given a function px, then the domain argument is required.
    '''
    return sum([f(e, p) * p for e, p in px.filter(byp=positive)])

def test_ev():
    assert ev({4: 0.5, 5: 0.5}) == 4.5
    p_animals = {'cat': 0.5, 'dog': 0.5}
    animal_value = {'cat': 4, 'dog': 5}
    assert ev(p_animals, lambda e, p: animal_value[e]) == 4.5, 'takes f arg'

def event_information(p):
    ''' The surprisal associated with an outcome given the probability. '''
    return - math.log2(p)

def test_event_information():
    assert event_information(1) == 0, 'inevitable event has no surprise'

@as_distribution
def h(px):
    '''
    H(x) = -Σp log p

    Calculates the amount of Shannon entropy of a random variable, \
    given its probability distribution.
    '''
    return ev(px, lambda e, p: event_information(p))

def test_h():
    assert h([1, 0]) == 0, 'absolute absolute certainty has no entropy'
    assert h([0.5, 0.5]) == 1, 'fair coins maximum entropy'

@as_distribution
def mi(pxy):
    '''
    I(X;Y) = H(X) - H(X|Y)

    Computes the mutual information between two variables given the
    joint probability distribution.
    '''
    px = pxy.marginalize()
    pyx = pxy.swap()
    return h(px) - cond_h(pyx.filter(byp=positive))

class TestMI():
    def test_independent_zero(self):
        assert mi({
            (0, 0): 0.25, (0, 1): 0.25,
            (1, 0): 0.25, (1, 1): 0.25
        }) == 0

    def test_perfect_correlation(self):
        assert mi({
            (0, 0): 0.5, (0, 1): 0.0,
            (1, 0): 0.0, (1, 1): 0.5
        }) == 1

    def test_blank_column_and_row(self):
        assert mi({
            (0, 0): 1, (0, 1): 0,
            (1, 0): 0, (1, 1): 0
        }) == 0

@as_distribution
def cond_mi(pxyz):
    ''' I(X;Y|Z) given P(X, Y, Z) '''
    pz = pxyz.marginalize(keep=2)
    pxy_given_z = pxyz.to_cond(given=2, to=(0, 1))
    return ev(pz, lambda z, _: mi(pxy_given_z[z]))

class TestCondMI():
    def test_reduces_to_mi(self):
        assert mi({
            (0, 0, 'nothing'): 0.5, (0, 1, 'nothing'): 0.0,
            (1, 0, 'nothing'): 0.0, (1, 1, 'nothing'): 0.5
        }) == 1

    def test_hand_calculated(self):
        pxyz = P({
            (0, 0, 0): 1, # When z = 0, x and y are correlated
            (1, 1, 0): 1,
            (0, 1, 1): 1, # When z = 1, x and y are anti-correlated
            (1, 0, 1): 1
        }).normalize()
        assert cond_mi(pxyz) == 1

def pick_one(s):
    ''' Picks one element from the given set. '''
    return next(iter(s))

def test_pick_one():
    s = {'a', 'b', 'c'}
    assert pick_one(s) in s, 'the actual value was taken from the set'

@as_distribution
def decompose_domain(pxyz):
    result = [set() for _ in pick_one(pxyz.domain)]
    for joint_event, _ in pxyz:
        for i, event in enumerate(joint_event):
            result[i].add(event)
    return result

def test_decompose_domain():
    assert decompose_domain({
        ('tomato', 'five'): 0.25, ('carrot', 'five'): 0.25,
        ('tomato', 'seven'): 0.25, ('carrot', 'seven'): 0.25
    }) == [set(['tomato', 'carrot']), set(['five', 'seven'])]

@as_distribution
def cond_h(pxy):
    ''' H(Y|X), given P(X,Y) '''
    px = pxy.marginalize()
    py_given_x = pxy.to_cond()
    return ev(px, lambda e, _: h(py_given_x[e]))

def test_cond_h():
    pxy = P({
        (0, 0): 0 * 0.4, (0, 1): 1 * 0.4, # p(y|x) = [0, 1], h = 0, p(x) = 0.4,
        (1, 0): 0.5 * 0.6, (1, 1): 0.5 * 0.6 # p(y|x) = [0.5, 0.5], h = 1, p(x) = 0.6
    })
    assert cond_h(pxy) == 0.6
