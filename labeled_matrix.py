import numbers
import unittest
import pymatrix
assertSequenceEqual = unittest.TestCase('__init__').assertSequenceEqual

class M():
    ''' A matrix whose indices can be any hashable objects. ''' 
    def __init__(self, m, row_symbols=None, column_symbols=None):
        ''' 
        m will be transformed into a matrix using 'pymatrix.matrix'.
        row_symbols and column_symbols can be of any sequence type.
        '''
        self._matrix = (m if isinstance(m, pymatrix.Matrix)
            else pymatrix.matrix(m))
        self.row_symbols = (row_symbols if row_symbols
            else range(self._matrix.numcols))
        self.column_symbols = (column_symbols if column_symbols
            else range(self._matrix.numrows))

    def __eq__(self, other):
        return all([
            self._matrix == other._matrix,
            self.row_symbols == other.row_symbols,
            self.column_symbols == other.column_symbols
        ])

    def _reorder_rows(self, new_ordering):
        assert set(self.row_symbols) == set(new_ordering)
        reordered = [None] * self._matrix.numrows
        for old_i, symbol in enumerate(self.row_symbols):
            new_i = new_ordering.index(symbol)
            reordered[new_i] = list(self._matrix.row(old_i))
        return M(reordered, new_ordering, self.column_symbols)

    def __add__(self, other):
        if (self.column_symbols == other.column_symbols and
            self.row_symbols == other.row_symbols):
            return M(
                self._matrix + other._matrix, 
                self.row_symbols,
                self.column_symbols
            )
        else:
            raise NotImplementedError("Code won't work for " +
                "differently labeled matrices.")

    def __mul__(self, other):
        ''' Implements matrix multiplication with labeled rows and cols. '''
        if isinstance(other, M):
            to_match = self.column_symbols
            fixed_other = other._reorder_rows(to_match)
            return M(
                self._matrix * fixed_other._matrix, 
                self.row_symbols, 
                fixed_other.column_symbols
            )
        if isinstance(other, numbers.Number):
            return M(self._matrix * other, self.row_symbols, self.column_symbols)

    def __rmul__(self, other):
        ''' This should only happen when other is not a M. So a number. '''
        return self * other

    def __str__(self):
        return "M('{}', {}, {})".format(
            self._matrix, repr(self.row_symbols), repr(self.column_symbols)
        )
    
    def is_square(self):
        return self._matrix.is_square()

    def det(self):
        return self._matrix.det()

    @property
    def numcols(self):
        return self._matrix.numcols

class TestM():
    def test_auto_matrix(self):
        assert M(pymatrix.matrix('0')) == M('0')

    def test_eq(self):
        assert M('0', 'A', 'X') == M('0', 'A', 'X'), 'should equal copy pasted'
        assert M('9 9 9') != M('0 1 2'), 'different matrix content'
        assert M('0', 'A', '_') != M('0', 'B', '_'), 'different row symbols'
        assert M('0', '_', 'X') != M('0', '_', 'Y'), 'different column symbols'

    def test_mult_number(self):
        example = M('1 2')
        expected = M('2 4')
        assert example * 2 == expected, 'matrix on left'
        assert 2 * example == expected, 'matrix on right'

    def test_str(self):
        assert str(M('1 2', 'A', 'AB')) == "M('1 2', 'A', 'AB')"

    def test_row_symbols(self):
        assert M('0 \n 1', 'AB', '_').row_symbols == 'AB', 'explicit labeling'
        assertSequenceEqual(M('0 1 \n 2 3').row_symbols, [0, 1]), 'implicit row indices'

    def test_column_symbols(self):
        assert M('0 1', '_', 'AB').column_symbols == 'AB', 'explicit labeling'
        assertSequenceEqual(M('0 1 \n 2 3').column_symbols, [0, 1]), 'implicit column indices'

    def test_reorder_rows(self):
        actual = M('0 \n 2 \n 1', 'ACB', '_')._reorder_rows('ABC')
        assert actual == M('0 \n 1 \n 2', 'ABC', '_')

    def test_mult_matrix(self):
        A = M('0 0\n0 0', 'AB', 'XY')
        B = M('0 0\n0 0', 'XY', 'CD')
        actual = A * B
        # The result will have A's row_symbols and B's column_symbols.
        assertSequenceEqual(actual.row_symbols, 'AB')
        assertSequenceEqual(actual.column_symbols, 'CD')
        flip = M('0 1\n1 0')
        assert flip * flip == M('1 0\n0 1')

    def test_add_matrix(self):
        assert M('1') + M('1') == M('2')

