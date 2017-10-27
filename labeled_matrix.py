import numbers
import unittest
import pytest
import pymatrix
assertSequenceEqual = unittest.TestCase('__init__').assertSequenceEqual
skip = pytest.mark.skip

class M():
    def __init__(self, m, row_symbols=None, column_symbols=None):
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

    def __mul__(self, other):
        ''' Implements matrix multiplication with labeled rows and cols. '''
        if isinstance(other, M):
            to_match = self.row_symbols
            current_columns = other.column_symbols
            raise NotImplementedError
        if isinstance(other, numbers.Number):
            return M(self._matrix * other, self.row_symbols, self.column_symbols)
    
    def __rmul__(self, other):
        return self * other
    
    #     r = {}
    #     for row in matrix:
    #         r[column_symbols[row]] = ble

    # def __mult__(self, other):
    #     fix_rows(matrix, row_permutation)
    #     return M.from_matrix(A * B, A.column_symbols, B.row_symbols)
    #     #[3,2, 1, 3]

class TestM():
    def test_auto_matrix(self):
        assert M(pymatrix.matrix('0')) == M('0')

    def test_eq(self):
        assert M('0', 'A', 'X') == M('0', 'A', 'X'), 'should equal copy pasted'
        assert M('9 9 9') != M('0 1 2'), 'different matrix content'
        assert M('0', 'A', 'X') != M('0', 'B', 'X'), 'different row symbols'
        assert M('0', 'A', 'X') != M('0', 'A', 'Y'), 'different column symbols'

    def test_mult_number(self):
        example = M('1 2')
        expected = M('2 4')
        assert example * 2 == expected, 'matrix on left'
        assert 2 * example == expected, 'matrix on right'

    def test_mult_matrix(self):
        A = M('0 0\n0 0', 'AB', 'XY')
        B = M('0 0\n0 0', 'XY', 'CD')
        actual = A * B
        # The result will have A's row_symbols and B's column_symbols.
        assertSequenceEqual(actual.row_symbols, 'AB')
        assertSequenceEqual(actual.column_symbols, 'CD')

