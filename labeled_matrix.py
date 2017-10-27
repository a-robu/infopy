from pymatrix import matrix

class M():
    def __init__(self, matrix, input_space=None, output_space=None):
        self.matrix = matrix
        self.input_space = (input_space if input_space
            else range(matrix.numrows))
        self.output_space = (output_space if output_space
            else range(matrix.numcols))

    def __eq__(self, other):
        return all([
            self.matrix == other.matrix,
            self.input_space == other.input_space,
            self.output_space == other.output_space
        ])
    
    def __mul__(self, other):
        pass
    
    def __rmul__(self, other):
        pass
    
    #     r = {}
    #     for row in matrix:
    #         r[input_space[row]] = ble

    # def __mult__(self, other):
    #     fix_rows(matrix, row_permutation)
    #     return M.from_matrix(A * B, A.input_space, B.output_space)
    #     #[3,2, 1, 3]

class TestM():
    def test_eq(self):
        assert (M(matrix([[0]]), ['A'], ['A']) ==
            M(matrix([[0]]), ['A'], ['A'])), 'should equal copy pasted'
        assert (M(matrix([[1]]), ['A'], ['A']) !=
            M(matrix([[0]]), ['A'], ['A'])), 'different matrix content'
        assert (M(matrix([[0]]), ['A'], ['A']) !=
            M(matrix([[0]]), ['B'], ['A'])), 'different input space'
        assert (M(matrix([[0]]), ['A'], ['A']) !=
            M(matrix([[0]]), ['A'], ['B'])), 'different input space'

    def test_mult_number(self):
        example = M(matrix([
            [1, 2]
        ]))
        expected = M(matrix([
            [2, 4]
        ]))
        assert example * 2, 'matrix on left'
        assert 2 * example, 'matrix on right'
