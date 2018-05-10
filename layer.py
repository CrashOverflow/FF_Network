class layer:
    """CLASSE LAYER"""
    def __init__(self, weights_matrix, b, g):
        self.weights_matrix = weights_matrix
        self.b = b
        self.g = g

    def print_weights_matrix(self):
        print(self.weights_matrix)

    def print_bias(self):
        print(self.b)
