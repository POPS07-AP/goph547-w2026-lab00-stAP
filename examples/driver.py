import numpy as np
from goph547lab00.arrays import (
    square_ones,
)

def main():
    # test creating square array of ones
    A_np = np.ones((3,3))
    A = square_ones(3)
    print(f'A_np:\n{A_np}')
    print(f'A:\n{A}')
if __name__ == '__main__':
 main()