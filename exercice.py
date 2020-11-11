#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.array([])


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([])


def find_closest_index(values: np.ndarray, number: float) -> int:
    return 0

def integral() -> tuple:
    result_inf = integrate.quad(lambda x: np.exp(-x ** 2), -np.inf, +np.inf)

    x = np.arange(-4, 4, 0.1)

    y = [integrate.quad(lambda x: np.exp(-x ** 2), 0, value)[0] for value in x]
    plt.plot(x, y)
    plt.show()
    return result_inf


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici

    pass
