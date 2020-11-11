#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(start=-1.3, stop=2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    # for c in cartesian_coordinates:
    #     radius = np.sqrt(c[0]**2 + c[1]**2)
    #     angle = np.arctan2(c[1], c[0])

    return np.array([(np.sqrt(coordinates[0] ** 2 + coordinates[1] ** 2), np.arctan2(c[1], c[0])) for coordinates in cartesian_coordinates])


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.absolute(values - number).argmin()


def create_plot():
    x = np.linspace(-1, 1, num=250)
    y = x ** 2 * np.sin(1 / x ** 2) + x

    plt.scatter(x, y, label="scatter")
    # plt.xlim((-2, 2))
    plt.plot(x, y, label="line", color="r")
    # plt.title("Titre")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    plt.legend()
    plt.show()
    # Scatter: Plot all the points. Plot: Plot the line


def monte_carlo(iteration: int = 5000000) -> float:
    x_inside_dots = []
    y_inside_dots = []
    x_outside_dots = []
    y_outside_dots = []
    # Need it to be this format to use mathplot properly

    for i in range(iteration):
        x = np.random.random()
        y = np.random.random()
        if np.sqrt(x ** 2 + y ** 2) < 1:
            x_inside_dots.append(x)
            y_inside_dots.append(y)
        else:
            x_outside_dots.append(x)
            y_outside_dots.append(y)

    plt.scatter(x_inside_dots, y_inside_dots, label="Inside dots")
    plt.scatter(x_outside_dots, y_outside_dots, label="Outside dots")
    plt.title("Calculation of Monte-Carlo")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    return float(len(x_inside_dots)) / iteration * 4


# Multiply by 4 because it's only 1/4 of the circle
# Could use y as well

def integral() -> tuple:
    result_inf = integrate.quad(lambda x: np.exp(-x ** 2), -np.inf, +np.inf)

    x = np.arange(-4, 4, 0.1)

    y = [integrate.quad(lambda x: np.exp(-x ** 2), 0, value)[0] for value in x]
    plt.plot(x, y)
    plt.show()
    return result_inf


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    # print(linear_values())
    # print(coordinate_conversion(np.array([(0, 0), (3, 4), (1, 2)])))
    # print(find_closest_index(np.array([0, 5, 10, 12, 8]), 10.5))
    # create_plot()
    # print(monte_carlo())
    print(integral())
    pass
