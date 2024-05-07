#!/usr/bin/env python3
"""does a stacked bar graph"""
import numpy as np
import matplotlib.pyplot as plt

def bars():
    """does a stacked bar graph"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))
    x = ["Farrah", "Fred", "Felicia"]
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    b_oranges = list(np.add(apples, bananas))
    b_peaches = list(np.add(b_oranges, oranges))
    plt.bar(x, apples, color='r', label='apples', width=0.5)
    plt.bar(x, bananas, bottom=apples, color='yellow', label='bananas', width=0.5)
    plt.bar(x, oranges, color='#ff8000', bottom=b_oranges, label='oranges', width=0.5)
    plt.bar(x, peaches, color='#ffe5b4', bottom=b_peaches, label='peaches', width=0.5)
    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.yticks(np.linspace(0, 80, 9))
    plt.legend()
    plt.show()
