#!/usr/bin/env python3
"""early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """early stopping
    
    cost - current validation cost of net
    opt_cost - lowest recorded validation cost
    threshold - threshold used to stop early
    patience - patience count
    count - how long the threshold has not been met
    """

    # false route
    count += 1
    if cost < opt_cost - threshold:
        return False, 0
    elif count >= patience:
        return True, count
    return False, count
