import numpy as np

def probability_to_logit(p):
    return np.log(p / (1 - p))


if __name__ == '__main__':

    p = 0.1  # Probability of class 1
    logit_p = probability_to_logit(p)
    print(logit_p)
    print(np.log(0.7) - np.log(1 - 0.7))