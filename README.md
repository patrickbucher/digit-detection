# Digit Classification

Simple digit classifier (using logistic regression) from scratch.

Training:

    $ ./train.py
    train classifier 0 with 50 examples for 20000 iterations
    train classifier 1 with 50 examples for 20000 iterations
    train classifier 2 with 50 examples for 20000 iterations
    train classifier 3 with 50 examples for 20000 iterations
    train classifier 4 with 50 examples for 20000 iterations
    train classifier 5 with 50 examples for 20000 iterations
    train classifier 6 with 50 examples for 20000 iterations
    train classifier 7 with 50 examples for 20000 iterations
    train classifier 8 with 50 examples for 20000 iterations
    train classifier 9 with 50 examples for 20000 iterations
    accuracy for digit 0: 100.00%
    accuracy for digit 1: 100.00%
    accuracy for digit 2: 100.00%
    accuracy for digit 3: 100.00%
    accuracy for digit 4: 100.00%
    accuracy for digit 5: 100.00%
    accuracy for digit 6: 100.00%
    accuracy for digit 7: 100.00%
    accuracy for digit 8: 100.00%
    accuracy for digit 9: 100.00%
    saved weights as CSV to weights.csv

Prediction:

    $ ./predict.py
    loaded weights from weights.csv

    image 0 has label 0
    prediction: 3 with 69.743%
        P(0) =   0.021%
        P(1) =   0.000%
        P(2) =   0.486%
        P(3) =  69.743%
        P(4) =   0.000%
        P(5) =   0.000%
        P(6) =   0.000%
        P(7) =   0.000%
        P(8) =   0.000%
        P(9) =   9.180%

    image 1 has label 1
    prediction: 1 with 100.000%
        P(0) =  61.748%
        P(1) = 100.000%
        P(2) =  99.982%
        P(3) =  81.553%
        P(4) =   0.000%
        P(5) =   0.003%
        P(6) =   1.218%
        P(7) =   0.119%
        P(8) =   0.000%
        P(9) =  48.384%

    image 2 has label 2
    prediction: 2 with 99.583%
        P(0) =   0.000%
        P(1) =   0.000%
        P(2) =  99.583%
        P(3) =   0.000%
        P(4) =   0.000%
        P(5) =   0.000%
        P(6) =   0.000%
        P(7) =   0.000%
        P(8) =   0.000%
        P(9) =   0.000%

    image 3 has label 3
    prediction: 3 with 100.000%
        P(0) =   0.034%
        P(1) =   0.000%
        P(2) =   0.199%
        P(3) = 100.000%
        P(4) =   0.000%
        P(5) =  54.518%
        P(6) =   0.000%
        P(7) =   0.000%
        P(8) =  99.544%
        P(9) =   2.446%

    image 4 has label 4
    prediction: 4 with 0.000%
        P(0) =   0.000%
        P(1) =   0.000%
        P(2) =   0.000%
        P(3) =   0.000%
        P(4) =   0.000%
        P(5) =   0.000%
        P(6) =   0.000%
        P(7) =   0.000%
        P(8) =   0.000%
        P(9) =   0.000%

    image 5 has label 5
    prediction: 5 with 100.000%
        P(0) =   0.000%
        P(1) =   0.000%
        P(2) =   0.000%
        P(3) =   3.456%
        P(4) =   4.889%
        P(5) = 100.000%
        P(6) =  99.929%
        P(7) =   0.000%
        P(8) =  99.997%
        P(9) = 100.000%

    image 6 has label 6
    prediction: 6 with 100.000%
        P(0) =   0.000%
        P(1) =   0.001%
        P(2) =   0.000%
        P(3) =   2.991%
        P(4) =   0.077%
        P(5) =   0.040%
        P(6) = 100.000%
        P(7) =   0.000%
        P(8) =   0.001%
        P(9) =   0.000%

    image 7 has label 7
    prediction: 7 with 100.000%
        P(0) =   0.000%
        P(1) =   0.000%
        P(2) =   0.000%
        P(3) =   0.000%
        P(4) =   0.000%
        P(5) =   0.000%
        P(6) =   0.000%
        P(7) = 100.000%
        P(8) =   0.000%
        P(9) =   0.000%

    image 8 has label 8
    prediction: 8 with 100.000%
        P(0) =   0.000%
        P(1) =   0.000%
        P(2) =   0.000%
        P(3) =   0.000%
        P(4) =   0.000%
        P(5) =   0.000%
        P(6) =   0.000%
        P(7) =   0.000%
        P(8) = 100.000%
        P(9) =   0.000%

    image 9 has label 9
    prediction: 8 with 2.427%
        P(0) =   0.000%
        P(1) =   0.000%
        P(2) =   0.000%
        P(3) =   0.000%
        P(4) =   0.000%
        P(5) =   0.000%
        P(6) =   0.000%
        P(7) =   0.000%
        P(8) =   2.427%
        P(9) =   1.447%
