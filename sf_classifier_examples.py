"""
Examples of running sf_classifiers.
"""
import numpy as np
import matplotlib.pyplot as plt
import sf_classifiers


# ----------------------------------------------
# Functions for generating synthetic test data
# ----------------------------------------------

def two_clusters(num_points, noise=.3, show=False):
    """ Synthetic two-class dataset. """
    features = np.random.randint(2, size=(num_points, 1))
    features = np.append(features, features, axis=1)
    labels = np.array(np.logical_or(features[:, 0], features[:, 1]),
                      dtype=np.float32)
    features = np.array(features + np.random.normal(0, noise, features.shape),
                        dtype=np.float32)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(features[labels == 1, 0], features[labels == 1, 1], 's')
        plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'o')
        plt.show()

    return features, labels


def noisy_xor(num_points, show=False):
    """ Synthetic Dataset that is not linearly separable. """

    features = np.random.randint(2, size=(num_points, 2))
    labels = np.array(np.logical_xor(features[:, 0], features[:, 1]),
                      dtype=np.float32)
    features = np.array(features + (np.random.random(features.shape) - .5),
                        dtype=np.float32)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(features[labels == 1, 0], features[labels == 1, 1], 's')
        plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'o')
        plt.show()

    return features, labels


# ----------------------------------------------
# Examples of training models
# ----------------------------------------------


def logistic_regression_clusters(lr=.01, epochs=100):
    dataset_train_x, dataset_train_y = two_clusters(500)
    dataset_test_x, dataset_test_y = two_clusters(500)

    classifier = sf_classifiers.LogisticRegression(2)
    classifier.graph.gen_dot("logres.dot")

    classifier.train(dataset_train_x,
                     dataset_train_y,
                     epochs=epochs, learning_rate=lr)

    classifier.score(dataset_test_x, dataset_test_y)
    classifier.plot_2d_predictions(dataset_train_x, dataset_train_y)


def mlp_xor(lr=.05, epochs=100, activation='sigmoid'):
    dataset_train_x, dataset_train_y = noisy_xor(500)
    dataset_test_x, dataset_test_y = noisy_xor(500)

    classifier = sf_classifiers.MLP(2, [10], activation=activation)

    classifier.train(dataset_train_x, dataset_train_y, epochs=epochs,
                     learning_rate=lr)

    classifier.score(dataset_test_x, dataset_test_y)
    classifier.plot_2d_predictions(dataset_train_x, dataset_train_y)



def run_experiment(activation, hidden_sizes, epochs=100):
    x, y = noisy_xor(500)

    classifier = sf_classifiers.MLP(
        2, hidden_sizes, activation=activation
    )

    return classifier.train(
        x, y, epochs=epochs, learning_rate=0.01
    )



if __name__ == "__main__":
    configurations = [
        ("sigmoid", [10]),
        ("relu", [10]),
        ("sigmoid", [10, 10, 10, 10, 10]),
        ("relu", [10, 10, 10, 10, 10])
    ]

    x, y = noisy_xor(500)

    for activation, hidden_sizes in configurations:
        gradients = []

        for run in range(10):
            classifier = sf_classifiers.MLP(
                2, hidden_sizes, activation=activation
            )

            classifier.train(
                x, y,
                epochs=1,
                learning_rate=0.01
            )

            gradients.append(classifier.first_layer_gradient_mean)

        average = np.mean(gradients)

        print(
            f"{activation} - {len(hidden_sizes)} layers: "
            f"{average:.8e}"
        )