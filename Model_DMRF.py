import tensorflow as tf

from Evaluation import evaluation


def create_dmrf_network(input_shape, num_classes):
    # Define the Deep Markov Random Field network architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_dmrf_network(model, X_train, y_train, X_test, y_test, num_epochs, batch_size):
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)

    # Obtain predictions on the test data
    predictions = model.predict(X_test)

    return predictions


def Model_DMRF(X_train, y_train, X_test, y_test, sol=None):
    if sol is None:
        sol = [5]

    # Usage example
    input_shape = (28, 28, 1)
    num_classes = y_test.shape[1]
    num_epochs = 10
    batch_size = 32

    # Create the DMRF network
    dmrf_net = create_dmrf_network(input_shape, num_classes)

    # Train the DMRF network and obtain predictions
    out = train_dmrf_network(dmrf_net, X_train, y_train, X_test, y_test, num_epochs=10, batch_size=32)
    Eval = evaluation(X_test, y_test)
    return Eval, out

# Additional resources for further exploration:
# 1. TensorFlow Documentation: https://www.tensorflow.org/
# 2. Keras Documentation: https://keras.io/
# 3. Deep Learning with Python Book: https://www.manning.com/books/deep-learning-with-python
# 4. Convolutional Neural Networks (CNNs) in TensorFlow: https://www.tensorflow.org/guide/keras/sequential_model
# 5. Dense Neural Networks in TensorFlow: https://www.tensorflow.org/guide/keras/functional
