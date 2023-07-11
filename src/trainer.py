import time
import tensorflow as tf

def get_classification_loss_fn(loss_id, logits, **kwargs):
    if loss_id == "scce":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=logits)
    elif loss_id == "cce":
        return tf.keras.losses.CategoricalCrossentropy(from_logits=logits)
    elif loss_id == "bce":
        return tf.keras.losses.BinaryCrossentropy(from_logits=logits)
    else:
        raise ValueError(f"Unknown loss_id {loss_id}")


def get_initializer(initializer_id, seed=None, **kwargs):
    if "mean" in kwargs:
        mean = kwargs["mean"]
    else:
        mean = 0.0
    if "stddev" in kwargs:
        stddev = kwargs["stddev"]
    else:
        stddev = 0.05
    if "minval" in kwargs:
        minval = kwargs["minval"]
    else:
        minval = -0.05
    if "maxval" in kwargs:
        maxval = kwargs["maxval"]
    else:
        maxval = 0.05
    if "value" in kwargs:
        value = kwargs["value"]
    else:
        value = 1.0

    if initializer_id == "rnd-normal":
        return tf.keras.initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)
    elif initializer_id == "rnd-uniform":
        return tf.keras.initializers.RandomUniform(
            minval=minval, maxval=maxval, seed=seed
        )
    elif initializer_id == "glorot-normal":
        return tf.keras.initializers.GlorotNormal(seed=seed)
    elif initializer_id == "glorot-uniform":
        return tf.keras.initializers.GlorotUniform(seed=seed)
    elif initializer_id == "he-normal":
        return tf.keras.initializers.HeNormal(seed=seed)
    elif initializer_id == "he-uniform":
        return tf.keras.initializers.HeUniform(seed=seed)
    elif initializer_id == "zeros":
        return tf.keras.initializers.Zeros()
    elif initializer_id == "constant":
        return tf.keras.initializers.Constant(value=value)
    else:
        raise ValueError(f"Unknown initializer_id {initializer_id}")


# wrapping tf function into a class to enable repetitive execution as per
# https://www.tensorflow.org/guide/function#creating_tfvariables
class TestStep(tf.Module):
    def __init__(self):
        self.count = None

    @tf.function
    def __call__(self, model, val_acc_metric, x_batch_val, y_batch_val):
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)


# wrapping tf function into a class to enable repetitive execution as per
# https://www.tensorflow.org/guide/function#creating_tfvariables
class TrainStep(tf.Module):
    def __init__(self):
        self.count = None

    @tf.function
    def __call__(
        self, loss_fn, model, optimizer, train_acc_metric, x_batch_train, y_batch_train
    ):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)
        return loss_value


def train_and_evaluate_model(
    model,
    optimizer,
    loss_fn,
    train_acc_metric,
    val_acc_metric,
    test_acc_metric,
    batch_size,
    train_dg,
    val_dg,
    test_dg,
    epochs,
    initializer,
    log,
):
    train_accuracy = []
    val_accuracy = []
    test_accuracy = []
    time_per_epoch = []
    overall_time = 0.0

    # initialize tf.function classes
    train_step = TrainStep()
    test_step = TestStep()

    for epoch in range(epochs):
        if epoch % 5 == 0:
            log.info("Start of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dg):
            loss_value = train_step(
                loss_fn,
                model,
                optimizer,
                train_acc_metric,
                x_batch_train,
                y_batch_train,
            )

            # Log every 300 batches.
            if step % 300 == 0:
                log.debug(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                log.debug("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        if epoch % 5 == 0:
            log.info("Training acc over epoch: %.4f" % (float(train_acc),))
        train_accuracy.append(float(train_acc))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dg:
            test_step(model, val_acc_metric, x_batch_val, y_batch_val)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        if epoch % 5 == 0:
            log.info("Validation acc: %.4f" % (float(val_acc),))
        val_accuracy.append(float(val_acc))

        # Run a test loop at the end of each epoch.
        for x_batch_test, y_batch_test in test_dg:
            test_step(model, test_acc_metric, x_batch_test, y_batch_test)
        test_acc = test_acc_metric.result()
        test_acc_metric.reset_states()
        if epoch % 5 == 0:
            log.info("Test acc: %.4f" % (float(test_acc),))
        test_accuracy.append(float(test_acc))

        # Time Calculation
        time_taken = time.time() - start_time
        time_per_epoch.append(time_taken)
        overall_time += time_taken
        if epoch % 5 == 0:
            log.info("Time taken: %.2fs" % (time_taken))

    return {
        "overall_time": overall_time,
        "time_per_epoch": time_per_epoch,
        "initializer": initializer,
        "batch_size": batch_size,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
    }