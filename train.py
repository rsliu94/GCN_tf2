import numpy as np
from GCN.layer import GCN
import tensorflow as tf
from utils import *

if __name__ == "__main__":
    path = "data/cora/"
    data_set = "cora"
    features, adj, labels, class_dict = load_data(path, data_set)
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask = get_splits(labels)
    A = preprocess_adj(adj)     # A_hat = D_^-0.5 * A_ * D_^-0.5
    X = features/features.sum(1).reshape(-1, 1)  # row-normalization of feature matrix

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    A = tf.convert_to_tensor(A, dtype=tf.float32)

    feature_dim = X.shape[-1]       # 2078*1433, feature_dim=1433
    num_class = y_train.shape[1]
    hidden_dim = 16

    model = GCN(hidden_dim=hidden_dim, out_dim=num_class, l2_reg=2.5e-4)
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='auto')
    metric = tf.keras.metrics.CategoricalAccuracy()

    MAX_EPOCH = 300

    print("start training")

    for epoch in range(MAX_EPOCH):
        with tf.GradientTape() as tape:
            y_pred = model([X, A])
            loss = criterion(labels, y_pred, sample_weight=train_mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        val_loss = criterion(labels, y_pred, sample_weight=val_mask)
        if (epoch+1) % 20 == 0:
            metric.update_state(labels, y_pred, sample_weight=test_mask)
            print("In epoch {}, train loss = {}, val_loss = {}, "
                  "current accuracy on test set = {}".format(epoch+1, loss, val_loss, metric.result().numpy()))

    res = model([X, A])
    y_true = labels
    y_pred = res
    metric.update_state(y_true, y_pred, sample_weight=test_mask)
    test_loss = criterion(y_true, y_pred, sample_weight=test_mask)
    print('Final accuracy on test set: ', metric.result().numpy())  # Final result: 0.792
    print('Final test loss: ', test_loss.numpy())





