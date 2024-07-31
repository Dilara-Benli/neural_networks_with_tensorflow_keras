import tensorflow as tf

def F1_Score(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    
    tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred, 'float32'))
    fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred, 'float32'))
    fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred), 'float32'))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    return f1

def specificity(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    
    confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=2, dtype=tf.float32) # 2 class
    tn = tf.keras.backend.cast(confusion[0, 0], dtype=tf.float32)
    fp = tf.keras.backend.cast(confusion[0, 1], dtype=tf.float32)
    
    specificity = tn / (tn + fp + tf.keras.backend.epsilon())
    return specificity

def cohen_kappa(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    
    confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=2, dtype=tf.float32) # 2 class
    total = tf.keras.backend.sum(confusion)
    row_sum = tf.keras.backend.sum(confusion, axis=1)
    col_sum = tf.keras.backend.sum(confusion, axis=0)
    pe = tf.keras.backend.sum(row_sum * col_sum) / (total ** 2)
    po = tf.keras.backend.sum(tf.linalg.diag_part(confusion)) / total
    
    kappa_value = (po - pe) / (1 - pe + tf.keras.backend.epsilon())
    return kappa_value
