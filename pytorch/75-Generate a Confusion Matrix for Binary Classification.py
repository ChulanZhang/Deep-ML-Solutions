def confusion_matrix(data):
    """
    Generate a 2x2 confusion matrix for binary classification.
    
    Args:
        data: A list of lists where each inner list is [y_true, y_pred]
              1 represents Positive class, 0 represents Negative class.
              
    Returns:
        A 2x2 list of lists representing the confusion matrix:
        [[True Positives (TP), False Negatives (FN)],
         [False Positives (FP), True Negatives (TN)]]
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    for y_true, y_pred in data:
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
            
    return [[tp, fn], [fp, tn]]
