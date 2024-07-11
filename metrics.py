def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    TP=0
    TN=0
    FP=0
    FN=0
    c_correct = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            if prediction[i] == True: TP+=1
            else: TN+=1

            c_correct+=1
        else:
            if prediction[i]: FN+=1
            else: FP+=1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = c_correct/len(prediction)
    c = 0.0001
    f1 = 2*precision*recall/max(precision+recall,c)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy

def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''

    c_correct = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]: c_correct+=1
    
    # TODO: Implement computing accuracy
    return c_correct/len(prediction)
