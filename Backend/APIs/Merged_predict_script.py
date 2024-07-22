# Merged_predict_script.py
from decimal import Decimal

def get_merged_output(ann_prob, cnn_prob):
    ann_prob = Decimal(float(ann_prob))
    cnn_prob = Decimal(float(cnn_prob))

    merged_output = 0
    merged_prob = 0

    if ann_prob and cnn_prob == 0.5:
        merged_output = 1
        merged_prob = ann_prob
    elif ann_prob >= 0.5 and cnn_prob >= 0.5:
        merged_output = 1
        merged_prob = max(ann_prob, cnn_prob)
    elif ann_prob < 0.5 and cnn_prob < 0.5:
        merged_output = 0
        merged_prob = min(ann_prob, cnn_prob)
    elif ann_prob >= 0.5 and cnn_prob < 0.5:
        if (1 - ann_prob) > cnn_prob:
            merged_output = 0
            merged_prob = cnn_prob
        else:
            merged_output = 1
            merged_prob = ann_prob
    elif ann_prob < 0.5 and cnn_prob >= 0.5:
        if (1 - cnn_prob) > ann_prob:
            merged_output = 0
            merged_prob = ann_prob
        else:
            merged_output = 1
            merged_prob = cnn_prob

    return merged_output, merged_prob
