from decimal import Decimal
import ANN_predict_script as ann
import CNN_predict_script as cnn


# ann_output = ann.output
ann_prob = Decimal(float(ann.probablity)) # using decimal library to get the precise floating point value

# cnn_output = cnn.output
cnn_prob = Decimal(float(cnn.probablity))

merged_output = 0
merged_prob = 0

if ann_prob and cnn_prob == 0.5:
    print("case 1")
    merged_output = 1
    merged_prob = ann_prob

elif ann_prob >= 0.5 and cnn_prob >= 0.5:
    print("case 2")
    merged_output = 1
    merged_prob = max(ann_prob, cnn_prob)

elif ann_prob < 0.5 and cnn_prob < 0.5:
    print("case 3")
    merged_output = 0
    merged_prob = min(ann_prob, cnn_prob)

elif ann_prob >= 0.5 and cnn_prob < 0.5:
    print("case 4")
    if (1 - ann_prob) > cnn_prob:
        print("case 4.1")
        merged_output = 0
        merged_prob = cnn_prob
    elif (1 - ann_prob) < cnn_prob:
        print("case 4.2")
        merged_output = 1
        merged_prob = ann_prob
    elif (1 - ann_prob) == cnn_prob:
        print("case 4.3")
        merged_output = 1
        merged_prob = ann_prob

elif ann_prob < 0.5 and cnn_prob >= 0.5:
    print("case 5")
    if (1 - cnn_prob) > ann_prob:
        print("case 5.1")
        merged_output = 0
        merged_prob = ann_prob
    elif (1 - cnn_prob) < ann_prob:
        print("case 5.2")
        merged_output = 1
        merged_prob = cnn_prob
    elif (1 - cnn_prob) == ann_prob:
        print("case 5.3")
        merged_output = 1
        merged_prob = cnn_prob

# print the output
print(f"merged output: {merged_output}, merged prob: {merged_prob}")