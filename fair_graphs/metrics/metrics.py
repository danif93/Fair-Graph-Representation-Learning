# ----- Standard Imports

# ----- Third-Party Imports
from torch.nn.functional import cosine_similarity
from sklearn.metrics import accuracy_score, roc_auc_score

# ----- Library Imports


def demographic_parity(predictions, sensitive, sens_value):
    group_0 = sensitive == sens_value
    group_1 = ~group_0
    parity = abs(predictions[group_0].float().mean() - predictions[group_1].float().mean())
    parity = -parity    
    return round(parity, 3)


def equal_opportunity(predictions, sensitive, labels, sens_value, label_value):
    label_mask = labels == label_value
    group_0_lbl = (sensitive == sens_value) & label_mask
    group_1_lbl = (sensitive != sens_value) & label_mask
    equality = abs(predictions[group_0_lbl].float().mean() - predictions[group_1_lbl].float().mean())
    equality = -equality
    return round(equality, 3)


def counterfactual_fairness(predictions, counterfactual_predictions):
    score = abs(predictions - counterfactual_predictions).mean()
    score = -score
    return round(score, 3)


def accuracy(true_labels, predicted):
    return round(accuracy_score(true_labels, predicted), 3)


def roc(true_labels, probabilities):
    return round(roc_auc_score(true_labels, probabilities), 3)


def cosine_distance(x1, x2):
    return -cosine_similarity(x1, x2.detach(), dim=-1).mean()