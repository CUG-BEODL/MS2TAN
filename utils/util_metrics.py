import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

# plt.rcParams["savefig.dpi"] = 300  # pixel
# plt.rcParams["figure.dpi"] = 300  # resolution
# plt.rcParams["figure.figsize"] = [8, 4]  # figure size


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_mse_cal(inputs, target, mask):
    """calculate Mean Square Error"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """calculate Root Mean Square Error"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """calculate Mean Relative Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (
        torch.sum(torch.abs(target * mask)) + 1e-9
    )


def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = metrics.precision_recall_curve(
        y_true=y_test, probas_pred=y_pred
    )
    area = metrics.auc(recalls, precisions)
    return area, precisions, recalls, thresholds


def auc_roc(y_pred, y_test):
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds


def auc_to_recall(recalls, precisions, recall=0.01):
    precisions_mod = precisions.copy()
    ind = np.where(recalls < recall)[0][0] + 1
    precisions_mod[:ind] = 0
    area = metrics.auc(recalls, precisions_mod)
    return area


def cal_classification_metrics(probabilities, labels, pos_label=1, class_num=1):
    """
    pos_label: The label of the positive class.
    """
    if class_num == 1:
        class_predictions = (probabilities >= 0.5).astype(int)
    elif class_num == 2:
        class_predictions = np.argmax(probabilities, axis=1)
    else:
        assert "args.class_num>2, class need to be specified for precision_recall_fscore_support"
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels, class_predictions, pos_label=pos_label, warn_for=()
    )
    precision, recall, f1 = precision[1], recall[1], f1[1]
    precisions, recalls, _ = metrics.precision_recall_curve(
        labels, probabilities[:, -1], pos_label=pos_label
    )
    acc_score = metrics.accuracy_score(labels, class_predictions)
    ROC_AUC, fprs, tprs, thresholds = auc_roc(probabilities[:, -1], labels)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        "classification_predictions": class_predictions,
        "acc_score": acc_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precisions": precisions,
        "recalls": recalls,
        "fprs": fprs,
        "tprs": tprs,
        "ROC_AUC": ROC_AUC,
        "PR_AUC": PR_AUC,
    }
    return classification_metrics


def plot_AUCs(
    pdf_file, x_values, y_values, auc_value, title, x_name, y_name, dataset_name
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x_values,
        y_values,
        ".",
        label=f"{dataset_name}, AUC={auc_value:.3f}",
        rasterized=True,
    )
    l = ax.legend(fontsize=10, loc="lower left")
    l.set_zorder(20)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title, fontsize=12)
    pdf_file.savefig(fig)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise TypeError("Boolean value expected.")
