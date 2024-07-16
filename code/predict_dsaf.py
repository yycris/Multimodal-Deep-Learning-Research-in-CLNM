import csv
import torch
import config
from torch.utils.data import DataLoader
from dsaf_dataloader import CSVDataset
from model import dsaf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score
from sklearn.utils import resample




activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    train_clinic_path = ''
    train_tumor_dl_path = ''
    train_clinical_feature_path = ''

    val_clinic_path = ''
    val_tumor_dl_path = ''
    val_clinical_feature_path = ''

    itest_clinic_path = ''
    itest_tumor_dl_path = ''
    itest_clinical_feature_path = ''

    etest_clinic_path = ''
    etest_tumor_dl_path = ''
    etest_clinical_feature_path = ''

    train_dataset = CSVDataset(train_tumor_dl_path, train_clinical_feature_path)
    val_dataset = CSVDataset(val_tumor_dl_path, val_clinical_feature_path)
    itest_dataset = CSVDataset(itest_tumor_dl_path, itest_clinical_feature_path)
    etest_dataset = CSVDataset(etest_tumor_dl_path, etest_clinical_feature_path)

    train_loader = DataLoader(train_dataset, **config.dataset_params)
    val_loader = DataLoader(val_dataset, **config.dataset_params)
    itest_loader = DataLoader(itest_dataset, **config.dataset_params)
    etest_loader = DataLoader(etest_dataset, **config.dataset_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = dsaf.Deep_fusion_classifier(2).to(device)
    predict_pro = []
    predict_cla = []
    test_label = []
    model_weight_path = f""
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    net.eval()
    with torch.no_grad():
        for (tumor_dl, fat_dl, labs) in etest_loader:
            # clinical_feature = clinical_feature.float()
            tumor_dl = tumor_dl.float()
            fat_dl = fat_dl.float()
            labels = labs.to(torch.int64)

            output = torch.squeeze(net(tumor_dl.to(device), fat_dl.to(device)))
            predict = torch.softmax(output, dim=0)
            output = predict[1]
            predict = predict[1]
            threshold = 0.4
            output = torch.where(output >= threshold, torch.ones_like(output), output)
            output = torch.where(output < threshold, torch.zeros_like(output), output)
            predict_class = output.cpu().numpy()
            predict_probability = predict.cpu().numpy()
            labels = labels.numpy()
            predict_pro.append(predict_probability)
            predict_cla.append(predict_class)
            test_label.append(labels)

            # net.fc1.register_forward_hook(get_activation('fc1'))
            # list = []
            # list.append(labels[0])
            #
            # tensor = activation['fc1'].cpu()
            # tensor = torch.squeeze(tensor)
            # # t = tensor[0].float().item()
            #
            # for x in range(0, len(tensor)):
            #     list.append(tensor[x].float().item())
            #
            # with open("./result/deep_fusion_classifier/etest_tsne.csv", "a", newline='', encoding='utf-8') as file:
            #     writer = csv.writer(file, delimiter=',')
            #     writer.writerow(list)

    matrix = confusion_matrix(test_label, predict_cla, labels=[0, 1])  #
    tn, fp, fn, tp = matrix.ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = accuracy_score(test_label, predict_cla)  #
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    AUC = roc_auc_score(test_label, predict_pro)  #

    y_true = np.array(test_label)
    y_pred = np.array(predict_cla)
    y_score = np.array(predict_pro)

    n_bootstraps = 1000
    auc_scores = []
    acc_scores = []
    sen_scores = []
    spe_scores = []
    ppv_scores = []
    npv_scores = []

    for _ in range(n_bootstraps):
        indices = resample(range(len(y_true)))
        y_true_sampled = y_true[indices]
        y_pred_sampled = y_pred[indices]
        y_score_sampled = y_score[indices]

        auc_sampled = roc_auc_score(y_true_sampled, y_score_sampled)
        acc_sampled = accuracy_score(y_true_sampled, y_pred_sampled)
        sen_sampled = recall_score(y_true_sampled, y_pred_sampled)
        conf_matrix_sampled = confusion_matrix(y_true_sampled, y_pred_sampled)
        tn_sampled, fp_sampled, fn_sampled, tp_sampled = conf_matrix_sampled.ravel()
        spe_sampled = tn_sampled / (tn_sampled + fp_sampled)
        ppv_sampled = tp_sampled / (tp_sampled + fp_sampled)
        npv_sampled = tn_sampled / (tn_sampled + fn_sampled)

        auc_scores.append(auc_sampled)
        acc_scores.append(acc_sampled)
        sen_scores.append(sen_sampled)
        spe_scores.append(spe_sampled)
        ppv_scores.append(ppv_sampled)
        npv_scores.append(npv_sampled)


    alpha = 0.95
    lower_percentile = (1.0 - alpha) / 2.0 * 100
    upper_percentile = (alpha + (1.0 - alpha) / 2.0) * 100

    auc_lower_bound = max(0.0, np.percentile(auc_scores, lower_percentile))
    auc_upper_bound = min(1.0, np.percentile(auc_scores, upper_percentile))

    acc_lower_bound = max(0.0, np.percentile(acc_scores, lower_percentile))
    acc_upper_bound = min(1.0, np.percentile(acc_scores, upper_percentile))

    sen_lower_bound = max(0.0, np.percentile(sen_scores, lower_percentile))
    sen_upper_bound = min(1.0, np.percentile(sen_scores, upper_percentile))

    spe_lower_bound = max(0.0, np.percentile(spe_scores, lower_percentile))
    spe_upper_bound = min(1.0, np.percentile(spe_scores, upper_percentile))

    ppv_lower_bound = max(0.0, np.percentile(ppv_scores, lower_percentile))
    ppv_upper_bound = min(1.0, np.percentile(ppv_scores, upper_percentile))

    npv_lower_bound = max(0.0, np.percentile(npv_scores, lower_percentile))
    npv_upper_bound = min(1.0, np.percentile(npv_scores, upper_percentile))

    print('AUC:', AUC)
    print(f"95% CI (AUC): ({auc_lower_bound:.4f}, {auc_upper_bound:.4f})")
    print('acc: ', acc)
    print(f"95% CI (Accuracy): ({acc_lower_bound:.4f}, {acc_upper_bound:.4f})")
    print('sensitivity: ', sen)
    print(f"95% CI (Sensitivity): ({sen_lower_bound:.4f}, {sen_upper_bound:.4f})")
    print('specificity: ', spe)
    print(f"95% CI (Specificity): ({spe_lower_bound:.4f}, {spe_upper_bound:.4f})")
    print('ppv:', ppv)
    print(f"95% CI (PPV): ({ppv_lower_bound:.4f}, {ppv_upper_bound:.4f})")
    print('npv:', npv)
    print(f"95% CI (NPV): ({npv_lower_bound:.4f}, {npv_upper_bound:.4f})")

if __name__ == '__main__':
    main()