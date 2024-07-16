import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score
import sklearn as skl
import numpy as np
from sklearn.utils import resample


train = pd.read_csv("./result/deep_fusion_classifier/train_tumor_fat_clinic.csv", encoding='gb18030', header=None)
test = pd.read_csv("./result/deep_fusion_classifier/valid_tumor_fat_clinic.csv", encoding='gb18030', header=None)

train = train.iloc[1:]
train_data = train.iloc[:, 2:].astype(float)
train_label = train.iloc[:, 1].astype(int)

test = test.iloc[1:]
test_data = test.iloc[:, 2:].astype(float)
test_label = test.iloc[:, 1].astype(int)

# classifier = GradientBoostingClassifier(n_estimators=45)
classifier = SVC(C=0.002, kernel='linear', gamma='scale', class_weight=None,probability=True)
# classifier = RandomForestClassifier(n_estimators=375,
#                                     min_samples_leaf=30,
#                                     min_samples_split=60,
#                                     max_depth=7,
#                                     class_weight=None,
#                                     criterion='entropy',
#                                     )
# classifier = LogisticRegression()
# classifier = xgb.XGBClassifier()

classifier.fit(train_data, train_label)
y_pred = classifier.predict(test_data)
y_proba = classifier.predict_proba(test_data)
threshold = 0.59
y_pred = np.where(y_proba[:,1] > threshold, 1, 0).astype(np.uint8)

matrix = skl.metrics.confusion_matrix(test_label, y_pred,labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()
auc1 = skl.metrics.roc_auc_score(test_label, y_proba[:,1])
sen = tp / (tp + fn)
spe = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
acc = accuracy_score(test_label, y_pred)
list=[]

y_true = np.array(test_label)
y_pred1 = np.array(y_pred)
y_score = np.array(y_proba[:,1])

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
    y_pred_sampled = y_pred1[indices]
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

print('AUC:', auc1)
print(f"95% CI (AUC): ({auc_lower_bound:.4f}, {auc_upper_bound:.4f})")
print('acc: ', acc)
print(f"95% CI (Accuracy): ({acc_lower_bound:.4f}, {acc_upper_bound:.4f})")
print('sensitivity: ', sen)
print(f"95% CI (Sensitivity): ({sen_lower_bound:.4f}, {sen_upper_bound:.4f})")
print('specificity: ', spe)
print(f"95% CI (Specificity): ({spe_lower_bound:.4f}, {spe_upper_bound:.4f})")
print('ppv:', ppv)
print(f"95% CI(PPV): ({ppv_lower_bound:.4f}, {ppv_upper_bound:.4f})")
print('npv:', npv)
print(f"95% CI (NPV): ({npv_lower_bound:.4f}, {npv_upper_bound:.4f})")
