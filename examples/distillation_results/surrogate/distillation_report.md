# Knowledge Distillation Report

## General Information
- Number of models tested: 4
- Temperatures tested: [1.0]
- Alpha values tested: [0.5]
- Total configurations: 4
- Valid results: 3
- Report date: 2025-03-17 23:05:57

## Best Configurations
### Best Model by Test Accuracy
- Model Type: DECISION_TREE
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9714285714285714
- Test Recall: 0.9315068493150684
- Test F1: 0.951048951048951
- Test AUC-ROC: 0.94954894754427
- Test AUC-PR: 0.967987498247578
- KL Divergence (Test): 0.536984568741799
- KS Statistic (Test): 0.32456140350877194
- KS p-value (Test): 1.0330645961231932e-05
- R² Score (Test): 0.8894601369339152
- Parameters: {}

### Best Model by Precision
- Model Type: DECISION_TREE
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9714285714285714
- Test Recall: 0.9315068493150684
- Test F1: 0.951048951048951
- Test AUC-ROC: 0.94954894754427
- Test AUC-PR: 0.967987498247578
- KL Divergence (Test): 0.536984568741799
- KS Statistic (Test): 0.32456140350877194
- KS p-value (Test): 1.0330645961231932e-05
- R² Score (Test): 0.8894601369339152
- Parameters: {}

### Best Model by Recall
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9459459459459459
- Test Recall: 0.958904109589041
- Test F1: 0.9523809523809523
- Test AUC-ROC: 0.9916471767457401
- Test AUC-PR: 0.9951706796371005
- KL Divergence (Test): 0.30908267004893536
- KS Statistic (Test): 0.2894736842105263
- KS p-value (Test): 0.0001292251272558313
- R² Score (Test): 0.9825078118922671
- Parameters: {}

### Best Model by F1 Score
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9459459459459459
- Test Recall: 0.958904109589041
- Test F1: 0.9523809523809523
- Test AUC-ROC: 0.9916471767457401
- Test AUC-PR: 0.9951706796371005
- KL Divergence (Test): 0.30908267004893536
- KS Statistic (Test): 0.2894736842105263
- KS p-value (Test): 0.0001292251272558313
- R² Score (Test): 0.9825078118922671
- Parameters: {}

### Best Model by AUC-ROC
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9459459459459459
- Test Recall: 0.958904109589041
- Test F1: 0.9523809523809523
- Test AUC-ROC: 0.9916471767457401
- Test AUC-PR: 0.9951706796371005
- KL Divergence (Test): 0.30908267004893536
- KS Statistic (Test): 0.2894736842105263
- KS p-value (Test): 0.0001292251272558313
- R² Score (Test): 0.9825078118922671
- Parameters: {}

### Best Model by AUC-PR
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9459459459459459
- Test Recall: 0.958904109589041
- Test F1: 0.9523809523809523
- Test AUC-ROC: 0.9916471767457401
- Test AUC-PR: 0.9951706796371005
- KL Divergence (Test): 0.30908267004893536
- KS Statistic (Test): 0.2894736842105263
- KS p-value (Test): 0.0001292251272558313
- R² Score (Test): 0.9825078118922671
- Parameters: {}

### Best Model by KL Divergence
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9459459459459459
- Test Recall: 0.958904109589041
- Test F1: 0.9523809523809523
- Test AUC-ROC: 0.9916471767457401
- Test AUC-PR: 0.9951706796371005
- KL Divergence (Test): 0.30908267004893536
- KS Statistic (Test): 0.2894736842105263
- KS p-value (Test): 0.0001292251272558313
- R² Score (Test): 0.9825078118922671
- Parameters: {}

### Best Model by KS Statistic
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9459459459459459
- Test Recall: 0.958904109589041
- Test F1: 0.9523809523809523
- Test AUC-ROC: 0.9916471767457401
- Test AUC-PR: 0.9951706796371005
- KL Divergence (Test): 0.30908267004893536
- KS Statistic (Test): 0.2894736842105263
- KS p-value (Test): 0.0001292251272558313
- R² Score (Test): 0.9825078118922671
- Parameters: {}

### Best Model by R² Score
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9385964912280702
- Test Precision: 0.9459459459459459
- Test Recall: 0.958904109589041
- Test F1: 0.9523809523809523
- Test AUC-ROC: 0.9916471767457401
- Test AUC-PR: 0.9951706796371005
- KL Divergence (Test): 0.30908267004893536
- KS Statistic (Test): 0.2894736842105263
- KS p-value (Test): 0.0001292251272558313
- R² Score (Test): 0.9825078118922671
- Parameters: {}

## Model Comparison
```
      model_type test_accuracy           train_accuracy           test_precision           test_recall             test_f1           test_auc_roc           test_auc_pr           test_kl_divergence          
                          mean       max           mean       max           mean       max        mean       max      mean       max         mean       max        mean       max               mean       min
0  DECISION_TREE      0.938596  0.938596       0.971429  0.971429       0.971429  0.971429    0.931507  0.931507  0.951049  0.951049     0.949549  0.949549    0.967987  0.967987           0.536985  0.536985
1            GBM      0.938596  0.938596       0.978022  0.978022       0.971429  0.971429    0.931507  0.931507  0.951049  0.951049     0.987304  0.987304    0.992927  0.992927           0.416592  0.416592
2            XGB      0.938596  0.938596       0.982418  0.982418       0.945946  0.945946    0.958904  0.958904  0.952381  0.952381     0.991647  0.991647    0.995171  0.995171           0.309083  0.309083
```

## Impact of Temperature
```
      model_type  temperature  test_accuracy  test_precision  test_recall   test_f1  test_auc_roc  test_auc_pr  test_kl_divergence
0  DECISION_TREE          1.0       0.938596        0.971429     0.931507  0.951049      0.949549     0.967987            0.536985
1            GBM          1.0       0.938596        0.971429     0.931507  0.951049      0.987304     0.992927            0.416592
2            XGB          1.0       0.938596        0.945946     0.958904  0.952381      0.991647     0.995171            0.309083
```

## Impact of Alpha
```
      model_type  alpha  test_accuracy  test_precision  test_recall   test_f1  test_auc_roc  test_auc_pr  test_kl_divergence
0  DECISION_TREE    0.5       0.938596        0.971429     0.931507  0.951049      0.949549     0.967987            0.536985
1            GBM    0.5       0.938596        0.971429     0.931507  0.951049      0.987304     0.992927            0.416592
2            XGB    0.5       0.938596        0.945946     0.958904  0.952381      0.991647     0.995171            0.309083
```

## Summary and Recommendations
### Key Findings
- Best accuracy achieved by DECISION_TREE with temperature=1.0 and alpha=0.5
- Best distribution matching achieved by XGB with temperature=1.0 and alpha=0.5
- For accuracy metrics, optimal temperature is around 1.0 and alpha around 0.5
- For distribution matching, optimal temperature is around 1.0 and alpha around 0.5

### Recommendations
Based on the results, we recommend:
- Model type: XGB
- Temperature: 1.0
- Alpha: 0.5