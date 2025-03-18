# Knowledge Distillation Report

## General Information
- Number of models tested: 4
- Temperatures tested: [0.5, 1.0, 2.0]
- Alpha values tested: [0.3, 0.5, 0.7]
- Total configurations: 36
- Valid results: 36
- Report date: 2025-03-17 23:07:38

## Best Configurations
### Best Model by Test Accuracy
- Model Type: GBM
- Temperature: 1.0
- Alpha: 0.3
- Test Accuracy: 0.9912280701754386
- Test Precision: 0.9864864864864865
- Test Recall: 1.0
- Test F1: 0.9931972789115646
- Test AUC-ROC: 0.9976612094888072
- Test AUC-PR: 0.9986483576733771
- KL Divergence (Test): 0.03946954794019648
- KS Statistic (Test): 0.45614035087719296
- KS p-value (Test): 4.5666411744519806e-11
- R² Score (Test): 0.9817913191376207
- Parameters: {'n_estimators': 165, 'learning_rate': 0.014720310807302158, 'max_depth': 7, 'subsample': 0.697191807634784}

### Best Model by Precision
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9824561403508771
- Test Precision: 1.0
- Test Recall: 0.9726027397260274
- Test F1: 0.9861111111111112
- Test AUC-ROC: 1.0
- Test AUC-PR: 1.0000000000000002
- KL Divergence (Test): 0.028359098190600403
- KS Statistic (Test): 0.40350877192982454
- KS p-value (Test): 1.1066231019704113e-08
- R² Score (Test): 0.9794702929765057
- Parameters: {'n_estimators': 143, 'learning_rate': 0.025072439219651146, 'max_depth': 10, 'subsample': 0.9298574139681484, 'colsample_bytree': 0.920902232579487}

### Best Model by Recall
- Model Type: GBM
- Temperature: 1.0
- Alpha: 0.3
- Test Accuracy: 0.9912280701754386
- Test Precision: 0.9864864864864865
- Test Recall: 1.0
- Test F1: 0.9931972789115646
- Test AUC-ROC: 0.9976612094888072
- Test AUC-PR: 0.9986483576733771
- KL Divergence (Test): 0.03946954794019648
- KS Statistic (Test): 0.45614035087719296
- KS p-value (Test): 4.5666411744519806e-11
- R² Score (Test): 0.9817913191376207
- Parameters: {'n_estimators': 165, 'learning_rate': 0.014720310807302158, 'max_depth': 7, 'subsample': 0.697191807634784}

### Best Model by F1 Score
- Model Type: GBM
- Temperature: 1.0
- Alpha: 0.3
- Test Accuracy: 0.9912280701754386
- Test Precision: 0.9864864864864865
- Test Recall: 1.0
- Test F1: 0.9931972789115646
- Test AUC-ROC: 0.9976612094888072
- Test AUC-PR: 0.9986483576733771
- KL Divergence (Test): 0.03946954794019648
- KS Statistic (Test): 0.45614035087719296
- KS p-value (Test): 4.5666411744519806e-11
- R² Score (Test): 0.9817913191376207
- Parameters: {'n_estimators': 165, 'learning_rate': 0.014720310807302158, 'max_depth': 7, 'subsample': 0.697191807634784}

### Best Model by AUC-ROC
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9824561403508771
- Test Precision: 1.0
- Test Recall: 0.9726027397260274
- Test F1: 0.9861111111111112
- Test AUC-ROC: 1.0
- Test AUC-PR: 1.0000000000000002
- KL Divergence (Test): 0.028359098190600403
- KS Statistic (Test): 0.40350877192982454
- KS p-value (Test): 1.1066231019704113e-08
- R² Score (Test): 0.9794702929765057
- Parameters: {'n_estimators': 143, 'learning_rate': 0.025072439219651146, 'max_depth': 10, 'subsample': 0.9298574139681484, 'colsample_bytree': 0.920902232579487}

### Best Model by AUC-PR
- Model Type: XGB
- Temperature: 1.0
- Alpha: 0.5
- Test Accuracy: 0.9824561403508771
- Test Precision: 1.0
- Test Recall: 0.9726027397260274
- Test F1: 0.9861111111111112
- Test AUC-ROC: 1.0
- Test AUC-PR: 1.0000000000000002
- KL Divergence (Test): 0.028359098190600403
- KS Statistic (Test): 0.40350877192982454
- KS p-value (Test): 1.1066231019704113e-08
- R² Score (Test): 0.9794702929765057
- Parameters: {'n_estimators': 143, 'learning_rate': 0.025072439219651146, 'max_depth': 10, 'subsample': 0.9298574139681484, 'colsample_bytree': 0.920902232579487}

### Best Model by KL Divergence
- Model Type: GBM
- Temperature: 0.5
- Alpha: 0.7
- Test Accuracy: 0.9824561403508771
- Test Precision: 0.9863013698630136
- Test Recall: 0.9863013698630136
- Test F1: 0.9863013698630136
- Test AUC-ROC: 0.9993317741396592
- Test AUC-PR: 0.9996272309088972
- KL Divergence (Test): 0.022456536255978744
- KS Statistic (Test): 0.40350877192982454
- KS p-value (Test): 1.1066231019704113e-08
- R² Score (Test): 0.9898142083819587
- Parameters: {'n_estimators': 143, 'learning_rate': 0.025261220878642583, 'max_depth': 4, 'subsample': 0.7263983527691179}

### Best Model by KS Statistic
- Model Type: DECISION_TREE
- Temperature: 2.0
- Alpha: 0.3
- Test Accuracy: 0.956140350877193
- Test Precision: 0.9722222222222222
- Test Recall: 0.958904109589041
- Test F1: 0.9655172413793104
- Test AUC-ROC: 0.9787838289341798
- Test AUC-PR: 0.9856207540170914
- KL Divergence (Test): 0.4351960095716805
- KS Statistic (Test): 0.15789473684210525
- KS p-value (Test): 0.11666226428730296
- R² Score (Test): 0.980100592666083
- Parameters: {'max_depth': 6, 'min_samples_split': 18, 'min_samples_leaf': 6}

### Best Model by R² Score
- Model Type: GBM
- Temperature: 0.5
- Alpha: 0.3
- Test Accuracy: 0.9824561403508771
- Test Precision: 0.9863013698630136
- Test Recall: 0.9863013698630136
- Test F1: 0.9863013698630136
- Test AUC-ROC: 0.9979953224189776
- Test AUC-PR: 0.9988498081165679
- KL Divergence (Test): 0.024935403951100422
- KS Statistic (Test): 0.2894736842105263
- KS p-value (Test): 0.0001292251272558313
- R² Score (Test): 0.9922797592872288
- Parameters: {'n_estimators': 59, 'learning_rate': 0.06354361487043671, 'max_depth': 9, 'subsample': 0.764829896478936}

## Model Comparison
```
            model_type test_accuracy                     train_accuracy                     test_precision                     test_recall                       test_f1                     test_auc_roc                     test_auc_pr                     test_kl_divergence                    
                                mean       max       std           mean       max       std           mean       max       std        mean       max       std      mean       max       std         mean       max       std        mean       max       std               mean       min       std
0        DECISION_TREE      0.953216  0.964912  0.006203       0.965079  0.982418  0.007958       0.962370  0.972603  0.012028    0.964992  0.986301  0.018265  0.963484  0.972603  0.005173     0.980863  0.990144  0.008972    0.985216  0.992827  0.009254           0.383178  0.298683  0.093327
1                  GBM      0.985380  0.991228  0.006203       0.998779  1.000000  0.003663       0.984903  0.986486  0.004475    0.992390  1.000000  0.007220  0.988621  0.993197  0.004832     0.998144  0.999666  0.001696    0.998932  0.999815  0.000987           0.058193  0.022457  0.057332
2  LOGISTIC_REGRESSION      0.931774  0.947368  0.012232       0.959707  0.975824  0.014906       0.965401  0.985507  0.014862    0.926941  0.958904  0.016777  0.945634  0.957746  0.009897     0.986450  0.990979  0.006253    0.992356  0.995172  0.004339           0.130671  0.100427  0.020620
3                  XGB      0.977583  0.991228  0.007736       0.996825  1.000000  0.002930       0.987758  1.000000  0.008171    0.977169  0.986301  0.009686  0.982396  0.993103  0.006112     0.999295  1.000000  0.000424    0.999607  1.000000  0.000238           0.031388  0.025171  0.008789
```

## Impact of Temperature
```
             model_type  temperature  test_accuracy  test_precision  test_recall   test_f1  test_auc_roc  test_auc_pr  test_kl_divergence
0         DECISION_TREE          0.5       0.956140        0.947368     0.986301  0.966443      0.973104     0.976516            0.451588
1         DECISION_TREE          1.0       0.953216        0.967779     0.958904  0.963280      0.987137     0.990895            0.346679
2         DECISION_TREE          2.0       0.950292        0.971961     0.949772  0.960728      0.982348     0.988236            0.351267
3                   GBM          0.5       0.982456        0.986301     0.986301  0.986301      0.998441     0.999109            0.029353
4                   GBM          1.0       0.988304        0.986425     0.995434  0.990899      0.998107     0.998906            0.037036
5                   GBM          2.0       0.985380        0.981982     0.995434  0.988662      0.997884     0.998780            0.108189
6   LOGISTIC_REGRESSION          0.5       0.941520        0.976121     0.931507  0.953281      0.989197     0.994157            0.144356
7   LOGISTIC_REGRESSION          1.0       0.935673        0.971291     0.926941  0.948587      0.988083     0.993579            0.130474
8   LOGISTIC_REGRESSION          2.0       0.918129        0.948792     0.922374  0.935034      0.982069     0.989331            0.117183
9                   XGB          0.5       0.973684        0.986109     0.972603  0.979278      0.999109     0.999504            0.028324
10                  XGB          1.0       0.979532        0.986425     0.981735  0.984001      0.999555     0.999756            0.027872
11                  XGB          2.0       0.979532        0.990741     0.977169  0.983908      0.999220     0.999562            0.037967
```

## Impact of Alpha
```
             model_type  alpha  test_accuracy  test_precision  test_recall   test_f1  test_auc_roc  test_auc_pr  test_kl_divergence
0         DECISION_TREE    0.3       0.959064        0.964064     0.972603  0.968188      0.978283     0.982563            0.447200
1         DECISION_TREE    0.5       0.950292        0.959368     0.963470  0.961227      0.979062     0.983088            0.386755
2         DECISION_TREE    0.7       0.950292        0.963677     0.958904  0.961037      0.985243     0.989996            0.315579
3                   GBM    0.3       0.982456        0.981920     0.990868  0.986363      0.996659     0.998070            0.089011
4                   GBM    0.5       0.985380        0.986363     0.990868  0.988600      0.998998     0.999431            0.035070
5                   GBM    0.7       0.988304        0.986425     0.995434  0.990899      0.998775     0.999295            0.050497
6   LOGISTIC_REGRESSION    0.3       0.929825        0.966460     0.922374  0.943892      0.989197     0.994169            0.127462
7   LOGISTIC_REGRESSION    0.5       0.938596        0.963423     0.940639  0.951580      0.981624     0.989068            0.145999
8   LOGISTIC_REGRESSION    0.7       0.926901        0.966322     0.917808  0.941429      0.988529     0.993831            0.118551
9                   XGB    0.3       0.982456        0.986425     0.986301  0.986332      0.999443     0.999692            0.028377
10                  XGB    0.5       0.976608        0.990741     0.972603  0.981577      0.999555     0.999751            0.030408
11                  XGB    0.7       0.973684        0.986109     0.972603  0.979278      0.998886     0.999379            0.035378
```

## Summary and Recommendations
### Key Findings
- Best accuracy achieved by GBM with temperature=1.0 and alpha=0.3
- Best distribution matching achieved by GBM with temperature=0.5 and alpha=0.7
- For accuracy metrics, optimal temperature is around 1.0 and alpha around 0.3
- For distribution matching, optimal temperature is around 1.0 and alpha around 0.4

### Recommendations
Based on the results, we recommend:
- Model type: GBM
- Temperature: 1.0
- Alpha: 0.4