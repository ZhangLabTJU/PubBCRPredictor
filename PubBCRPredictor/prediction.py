import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


import PubBCRPredictor
project_path = os.path.dirname(os.path.realpath(PubBCRPredictor.__file__))

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class PubBCRPredictor_Runner():
    def __init__(self,model):
        """
        Initialize public antibody prediction model.

        Args:
            chain: antibody chain, including heavy, light

        Returns:
        """
        self.model = model
        if self.model=='cdrh':
            self.path_model = os.path.join(project_path,'model/prediction/cdrh','publich_p10_n1_10M_mlp_model.pth')
            self.PubBCRPredictor = torch.load(self.path_model, map_location=torch.device('cpu'))
        elif self.model=='cdrh3':
            self.path_model = os.path.join(project_path,'model/prediction/cdrh3','publiccdr3h_p10_n1_10M_mlp_model.pth')
            self.PubBCRPredictor = torch.load(self.path_model, map_location=torch.device('cpu'))
        elif self.model=='cdrl':
            self.path_model = os.path.join(project_path,'model/prediction/cdrl','publiccdrl_p5reg_n1_10M_mlp_model.pth')
            self.PubBCRPredictor = torch.load(self.path_model, map_location=torch.device('cpu'))        
        elif self.model=='cdrl3':
            self.path_model = os.path.join(project_path,'model/prediction/cdrl3','publiccdr3l_p5reg_n1_10M_mlp_model.pth')
            self.PubBCRPredictor = torch.load(self.path_model, map_location=torch.device('cpu'))

    def predict(self, feature):
        """
        predict feature.

        Args:
            feature: embeddings of bcr-v-bert

        Returns:
            list(torch.Tensor): list of prob
        """
        with torch.no_grad():
            if self.model=='cdrh' or self.model=='cdrh3': 
                outputs = self.PubBCRPredictor(torch.tensor(feature, dtype=torch.float32))
                return torch.sigmoid(outputs)
            elif self.model=='cdrl' or self.model=='cdrl3':
                outputs = self.PubBCRPredictor(torch.tensor(feature, dtype=torch.float32))
                return outputs
    
    def plot_metric(self,input_labels,input_probs):
        if self.model=='cdrh' or self.model=='cdrh3': 
            threshold = 0.5
            input_predictions = (input_probs >= threshold).astype(int)
            train_accuracy = accuracy_score(input_labels, input_predictions)
            precision = precision_score(input_labels, input_predictions)
            recall = recall_score(input_labels, input_predictions)
            f1 = f1_score(input_labels, input_predictions)
            
            print(f'train Accuracy: {100 * train_accuracy:.2f}%')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')

            # 计算并打印混淆矩阵（数量）
            cm = confusion_matrix(input_labels, input_predictions)

            # 计算混淆矩阵（比率）
            cm_normalized = confusion_matrix(input_labels, input_predictions, normalize='true')

            print("Confusion Matrix (Counts):")
            print(cm)

            print("\nConfusion Matrix (Normalized by true condition):")
            print(cm_normalized)

            # 使用 seaborn 热图绘制归一化的混淆矩阵
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'],
                        annot_kws={"size": 20})
            plt.title('Normalized Confusion Matrix')
            plt.show()


            # 绘制 ROC 曲线
            fpr, tpr, _ = roc_curve(input_labels, input_probs)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
        elif self.model=='cdrl' or self.model=='cdrl3':
            mse = mean_squared_error(input_labels, input_probs)
            # rmse = mean_squared_error(input_labels, input_probs, squared=False)  # 设置squared=False以获取RMSE
            mae = mean_absolute_error(input_labels, input_probs)
            mape = mean_absolute_percentage_error(input_labels, input_probs) * 100  # MAPE通常表示为百分比
            r2 = r2_score(input_labels, input_probs)

            print(f'Mean Squared Error (MSE): {mse:.4f}')
            # print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
            print(f'Mean Absolute Error (MAE): {mae:.4f}')
            print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
            print(f'R-Square (R²): {r2:.4f}')

            # 计算 Spearman 相关系数
            spearman_corr, _ = spearmanr(input_labels, input_probs)
            print(f'Spearman Correlation: {spearman_corr:.4f}')

            # 绘制散点图及回归线，并在图上标注Spearman相关系数和其他指标
            plt.figure(figsize=(8, 6))
            fontsize=20

            scatter = sns.scatterplot(x=input_labels, y=input_probs, alpha=0.6)
            regline = sns.regplot(x=input_labels, y=input_probs, scatter=False, color='red', line_kws={"lw": 2})
            
            # 在图上添加Spearman相关系数及其他指标的文本注释
            textstr = '\n'.join((
                f'Spearman r: {spearman_corr:.4f}',
                # f'MSE: {mse:.4f}',
                # f'RMSE: {rmse:.4f}',
                # f'MAE: {mae:.4f}',
                # f'MAPE: {mape:.2f}%',
                # f'R²: {r2:.4f}'
            ))
            props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.5)
            plt.text(0.05, 0.95, textstr,
                    transform=plt.gca().transAxes,
                    fontsize=fontsize-8,
                    verticalalignment='top',
                    bbox=props)
            plt.xticks(fontsize=fontsize-6)
            plt.yticks(fontsize=fontsize-6)
            plt.title('Scatter Plot with Regression Line',fontsize=fontsize)
            plt.xlabel('True Values',fontsize=fontsize-2)
            plt.ylabel('Predicted Values',fontsize=fontsize-2)
            plt.show()