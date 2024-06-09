import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import CSVUploadForm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .models import CSVUpload

def upload_file(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Process the CSV file
            df = pd.read_csv(request.FILES['file'])
            
            # Display basic information
            info = df.info()
            description = df.describe()
            
            # Plot the distribution of the target variable
            plt.figure(figsize=(6,4))
            sns.countplot(x='Class', data=df)
            plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
            distribution_chart = 'distribution.png'
            plt.savefig(f'detection/static/{distribution_chart}')
            plt.clf()

            # Plot a correlation matrix
            plt.figure(figsize=(12,10))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
            plt.title('Correlation Matrix')
            correlation_chart = 'correlation.png'
            plt.savefig(f'detection/static/{correlation_chart}')
            plt.clf()

            # Splitting the data into features and target
            X = df.drop('Class', axis=1)
            y = df['Class']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Standardizing the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Initialize the Logistic Regression model
            model = LogisticRegression()

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluation
            confusion = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)

            # Identify fraudulent transactions
            X_test_original = scaler.inverse_transform(X_test)
            df_test = pd.DataFrame(X_test_original, columns=X.columns)
            df_test['ID'] = df_test.index
            df_test['Amount'] = df_test['Amount']  # Assuming 'Amount' is one of the features
            df_test['Actual'] = y_test.values
            df_test['Predicted'] = y_pred
            fraudulent_transactions = df_test[df_test['Predicted'] == 1][['ID', 'Amount', 'Actual', 'Predicted']]

            # Plot the confusion matrix
            plt.figure(figsize=(6,4))
            sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            confusion_matrix_chart = 'confusion_matrix.png'
            plt.savefig(f'detection/static/{confusion_matrix_chart}')
            plt.clf()

            context = {
                'info': info,
                'description': description,
                'distribution_chart': distribution_chart,
                'correlation_chart': correlation_chart,
                'confusion_matrix_chart': confusion_matrix_chart,
                'report': report,
                'accuracy': accuracy,
                'fraudulent_transactions': fraudulent_transactions.to_html(classes='table table-striped', index=False)
            }
            return render(request, 'detection/results.html', context)
    else:
        form = CSVUploadForm()
    return render(request, 'detection/upload.html', {'form': form})