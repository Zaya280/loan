import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Өгөгдлийг унших
df = pd.read_excel("Shiinechilsenbagts.xlsx")
print(df.head())
if 'Орлого' not in df.columns:
    df['Орлого'] = 0

if 'Зээл хэмжээ' not in df.columns:
    df['Зээл хэмжээ'] = 0
    
# Өгөгдлийг цэвэрлэх
df["Зээл авсан огноо"] = pd.to_datetime(df["Зээл авсан огноо"])
df["Төлөх огноо"] = pd.to_datetime(df["Төлөх огноо"])
df["Огнооны ялгаа (хоног)"] = (df["Төлөх огноо"] - df["Зээл авсан огноо"]).dt.days

df["Зээл төлөлтийн ангилал"] = df["Зээлийн төлөв"].apply(lambda x: 1 if x == 1 else 0)
df.drop(columns=["Зээлийн төлөв"], inplace=True)



# Хэрэггүй багануудыг устгах
drop_cols = [
    "Зээл авсан огноо", "Төлөх огноо", "Зээлийн үлдэгдэл", "Идэвхитэй хоног",
    "Торгуулийн хүү %", "Зээлийн хугацаа", "Төлсөн шимтгэл",
    "Хэтэрсэн хоног", "Зээл төлөлтийн хувь", "Огнооны ялгаа (хоног)"
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Зөвхөн 5 талбарыг ашиглах болно:
# Хүйс (gender), Нас (age), Орлого (income), Кредит оноо (credit_score), Зээл хэмжээ (loan_amount)
df["Хүйс"] = df["Хүйс"].astype(int)
X = df[['Хүйс', 'Нас', 'Орлого', 'Зээл хэмжээ']]
y = df["Зээл төлөлтийн ангилал"]

# Train-test dataset хуваах
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest-ийн hyperparameter tuning хийх
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=20, cv=3, verbose=2, n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Train болон Test дээр таамаглал хийх
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Үнэлгээний хэмжүүрүүдийг тооцоолох
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Train-set үнэн зөв байдал: {train_accuracy:.2f}")
print(f"Test-set үнэн зөв байдал: {test_accuracy:.2f}")
print("\nTrain-set Classification Report:")
print(classification_report(y_train, y_train_pred))
print("\nTest-set Classification Report:")
print(classification_report(y_test, y_test_pred))

# Feature Importance дүрслэл
feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis", legend=False)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

# Confusion Matrix дүрслэл функц
def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_confusion_matrix(y_train, y_train_pred, "Confusion Matrix - Train Set", "static/confusion_matrix_train.png")
plot_confusion_matrix(y_test, y_test_pred, "Confusion Matrix - Test Set", "static/confusion_matrix_test.png")

# Загварыг pickle ашиглан хадгалах
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
