import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

#başarı metriklerini bastırmak için yazılan fonksiyon
def print_metrics(Y_true, Y_predicted):
    print("Confusion Matrix : \n")
    conf_mat = confusion_matrix(Y_true, Y_predicted)
    print(conf_mat)
    print("Accuracy Score :" + str(round(accuracy_score(Y_true, Y_predicted ), 2)))
    print("Precision Score :" + str(round(precision_score(Y_true, Y_predicted, average='macro'), 2)))
    print("Recall Score :" + str(round(recall_score(Y_true, Y_predicted, average='macro'), 2)))
    print("F1_score :" + str(round(f1_score(Y_true, Y_predicted, average='macro'), 2)))
    print("\nClassification report:")
    print(classification_report(Y_true, Y_predicted))

    # görselleştirme
    table = sns.heatmap(conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis],
                        annot=True, fmt='.2f', cmap='Blues')
    table.set_xlabel('\nPredicted Values')
    table.set_ylabel('Actual Values')
    plt.show()
    return accuracy_score(Y_true, Y_predicted )

#başarı artırmak için denenen pca dönüşümü ve özellik seçme bayrakları
kselect = False
pca_flag = True

#veri setinin okunması ve gereksiz kolonun atılması
df = pd.read_csv("islenmis_csv.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)
# df2 = df[221:224]
# df.drop([221,222,223], axis=0, inplace=True)
kolonlar = df[df.columns[:-1]]

if pca_flag:
      pca = PCA(n_components=3)
      kolonlar = pca.fit_transform(df[df.columns[:-1]])

#verisetinin eğitim ve test olarak bölünmesi
if kselect:
    X_new = SelectKBest(chi2, k=7).fit_transform(kolonlar, df['karakter'])
    X_train, X_test, y_train, y_test = train_test_split(X_new, df['karakter'], test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(kolonlar, df['karakter'], test_size=0.2, random_state=42)

# algoritmaların tanıtımı ve fit fonksiyonuna sokulması
# X_train train kümesinin karakter dışındakii kolonları, Y_train karakter kolonu : tahmin edilecek laebl
lr = make_pipeline(StandardScaler(), LogisticRegression( )).fit(X_train, y_train)
svm = make_pipeline(MinMaxScaler(), SVC(max_iter=15)).fit(X_train, y_train)
gnb = make_pipeline(MinMaxScaler(), GaussianNB()).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

#modelleri listeledik bir döngüde çalıştırabilmek için skorları sakladık ve tabloladık
models = [lr, svm, rf, gnb, knn]
scores = []
for model in models:
    y_pred = model.predict(X_test)
    scores.append(print_metrics(y_test, y_pred))

#görselleştirme için gerekli işlemler
models = ["LogisticReg. "+ str(round(scores[0],2)) , "SVM :"+ str(round(scores[1],2)), "GaussianNB :"+ str(round(scores[2],2)) , "RandomForest :"+ str(round(scores[3],2)),
          "KNeighbors :"+ str(round(scores[4],2))]
fig = plt.figure(figsize=(10, 5))
plt.bar(models, scores, color='green', width=0.4)
plt.xlabel("Model :Accuracy Skoru")
plt.ylabel("Skor")
plt.title("Modellerin Sonuçları")
plt.show()

#####bu testler için df bölünmüştü ancak genel eğitimler tüm verisetiyle yapıldı
# print("Gerçek labeller: [4,3,3]")
# predler =models[0].predict(df2[df2.columns[:-1]])
# print("LogisicReg Tahmini:", predler)
# predler =models[1].predict(df2[df2.columns[:-1]])
# print("SVM Tahmini:",predler)
# predler =models[2].predict(df2[df2.columns[:-1]])
# print("Random Forest Tahmini:",predler)
# predler =models[3].predict(df2[df2.columns[:-1]])
# print("Gaussian NB Tahmini:",predler)
# predler =models[4].predict(df2[df2.columns[:-1]])
# print("KNN Tahmini:",predler)



