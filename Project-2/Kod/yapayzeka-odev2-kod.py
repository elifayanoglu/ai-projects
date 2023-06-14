# -*- coding: utf-8 -*-
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score,confusion_matrix, accuracy_score, roc_auc_score, recall_score,precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_ind
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
nltk.download('stopwords')
stopWords = set(stopwords.words('turkish'))

vectorization_type = 1       # 0 iken bow 1 ken tfidf
classification_model = 0    # 0 =naivebayes, 1= svm , 2= logistik regresyon , 3= knn(k=5) , 4= randomforest

pca_flag = False
norm_flag = False
kikare_flag = True

dosya_adi = "deneme"  #"Kİ-10"
k_parametresi = 800

# Verisetini okuduğumuz yer
# path = "C:\\Users\\zeynep.colak\\Desktop\\"
path =""
df = pd.read_csv(path + "veri_3000.csv")
sonuc_dosyasi = open(dosya_adi+"\\" +"Sonuçlar.txt", "w")

# Verisetinden duplicateleri siliyoruz
print("head:\n",df.head())
print("\nShape (satırxkolon):",df.shape)
df = df.drop_duplicates()
print("\nDublicateleri attınca:",df.shape)
print(df.head())

#foldların ortalamalarını alıp bastırdığımız fonksiyon
def print_ort():
    sonuc = '\nTüm Foldların Ortalama Accuracy Skoru :  ' + str(round(np.mean(scores), 2))
    print(sonuc)
    sonuc_dosyasi.write(sonuc)
    sonuc = '\nTüm Foldların Ortalama F1 Skoru :        ' + str(round(np.mean(scoresf1), 2))
    print(sonuc)
    sonuc_dosyasi.write(sonuc)
    sonuc = '\nTüm Foldların Ortalama Recall Skoru :    ' + str(round(np.mean(scores_recall), 2))
    print(sonuc)
    sonuc_dosyasi.write(sonuc)
    sonuc = '\nTüm Foldların Ortalama ROC-AUC Skoru :   ' + str(round(np.mean(scores_roc), 2))
    print(sonuc)
    sonuc_dosyasi.write(sonuc)
    sonuc = '\nTüm Foldların Ortalama Precision Skoru : ' + str(round(np.mean(scores_pres), 2))
    print(sonuc)
    sonuc_dosyasi.write(sonuc)

    bilgi = "\n\nTüm Foldların Accuracy Skorları: "
    print(bilgi)
    sonuc_dosyasi.write(bilgi+"".join(str(scores)))
    bilgi = "\nTüm Foldların F1 Skorları: "
    print(bilgi)
    sonuc_dosyasi.write(bilgi + "".join(str(scoresf1)))
    bilgi = "\nTüm Foldların Recall Skorları: "
    print(bilgi)
    sonuc_dosyasi.write(bilgi + "".join(str(scores_recall)))
    bilgi = "\nTüm Foldların ROC-AUC Skorları: "
    print(bilgi)
    sonuc_dosyasi.write(bilgi + "".join(str(scores_roc)))
    bilgi = "\nTüm Foldların Precision Skorları: "
    print(bilgi)
    sonuc_dosyasi.write(bilgi + "".join(str(scores_pres)))

#noktalama işaretleri, stopwordler atılır, kelimeler küçük harfe çevrilir
def veri_onisleme(veri_text):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in veri_text:
        if ele in punc:
            veri_text = veri_text.replace(ele, "")
    split_words = []
    for w in veri_text.split():
        w = w.lower()
        if w not in stopWords:
            split_words.append(w)
    return split_words

#sırayla çalıştırdığımız tahminleyici algoritmalar
classlar = ["GradientBoosting", "SVM" , "LogisticRegression", "KNN", "RandomForest"]
# for döngüsünde indislerin anlamları: 0 =naivebayes, 1= svm , 2= logistik regresyon , 3= knn(k=5) , 4= randomforest
for i in range (5):
    classification_model = i
    if vectorization_type==0:
        print("BOW")
        comments_vectors = CountVectorizer(analyzer=veri_onisleme).fit_transform(df['text'])
    else:
        print("TF-IDF")
        comments_vectors = TfidfVectorizer(analyzer=veri_onisleme).fit_transform(df['text'])

    if norm_flag == True:
        # normalizasyon
        scaler = MaxAbsScaler()
        # scaler =MinMaxScaler()
        comments_vectors = scaler.fit_transform(comments_vectors)

    if pca_flag==True:
       # PCA uygula
      pca = PCA(n_components=40)
      comments_vectors_pca = pca.fit_transform(comments_vectors.toarray())
      comments_vectors= comments_vectors_pca


    if kikare_flag ==True:
      # #Chi-Squared ile Feature Selection
      X_new = SelectKBest(chi2, k=k_parametresi).fit_transform(comments_vectors, df['pos'])
      comments_vectors= X_new


    # classifier seçimi yükle
    if classification_model==0:
        print("\nGradientBoosting")
        sonuc_dosyasi.write("\nGradientBoosting")
        classifier = GradientBoostingClassifier(n_estimators=300,
                                   learning_rate=0.05,
                                   random_state=100,
                                   max_features=10)
    elif classification_model==1:
        print("\nSVC")
        sonuc_dosyasi.write("\nSVC")
        classifier = SVC()
    elif classification_model==2:
        print("\nLogisticRegression")
        sonuc_dosyasi.write("\nLogisticRegression")
        classifier = LogisticRegression()
    elif classification_model==3:
        print("\nKnn5")
        sonuc_dosyasi.write("\nKnn5")
        classifier =KNeighborsClassifier(n_neighbors=5)
    else: #classification_model==4:
        print("\nRandomForest")
        sonuc_dosyasi.write("\nRandomForest")
        classifier =RandomForestClassifier(n_estimators=200, random_state=0)

    """# ** K fold başlangıç**"""
    # k=10 için kFold tanımlanır
    folds = KFold(n_splits = 10, shuffle = True, random_state = None)
    X, y = comments_vectors, df.pos
    y = np.array(y)

    # 10 foldun değerlerinin toplanacağı listeler
    scores = []
    scoresf1 = []
    scores_recall =[]
    scores_pres =[]
    scores_roc=[]
    TPs= []
    FPs= []
    FNs= []
    TNs= []
    # k=10 iken Kfold döngüsü başlatılır
    for n_fold, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('\n Fold '+ str(n_fold+1 ) )
              # ' \n\n train ids :' +  str(train_index) +
              # ' \n\n validation ids :' +  str(valid_index))

        #kfoldun ayırdığı indislere göre train ve test ayrılır
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        #Eğitim yapılır ve tahminleyici test edilir
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_valid)

        #Her foldun sonucu kayıt edilir. İlerde ortalaması alınacaktır
        acc_score = accuracy_score(y_valid, y_pred)
        f1score = f1_score(y_valid, y_pred)
        scorerecall= recall_score(y_valid, y_pred)
        scorepres= precision_score(y_valid, y_pred)
        scoreroc= roc_auc_score(y_valid, y_pred)
        scores.append(acc_score)
        scoresf1.append(f1score)
        scores_recall.append(scorerecall)
        scores_pres.append(scorepres)
        scores_roc.append(scoreroc)
        print('\n Accuracy score for Fold ' +str(n_fold+1) + ' --> ' + str(acc_score)+' ')
        print('\n F1 score for Fold ' +str(n_fold+1) + ' --> ' + str(f1score)+' ')
        print('\n Recall score for Fold ' +str(n_fold+1) + ' --> ' + str( scorerecall)+' ')
        print('\n ROC score for Fold ' +str(n_fold+1) + ' --> ' + str(scoreroc)+' ')
        print('\n Precision score for Fold ' +str(n_fold+1) + ' --> ' + str(scorepres)+'\n')

        #Conf matrisin düzenlemeleri
        con =confusion_matrix(y_valid, y_pred)
        TPs.append(con[0][0])
        FPs.append(con[0][1])
        FNs.append(con[1][0])
        TNs.append(con[1][1])
        print("\n Confusion Matrix : ")
        print(con)

    # Başarı yüzdeleri için grafik hazırlandı ve dosyaya kayıt edildi
    list = [0]
    fig = plt.figure()
    plt.plot(list +scores,label='Accuracy')
    plt.plot(list+scores_roc,label='ROC_AUC')
    plt.plot(list+scores_recall,label='Recall')
    plt.plot(list+scores_pres,label='Precision')
    plt.plot(list+scoresf1,label='F1-Score')
    plt.title("10 Fold İçin Başarı Oranları ")
    plt.xlabel("Foldlar")
    plt.ylabel(" Başarı Yüzdeleri")
    plt.axis([1,10,0.7,0.99]) #[xmin, xmax, ymin, ymax]
    plt.legend()
    fig.savefig(path+dosya_adi+"\\"+classlar[i]+'_metrikler.jpg')
    #plt.show()

    # Foldların ortalama başarıları farklı metriklerle histogram şeklinde sunuldu
    veri_sutun = ['Accuracy', 'ROC_AUC', 'Recall','Precision', 'F1-Score']
    ort_deger = [round(np.mean(scores),2), round(np.mean(scores_roc),2), round(np.mean(scores_recall),2), round(np.mean(scores_pres),2), round(np.mean(scoresf1),2)]
    fig2 = plt.figure()
    plt.bar(veri_sutun,ort_deger )
    plt.ylabel("Ortalama Başarı Yüzdeleri")
    plt.title(" 10 Fold İçin Başarı Oranları ")
    #plt.show()
    fig2.savefig(path+dosya_adi+"\\"+classlar[i] + '_histogram.jpg')

    #ortalamalar ekrana bastırıldı
    print_ort()

    # conf matris tablo olarak oluşturuldu ve klasöre kaydedildi
    conf_mat  = np.empty((2, 2), float)
    conf_mat[0][0]=np.mean(TPs)
    conf_mat[0][1]=np.mean(FPs)
    conf_mat[1][0]=np.mean(FNs)
    conf_mat[1][1]=np.mean(TNs)
    fig3 = plt.figure()
    table = sns.heatmap(conf_mat/np.sum(conf_mat, axis=1)[:, np.newaxis],
                        annot=True, fmt='.2f', cmap='Blues')
    print(conf_mat )
    table.set_xlabel('\nPredicted Values')
    table.set_ylabel('Actual Values')
    table.set_title(" Confusion Matrix")
    fig3.add_axes(table)
    fig3.savefig(path+dosya_adi+"\\"+classlar[i] + '_conf.jpg')
    #plt.show()

    if i ==0:
        accuracy_gb =  scores      #gradient
    elif i == 1:
        accuracy_svm =  scores      #SVM
    elif i == 2:
        accuracy_lr = scores    #LOGİSTİCREGRESSİON
    elif i == 3:
        accuracy_knn = scores      #KNN
    elif i == 4:
        accuracy_rf = scores      #RANDOMFOREST

print("***accuracyler*********,")
print(accuracy_gb, accuracy_svm, accuracy_lr, accuracy_rf,accuracy_knn)
# t-test uygula
ttest1, pval1 = ttest_ind(accuracy_gb, accuracy_svm)
ttest2, pval2 = ttest_ind(accuracy_gb, accuracy_lr)
ttest3, pval3 = ttest_ind(accuracy_gb, accuracy_knn)
ttest4, pval4 = ttest_ind(accuracy_gb, accuracy_rf)
ttest5, pval5 = ttest_ind(accuracy_svm, accuracy_lr)
ttest6, pval6 = ttest_ind(accuracy_svm, accuracy_knn)
ttest7, pval7 = ttest_ind(accuracy_svm, accuracy_rf)
ttest8, pval8 = ttest_ind(accuracy_lr, accuracy_knn)
ttest9, pval9 = ttest_ind(accuracy_lr, accuracy_rf)
ttest10, pval10 = ttest_ind(accuracy_knn, accuracy_rf)

my_list = [pval1, pval2, pval3, pval4, pval5, pval6, pval7, pval8, pval9, pval10]

# t- test sonuçlarını yazdır
sonuc_dosyasi.write("\n\nT-test Sonuçları:")
for idx, pval in enumerate(my_list):
    if idx == 0:
        yazi = "GB - SVM:"
        print(yazi)
    elif idx == 1:
        yazi = "GB - LR:"
        print(yazi)
    elif idx == 2:
        # print("GB - KNN:")
        yazi = "GB - KNN:"
        print(yazi)
    elif idx == 3:
        # print("GB - RF:")
        yazi = "GB - RF:"
        print(yazi)
    elif idx == 4:
        # print("SVM - LR:")
        yazi = "SVM - LR:"
        print(yazi)
    elif idx == 5:
        # print("SVM - KNN:")
        yazi = "SVM - KNN:"
        print(yazi)
    elif idx == 6:
        # print("SVM - RF:")
        yazi = "SVM - RF:"
        print(yazi)
    elif idx == 7:
        # print("LR - KNN:")
        yazi = "LR - KNN:"
        print(yazi)
    elif idx == 8:
        # print("LR - RF:")
        yazi = "LR - RF:"
        print(yazi)
    elif idx == 9:
        # print("KNN - RF:")
        yazi = "KNN - RF:"
        print(yazi)

    sonuc_dosyasi.write("\n"+yazi)
    if pval < 0.05:
        print("İstatistiksel olarak anlamlı bir fark var.")
        sonuc_dosyasi.write("\nİstatistiksel olarak anlamlı bir fark var.")
    else:
        print("İstatistiksel olarak anlamlı bir fark yok.")
        sonuc_dosyasi.write("\nİstatistiksel olarak anlamlı bir fark yok.")

sonuc_dosyasi.close()