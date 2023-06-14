import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("kisilik_testi.csv")
df.drop("Zaman", axis=1, inplace=True)
print(df)

le = LabelEncoder()
df.cinsiyet = le.fit_transform(df.cinsiyet)
df.burc_grup = le.fit_transform(df.burc_grup)
df.karakter = le.fit_transform(df.karakter)
df.renk = le.fit_transform(df.renk)
df.kahve = le.fit_transform(df.kahve)
df.yas = le.fit_transform(df.yas)

# karaktere göre burç grupları üzerindeki dağılımları tablo haline getirme
karakterler = [ 'ciddi','dışadönük', 'enerjik' ,'güvenilir', 'sabırlı','sorumlu']
gruplar = ['ateş', 'hava', 'su', 'toprak']

frekanslar = {}
#her indis bir karakter, sahip olunan karakterler için hangi burçda olduğu sayılıyor
for i in np.unique(df['karakter']):
    col = df[df['karakter'] == i]['burc_grup']
    freq = [col.value_counts()[i] for i in np.unique(col)]
    frekanslar[karakterler[i]] = freq

# bu sayılar tabulate için gerekli hale getiriliyor
data = [[list(frekanslar)[0], frekanslar[(list(frekanslar)[0])]],
        [list(frekanslar)[1], frekanslar[(list(frekanslar)[1])]],
        [list(frekanslar)[2], frekanslar[(list(frekanslar)[2])]],
        [list(frekanslar)[3], frekanslar[(list(frekanslar)[3])]],
        [list(frekanslar)[4], frekanslar[(list(frekanslar)[4])]],
        [list(frekanslar)[5], frekanslar[(list(frekanslar)[5])]]]
# tabulate fonksiyonuyla ekrana bastırlıyor
print(tabulate(data, headers=['karakter', '[ateş,hava,su,toprak]'], tablefmt="fancy_grid"))

#burçların frekanslarını alıyoruz, pasta grafiği için
burc=[0,0,0,0] # 0:ateş 1:hava  2: su 3:toprak
for i in range(6):
    burc[0] += data[i][1][0]
    burc[1] += data[i][1][1]
    burc[2] += data[i][1][2]
    burc[3] += data[i][1][3]

# bu sayılar tabulate için gerekli hale getiriliyor
data = [[list(frekanslar)[0], gruplar[frekanslar[list(frekanslar)[0]].index(max(frekanslar[(list(frekanslar)[0])]))]],
        [list(frekanslar)[1], gruplar[frekanslar[list(frekanslar)[1]].index(max(frekanslar[(list(frekanslar)[1])]))]],
        [list(frekanslar)[2], gruplar[frekanslar[list(frekanslar)[2]].index(max(frekanslar[(list(frekanslar)[2])]))]],
        [list(frekanslar)[3], gruplar[frekanslar[list(frekanslar)[3]].index(max(frekanslar[(list(frekanslar)[3])]))]],
        [list(frekanslar)[4], gruplar[frekanslar[list(frekanslar)[4]].index(max(frekanslar[(list(frekanslar)[4])]))]],
        [list(frekanslar)[5], gruplar[frekanslar[list(frekanslar)[5]].index(max(frekanslar[(list(frekanslar)[5])]))]]]
# tabulate fonksiyonuyla ekrana bastırlıyor
print(tabulate(data, headers=['karakter', 'Frekansı en yüksek burç grubu'], tablefmt="fancy_grid"))

icerik = ['ates', 'hava', 'su', 'toprak']
renkler = ['#F1493E', '#B0E5E8', '#447FCB', '#6B4C4C']
plt.pie(burc, labels=icerik, colors=renkler,  startangle=90, shadow=True, explode=(0, 0, 0, 0), autopct='%1.1f%%')
plt.title('Burç Gruplarının Dağılımı')
plt.show()

# şekerli:0 şekersiz:1 frekansı hesaplama ve pasta grafiğinde bastırma
sekersiz = df.kahve.sum()
sekerli = len(df) - sekersiz
icerik = ['Şekerli kahve içenler', 'Şekersiz kahve içenler']
renkler = ['c', 'm']
plt.pie([sekerli, sekersiz],labels=icerik, colors=renkler, startangle=90, shadow=True, explode=(0.1, 0), autopct='%1.1f%%')
plt.title('Kahvede Şeker Kullanımı')
plt.show()

# renklere göre pasta grafiği görme
renk=[0,0,0]  #0: nötr  1:soğuk 2:sıcak
for i in range(len(df)):
    renk[df['renk'][i]] += 1
icerik = ['Nötr Renkler', 'Soğuk Renkler', 'Sıcak Renkler']
renkler = ['#857F7F', '#15D1E7', '#E74215']
plt.pie(renk, labels=icerik, colors=renkler, startangle=90, shadow=True, explode=(0, 0, 0), autopct='%1.1f%%')
plt.title('Sevilen Renklerin Dağılımı')
plt.show()

# Kişinin kendi işaretlediği karakterine göre diğer özelliklerinin görüleebileceği bir grafik
acik_soz= []
sorumlu= []
durust= []
stres= []
disa= []
duzen= []
sabir = []
for i in range(6):
    acik_soz.append(df[df['karakter'] == i]['acik_sozluluk'].mean())
    sorumlu.append(df[df['karakter'] == i]['sorumluluk'].mean())
    durust.append(df[df['karakter'] == i]['durust'].mean())
    stres.append(df[df['karakter'] == i]['stres'].mean())
    disa.append(df[df['karakter'] == i]['disa_donuk'].mean())
    duzen.append(df[df['karakter'] == i]['duzenli'].mean())
    sabir.append(df[df['karakter'] == i]['sabirli'].mean())
data = {'açık sözlülük': acik_soz,
         'sorumluluk': sorumlu,
         'dürüstlük': durust,
         'stres':stres,
         'dişa dönüklülük': disa,
         'düzenlilik': duzen,
          'sabırlılık' : sabir,
         'karakterler': karakterler}
data = pd.DataFrame(data)
data.plot(x="karakterler", y=data.columns[:-1].to_list(), kind="bar", figsize=(10, 5))
plt.show()

# cinsiyete göre katılım oranları hesaplama
kadin =[0,0,0,0]
erkek =[0,0,0,0]
for i in range(len(df)):
    if df['cinsiyet'][i] == 1: #kadın yaş
        kadin[df['yas'][i]] += 1
    else:
        erkek[df['yas'][i]] += 1
# grafikte katılım oranlarını gösterme
plt.bar(["18 altı" , "18 - 30", "30 - 50", "50 üstü"],kadin, label="Kadınlar",width=.5)
plt.bar(["18 altı" , "18 - 30", "30 - 50", "50 üstü"],erkek, label="Erkekler",width=.5)
plt.legend()
plt.xlabel('Yaşa Göre Dağılım')
plt.ylabel('Frekanslar')
plt.title('Yaş Aralıkları')
plt.show()

# df.to_csv('islenmis_csv.csv', sep=',')