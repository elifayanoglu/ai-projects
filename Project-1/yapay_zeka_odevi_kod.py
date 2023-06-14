# -*- coding: utf-8 -*-

# sinif1 label 1 = ulasim , sinif 2 label0 finans
import pandas as pd
import random
import sys
import numpy as np 

dataset_isim = "dataset3_yeni.csv"

# birey_boyut bir bireyin kaç kelimeden oluşacağını gösteriyor
POPULATION_SIZE = 50
birey_boyut = 50
sinif_boyut = (birey_boyut)/2
mutasyon_yuzde = 20         #Başlangıç mutasyonn değeri


#poppülasyon listesi içinde POPULATION_SIZE kadar birey olacak / fitnesses ise o bireylerin skoru
pop = []
fitnesses = []
 
eniyi_kelimeler=[]
eniyi_fitness_genel = 0
 

#Tüm kelimeleri tuttuğumuz Sözlüğün sınıf yapısı... sınıf1 1.sınıfa ait tüm kelimeleri içerecek vb.
class Dictionary_struct:
    sınıf1 = []
    sınıf2 = []

#RASTGELE BİREY OLUŞTURAN FONKSİYON - ilk başta bireyler rastgele oluşturuldu
def get_birey():
    birey = []
    for i in range(int(sinif_boyut)):
        kelime = random.choice(Dictionary.sınıf1)
        birey.append(kelime)
    for i in range(int(sinif_boyut),int(sinif_boyut*2)):
        kelime = random.choice(Dictionary.sınıf2)
        birey.append(kelime)   
    return birey

    # bir örnek cümlenin doğru tahmin edilip edilmediğini bulan fonksiyon.
def get_fitness(cnt1, cnt2 , cat):
    #cümleyi ne olarak bulduğunu bulmak için count değerlerine bakılır
    if (cnt1>cnt2):
        label =1 #ulasim
    elif (cnt1<cnt2):
        label =0
    elif (cnt1 ==0 & cnt2 ==0):
        return 0
    else:
        label = random.randint(0, 1)   #eğer iki sınıftan da aynı sayıda kelime bulunmuşsa rastgele atanacak.      
    # bulduğu label gerçek etiket mi bakılır
    if label==1 & ("ulasim" == cat):
            return 1
    elif label == 0 & ("finans" == cat):
            return 1
    else:
        return 0
    
def Average(lst):
    return sum(lst) / len(lst)
        
# bir bireyin 100 örneğin kaçını doğru bildiğini bulan fonksiyon. bu değer fitnes değeri olarak döndürülür
def degerlendirme(bry):
    fitness = 0
    for i in range(100): 
        cumle= df["text"][i].split() 
        cnt_sinif1=0
        cnt_sinif2=0 
        for j in range(len(cumle)):    #bir cümlenin tüm kelimelerine bakıyoruz
            for k in range(len(bry)):   
               #cümlenin bir kelimesini bireydeki tüm kelimelerle karşılaştırıyorum
                if((cumle[j] == bry[k]) & (k<sinif_boyut)):
                    #label = 1  ulasım kategorisi
                    cnt_sinif1 = cnt_sinif1 + 1 
                elif((cumle[j] == bry[k]) & (k>=sinif_boyut)):
                    #label=0 finans kategorisi 
                    cnt_sinif2 = cnt_sinif2 + 1 
        fitness =  fitness + get_fitness(cnt_sinif1, cnt_sinif2 , df["cat"][i] ) #fitness değerleri toplanıyor
    return fitness

#parent seçerken randoma verilen fitnes değerleri düzenlendi - Sıralama Seçimi tekniği kullanılmak istendi
def siralama_fonk(fitnes_list): 
    sira =fitnes_list
    sira = np.unique(sira)
    sira2 = list(sira)
    sira2.sort()
    n= len(sira2) 
    fitneslar=[]
    for i in range(len(fitnes_list)):
        index = sira2.index(fitnes_list[i]) 
        yeni =(index+1)/(n*(n+1)/2)   # fitnesın kaçıncı sırada olduğu / (tüm sıralar toplamı)  örn: 4/(1+2+3+4)
        fitneslar.append(yeni)  
    return fitneslar

# bireylerin mutasyonu için yazılan fonksiyon: child bireyinin %yuzdesi dictionaryden rastgele seçilir
def mutasyon(child, yuzde):
    for i in range(int((yuzde/100) * birey_boyut) ):
        indis = random.randint(0, birey_boyut)
        if indis <  birey_boyut/2:
            child[i] = random.choice(Dictionary.sınıf1)
        else:
            child[i] = random.choice(Dictionary.sınıf2)
    return child
        
# parent1 ve parent2 'den child oluşturulurken Uniform Crossover tekniği kullanıldı. child1 işleme alındı
def mate(par1, par2):
      child_gen = []
      temp = []             #Rastgele sayılardan oluşan Template listesi oluşşturuldu
      for i in range(birey_boyut):
          gen = random.randint(0, 1)
          temp.append(gen)
          if gen ==0:
              child_gen.append(par2[i])   # 0 olan indislerin yerine parent2 indisi getirildi
          else:
              child_gen.append(par1[i])     # 1 olan indislerin yerine parent1 indisi getirildi 
      return child_gen
     

# yeni populasyon oluşturan fonksiyon
def yeni_pop_olustur(pop,fit, mutasyon_yuzde,nesil , eniyi_fitness):
    #fitnessların olasılık haline çevirilmesi
    fit = siralama_fonk(fit)
    fit = np.array(fit)
    fit /= fit.sum()
    
    new_pop = []
    x= POPULATION_SIZE
    
    for i in range(x): 
        parent1_indis =  np.random.choice(len(pop), 1, p=fit)   #fitnesslarına göre random seçilen parentlar
        parent1 = pop[parent1_indis[0]]                         #ve parent olarak atanması
        
        parent2_indis =  np.random.choice(len(pop), 1, p=fit) 
        parent2 = pop[parent2_indis[0]]
        
        child = mate(parent1, parent2)      #uniform crossover yapan fonksiyon
        
        child = mutasyon(child, mutasyon_yuzde)    # child bireyi mutasyona uğratan fonksiyon
        new_pop.append(child)                       #oluşan child popülasyona eklenir
        
    #yeni oluşan popülasyonun fitnesları hesaplanır
    fitnes= []
    eniyi_birey = []
    for i in range(POPULATION_SIZE):
        fitn = degerlendirme(new_pop[i])
        fitnes.append(fitn)
        if eniyi_fitness <= fitn:
            eniyi_fitness = fitn
            eniyi_birey = new_pop[i]
            
        
    mutasyon_yuzde = mutasyon_yuzde - (3/nesil)  # mutasyonun nesil arttıkça azaltılması. örnek azaltma formül: (20 - 3/nesil_sayisi)
    return new_pop, fitnes , mutasyon_yuzde , eniyi_birey , eniyi_fitness

path = "C:\\Users\\zeynep.colak\\Desktop\\"
df = pd.read_csv(path + dataset_isim)

global Dictionary
Dictionary = Dictionary_struct() 
       
#DİCTİONARYNİN İÇİNİ DOLDURMA
for i in range(100):
    veri_text = df["text"][i]
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789+'''
    for ele in veri_text:
        if ele == "." or ele == "," or ele == "-":
            veri_text = veri_text.replace(ele, " ")
        elif ele in punc:
            veri_text = veri_text.replace(ele, "")
    
    df["text"][i] = veri_text   # verisetinin heopsinin içinden noktalama işaretleri ve sayıları çıkardık
    
    if df["cat"][i] == "ulasim":    #cümlenin labeli bulundu
        label = 1
    else:
        label = 0
                
    for w in veri_text.split():      # cümlenin labeline göre sözlüğün ilgili yerine kelime eklendi
        # w = w.lower() 
        if label:
            Dictionary.sınıf1.append(w)
        else:
            Dictionary.sınıf2.append(w)
      
np.set_printoptions(threshold=sys.maxsize)
Dictionary.sınıf2 = np.unique(Dictionary.sınıf2)    # Sözlükte birden fazla kez bulunan kelimeler azaltıldı
Dictionary.sınıf1 = np.unique(Dictionary.sınıf1)

print(len(Dictionary.sınıf1))
print(len(Dictionary.sınıf2))

# İlk popülasyonun oluşması için RASGELE BİREYLER OLUŞTURULUP POP DİZİSİNE ATANDI ve fitnessları hesaplandı
for i in range(POPULATION_SIZE):
    birey1 = get_birey()
    fit = degerlendirme(birey1)
    pop.append(birey1)
    fitnesses.append(fit)
  
pop_ort_fitness =[]
pop_max_fitness =[]
average = Average(fitnesses)
pop_ort_fitness.append(Average(fitnesses))
pop_max_fitness.append(max(fitnesses))
print("0. Nesil Fitness değerleri: ",fitnesses)
print("0. Nesil Ortalama Fitness değeri:",Average(fitnesses))
print("0. Nesil En yüksek Fitness değeri:",max(fitnesses))

i=0
eniyi_fitness =0
while True:       # yeterli iyiliğe ulaşana kadar nesil devam et
    print((i+1),".nesil") 
    pop , fitnesses , mutasyon_yuzde , eniyi_birey , eniyi_fitness = yeni_pop_olustur(pop, fitnesses, mutasyon_yuzde,i+1, eniyi_fitness)
    
    if len(eniyi_birey)>0:
        eniyi_kelimeler = eniyi_birey  
        
    print("Mutasyon Yüzdesi:",mutasyon_yuzde)  #nesil devam ettikçe azalacak
    print("Fitness değerleri:",fitnesses)
    average = Average(fitnesses)
    maxfit = max(fitnesses)
    pop_ort_fitness.append(average)
    pop_max_fitness.append(maxfit)
    print("Ortalama Fitness değeri:",average)
    print("En yüksek Fitness değeri:",maxfit)
      
    #Durma koşulu: Son 10 popülasyon fitnes ortlaması, son pop'un ortalaması arasındaki farka bakılır
    #Bu fark istenenden küçük olduğunda nesil durur
    if( i>10):
        son10ort = Average(pop_ort_fitness[-10:]) 
        print("Son 10 ortalamanın ortalaması - şu anki ortalama:", abs(son10ort - average))
        if abs(son10ort - average) < 0.05:
            break
    print("En iyi Birey:", eniyi_kelimeler)
    print("En iyi Fitness:", eniyi_fitness) 
    i +=1
    
print("En iyi Birey:", eniyi_kelimeler)
print("En iyi Fitness:", eniyi_fitness)    

from matplotlib import pyplot as plt
# ortalamaların grafiği
plt.plot(pop_ort_fitness , label='Ortalama Fitness Değerleri')
plt.plot(pop_max_fitness , label='Maximum Fitness Değerleri')
plt.legend()
plt.show()





