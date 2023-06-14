# ai-projects
Bu repoda yapay zeka dersi kapsamında geliştirilen 3 adet proje ve bu projelerin ayrıntılı raporları bulunmaktadır.  


## Proje-1- Genetik algoritma / Tepe tırmanma ile metin sınıflandırma 
Bu projede kullanılan verisetleri [bu linkteki](https://www.kaggle.com/datasets/savasy/multiclass-classification-data-for-turkish-tc32) yorumlardan oluşturulmuştur. 

Genetik Algoritma olasılıklar üzerinden yakınsama kriterleri altında arama yapan bir fonksiyondur. Doğada gözlemlenen evrimsel sürece benzer bir şekilde -en iyinin yaşaması gibi- çalışır. Bu projede genetik algoritma, yorumlara doğru sınıf atamasının yapılması için kullanılmıştır.

Bu çalışma, her biri 100 örnekten oluşan 5 adet veri seti üzerinde test edilmiştir. Veri setleri "ulaşım" ve "finans" kategorilerine ait etiketli verilerden oluşmaktadır. Raporda [YapayZekaRapor.pdf](https://github.com/colakzeyn/ai-projects/blob/main/Project-1/YapayZekaRapor.pdf) , "birey boyutu" ve "popülasyon büyüklüğü" parametrelerindeki değişimin test sonuçları üzerindeki etkisi grafikler üzerinden incelenmiştir. Sonuç olarak bu hiperparametrelerinin başarıyı yüksek oranda etkileyen parametreler olduğu kanıtlanmıştır. 

## Proje-2- Özellik Seçimi, Özellik Dönüşümü ve Normalizasyon Tekniklerinin Modellerin Başarılarına Etkisi
Bu proje kapsamında Trendyol sitesinden 3000 adet yorum çekilip bir veri seti oluşturulmuştur.  
Veri setine [bu linke](https://github.com/elifayanoglu/ai-projects/blob/main/Project-2/veri_3000.csv) tıklayarak ulaşabilirsiniz.  

5 farklı tahminleyici algoritma 10 fold cross validation yöntemi uygulanarak eğitilmiştir.   
Asıl amacımız olan "başarıya etki eden faktörlerin tespit edilmesi" durumunu ortaya koyabilmek adına aşağıda bulunan işlemler sırasıyla tüm tahminleyici algoritamlara entegre edilerek sonuçlar gözlemlenmiştir.  

1- Chi Squared Tekniği (Özellik Seçimi)  
2- Principal Component Analysis (Özellik Dönüşümü)  
3- Maximum Absolute Scaling Tekniği (Normalizasyon)  
  
En son tahminleyici algortimaların bu işlemlerle birlikte performanslarının arasında anlamlı bir fark olup olmadığını test etmek için "t-test" ile ölçüm yapılmıştır.  
  
Elde ettiğimiz sonuçlara dair detaylı açıklamalar, tablolar ve grafikleri [RAPOR](https://github.com/elifayanoglu/ai-projects/blob/main/Project-2/RAPOR.pdf) isimli pdf'te yer alan makalede inceleyebilirsiniz.  
  
Kodlara [yapayzeka-odev2-kod.py](https://github.com/elifayanoglu/ai-projects/blob/main/Project-2/Kod/yapayzeka-odev2-kod.py) dosyasından ulaşabilirsiniz.  
Bu projeye dair yukarıda bahsetmiş olduğumuz tüm kaynakları [Project-2](https://github.com/elifayanoglu/ai-projects/tree/main/Project-2) klasöründe bulabilirsiniz.


## Proje-3- Kişilik Analizi Projesi 
Bu proje kapsamında insanlara kendileri ve düşünceleri hakkında çeşitli sorular yönelttiğimiz toplamda 13 sorudan oluşan bir [anket](https://github.com/elifayanoglu/ai-projects/blob/main/Project-3/Kodlar%20ve%20veriseti/Ki%C5%9Filik%20Anketi%20.pdf) hazırlanmıştır.  
  
Bu anket ile 224 kişiden veri toplanmıştır. Toplanan veri setini [bu linke](https://github.com/elifayanoglu/ai-projects/blob/main/Project-3/Kodlar%20ve%20veriseti/kisilik_testi.csv) tıklayarak inceleyebilirsiniz.  
  
İlk aşamada veri setine çeşitli görselleştirme teknikleri uygulanarak elde ettiğimiz veriden oran bazında anlamlı sonuçlar çıkarılmıştır. Ayrıca veri setine çeşitli yöntemler uygulanarak işlenmiş ve [islenmis_csv](https://github.com/elifayanoglu/ai-projects/blob/main/Project-3/Kodlar%20ve%20veriseti/islenmis_csv.csv) ismiyle kaydedilmiştir.   
İkinci aşamdaki modellerin çalışabilmesi için işlenmiş olan veri seti kullanılmıştır. Bu aşamaya ait kodlar [yapayzeka_part1.py](https://github.com/elifayanoglu/ai-projects/blob/main/Project-3/Kodlar%20ve%20veriseti/yapayzeka_part1.py) dosyasında mevcuttur.  
  
İkinci aşamada 5 farklı model ile eğitim gerçekleştilimiş ve elde edilen sonuçlar görselleştirilerek çıkarım ve yoruma elverişli hale getirilmiştir.  
İkinci aşamaya ait kodlar [yapayzeka_part2.py](https://github.com/elifayanoglu/ai-projects/blob/main/Project-3/Kodlar%20ve%20veriseti/yapayzeka_part2.py) dosyasında mevcuttur.  
  
Yukarıda anlatılan tüm adımların, görselleştirmelerin  ve çıkarımların ayrıntılı bir şekilde açıklanmış biçimini [YapayZekaProje_Rapor](https://github.com/elifayanoglu/ai-projects/blob/main/Project-3/YapayZekaProje_Rapor.pdf) isimli pdf'te bulunan makalede inceleyebilirsiniz.  
Yukarıda bahsi geçen tüm kaynaklara [Project-3](https://github.com/elifayanoglu/ai-projects/tree/main/Project-3) klasöründen ulaşabilirsiniz.  
  


----
### EN:
In this repo, there are 3 projects developed within the scope of artificial intelligence course and detailed reports of these projects.
