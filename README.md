# ai-projects
Bu repoda yapay zeka dersi kapsamında geliştirilen 3 adet proje ve bu projelerin ayrıntılı raporları bulunmaktadır.  

## 2- Özellik Seçimi, Özellik Dönüşümü ve Normalizasyon Tekniklerinin Modellerin Başarılarına Etkisi
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



----
### EN:
In this repo, there are 3 projects developed within the scope of artificial intelligence course and detailed reports of these projects.
