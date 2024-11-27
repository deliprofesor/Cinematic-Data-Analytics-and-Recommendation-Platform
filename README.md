# Cinematic-Data-Analytics-and-Recommendation-Platform

![movie](https://github.com/user-attachments/assets/78507f89-5b83-4d01-b937-c1f76344046d)

Projede, TMDB'nin sağladığı film veri seti kullanılmıştır. Bu veri seti, her bir filmi temsil eden 4803 satır ve 20 farklı özelliğe sahiptir. Özellikler arasında film başlıkları, türleri, bütçeleri, gelirleri, popülerlik puanları, izleyici oyları, çıkış tarihleri ve filmlerin kısa özetleri gibi bilgiler bulunmaktadır. Veri seti, filmlerle ilgili hem sayısal hem de metinsel veriler içererek geniş bir analiz yelpazesine olanak tanır. Veri seti, eksik ve aykırı veriler gibi sorunları çözmeyi gerektirerek veri işleme adımlarını daha önemli hale getirir.

- **budget (Bütçe): Filmin prodüksiyon bütçesi (sayısal).**
- **genres (Türler): Filmin tür bilgisi. JSON formatında, birden fazla tür içerebilir (ör. Action, Adventure).**
- **homepage: Filmin resmi web sitesi URL'si.**
- **id: Her filme özel benzersiz kimlik numarası.**
- **keywords (Anahtar Kelimeler): Filmi tanımlayan anahtar kelimeler (ör. "space", "future").**
- **original_language (Orijinal Dil): Filmin dil kodu (ör. en İngilizce).**
- **original_title: Filmin orijinal adı.**
- **overview: Filmin kısa özeti.**
- **popularity (Popülerlik): Filmin izlenme ve beğenilme derecesi (sayısal bir skor).**
- **production_companies: Yapımcı şirket bilgileri (JSON formatında).**
- **production_countries: Filmin üretildiği ülkeler (JSON formatında).**
- **release_date: Filmin vizyon tarihi.**
- **revenue (Gelir): Filmin dünya çapındaki toplam hasılatı.**
- **runtime: Filmin süresi (dakika cinsinden).**
- **spoken_languages: Filmde konuşulan diller (JSON formatında).**
- **status: Filmin durumu (ör. Released, Post Production).**
- **tagline: Filmin sloganı veya tanıtım cümlesi.**
- **title: Filmin adı.**
- **vote_average: İzleyici oylamalarının ortalaması.**
- **vote_count: İzleyici oylamalarının toplam sayısı.**

# Projenin Amacı 

Bu projenin temel amacı, bir film veri setini analiz ederek farklı modeller ve yöntemler geliştirmektir. Bu kapsamda, filmlerin başarısını tahmin etmek, gelir ve popülerlik ilişkisini incelemek, doğal dil işleme (NLP) tekniklerini kullanarak metinsel içeriklerden içgörüler çıkarmak ve izleyicilere öneriler sunmak üzere tavsiye sistemleri geliştirilmiştir. Sonuçların daha anlamlı hale getirilmesi için veri görselleştirme yöntemleri kullanılmıştır. Proje, film endüstrisindeki trendleri anlamak ve kullanıcı odaklı öneri sistemleri oluşturmak açısından güçlü bir temel sağlar.


# 1. Veri Yükleme ve Temizleme

Projenin ilk adımı, film veri setinin yüklenmesidir. Veri, pandas ile CSV formatından yüklenmiş ve eksik veriler incelenmiştir. Örneğin, budget, revenue ve popularity gibi sütunlardaki eksik değerler, veri analizlerini etkilememesi için ortanca (median) değerlerle doldurulmuştur. Ayrıca, budget sütunundaki aykırı değerler %95'lik eşik değeri kullanılarak sınırlandırılmış ve release_date sütunu datetime formatına dönüştürülerek hatalı değerler temizlenmiştir. Bu adım, sonraki analizlerin doğruluğunu artırmak için kritik bir aşamadır.

# 2. Veri Analizi ve Görselleştirme

Verinin dağılımlarını ve ilişkilerini anlamak için çeşitli analizler ve görselleştirme yöntemleri kullanılmıştır. Log dönüşümleri, budget ve revenue sütunlarında yapılmış ve dağılımlar normalize edilmiştir. genres sütunu sayısal değerlere dönüştürülerek sınıflandırma modellerine uygun hale getirilmiştir. K-means kümeleme algoritması, popularity, vote_count ve revenue gibi özellikler üzerinde uygulanmış ve filmler 3 kümeye ayrılmıştır. Küme sonuçları, scatterplot grafiklerle görselleştirilmiştir. Ayrıca, release_year bilgisi türetilerek yıllara göre toplam gelir ve bütçe trendleri analiz edilmiştir.

# 3. Tahmin Modelleri

Proje kapsamında üç farklı tahmin modeli geliştirilmiştir:

- **Başarı Tahmini (Random Forest): Filmlerin başarılı (1) veya başarısız (0) olarak sınıflandırılması amacıyla, budget, popularity, vote_count, runtime ve vote_average özellikleri kullanılarak bir Random Forest modeli eğitilmiştir. Bu model, %85 doğruluk oranıyla başarılı bir performans göstermiştir.**
- **Gelir Tahmini (Doğrusal Regresyon): budget bağımsız değişkeni ile filmlerin revenue değerlerini tahmin etmek için bir doğrusal regresyon modeli eğitilmiştir. Model, gelir tahmininde orta düzeyde bir başarı sağlamıştır.**
- **Popülerlik Tahmini (Çoklu Regresyon): budget, vote_count, runtime ve vote_average gibi özellikler kullanılarak filmlerin popülerlik puanları tahmin edilmiştir. Model, popülerlik tahmininde makul bir performans göstermiştir.**
  
# 4. Doğal Dil İşleme (NLP)

Film özetlerinden içgörüler elde etmek için metin analizi yapılmıştır. TF-IDF yöntemi, overview sütunundaki metinlerden önemli anahtar kelimeler çıkarmak için kullanılmıştır. Bunun yanı sıra, textblob ile her filmin özet metni üzerinde duygu analizi yapılmış ve duygu puanları hesaplanmıştır. Elde edilen duygu puanları, histogram grafiklerle görselleştirilmiştir.

# 5. Tavsiye Sistemleri

Projede üç farklı tavsiye sistemi geliştirilmiştir:

- **İçerik Tabanlı Tavsiye: genres, keywords ve overview sütunlarındaki metinler birleştirilmiş ve TF-IDF ile vektörleştirilmiştir. Cosine similarity yöntemiyle benzer filmler önerilmiştir.**
- **Kullanıcı Tabanlı Tavsiye: Filmleri izleyici oylarına göre sıralayan bir sistem geliştirilmiştir. IMDB tarzı ağırlıklı puanlama yöntemi kullanılarak en popüler 5 film önerilmiştir.**
- **Hibrit Tavsiye Sistemi: İçerik tabanlı ve kullanıcı tabanlı sistemlerin sonuçları birleştirilmiş ve hibrit skorlarla en iyi 5 film önerilmiştir.**
