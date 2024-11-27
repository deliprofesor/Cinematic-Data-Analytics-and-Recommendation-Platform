# Gerekli kütüphanelerin import edilmesi
import pandas as pd  # Veri işleme için
import numpy as np  # Sayısal işlemler için
import seaborn as sns  # Veri görselleştirme için
import matplotlib.pyplot as plt  # Grafik çizimleri için
from sklearn.model_selection import train_test_split  # Eğitim ve test veri ayırma
from sklearn.linear_model import LinearRegression  # Doğrusal regresyon modeli
from sklearn.metrics import (mean_squared_error, 
                              mean_absolute_error, 
                              r2_score)  # Model değerlendirme metrikleri
from sklearn.cluster import KMeans  # Kümeleme algoritması
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Ölçeklendirme ve kodlama
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Rastgele Orman modelleri
from sklearn.feature_extraction.text import TfidfVectorizer  # Metin vektörleştirme
from textblob import TextBlob  # Metin duygu analizi

# Kullanılmayan importların kaldırılması
# Not: sklearn.metrics.pairwise.cosine_similarity kullanılmadığı için kaldırıldı.

# Veri setini yükleme
# movies_path değişkenine yol atanıyor
movies_path = 'C:\\Users\\LENOVO\\Desktop\\tmbm\\tmdb_5000_movies.csv'

# Veri seti pandas ile okunuyor
try:
    movies_df = pd.read_csv(movies_path)
    print("Veri seti başarıyla yüklendi.")
except FileNotFoundError:
    print("Veri seti bulunamadı. Lütfen doğru yolu kontrol edin.")

# 1. Eksik Verilerin İşlenmesi
# Not: Median ile dolduruluyor. Ancak verilerin anlamına göre bu yöntem her zaman uygun olmayabilir.

# Eksik veri içeren sütunlar median (ortanca) değeriyle dolduruluyor
missing_columns = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
for col in missing_columns:
    movies_df[col].fillna(movies_df[col].median(), inplace=True)

print("Eksik veri doldurma işlemi tamamlandı.")

# 2. Aykırı Değerlerin İşlenmesi
# Not: Aykırı değerler belirli bir eşikle sınırlanıyor (örneğin %95'lik dilim).

# Budget sütunu için %95'lik eşik değeri belirleniyor
threshold_budget = movies_df['budget'].quantile(0.95)

# Budget sütunundaki aykırı değerler bu eşikle sınırlandırılıyor
movies_df.loc[movies_df['budget'] > threshold_budget, 'budget'] = threshold_budget

print(f"'budget' sütunundaki aykırı değerler %95'lik eşik değeri olan {threshold_budget} ile sınırlandırıldı.")

# 3. Veri Tipi Dönüşümleri
# release_date sütunu datetime formatına çevriliyor
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

# Hatalı veya boş değerler NaT (Not a Time) olarak atanıyor
missing_dates = movies_df['release_date'].isna().sum()
print(f"'release_date' sütununda {missing_dates} geçersiz tarih değeri bulundu ve NaT olarak atandı.")

# 4. Log Dönüşümü
# Log dönüşümü (1 eklenerek negatif ve sıfır değerlerden kaçınılıyor)
movies_df['log_budget'] = np.log1p(movies_df['budget'])
movies_df['log_revenue'] = np.log1p(movies_df['revenue'])

print("Log dönüşümleri başarıyla tamamlandı: 'log_budget' ve 'log_revenue' sütunları oluşturuldu.")

# 5. Kategorik Verilerin Kodlanması
# genres sütunu sayısal değerlere dönüştürülüyor
encoder = LabelEncoder()

# NaN değerler 'Unknown' ile değiştirilerek kodlama yapılabilir
movies_df['genres'] = movies_df['genres'].fillna('Unknown')
movies_df['genres_encoded'] = encoder.fit_transform(movies_df['genres'].astype(str))

print("Kategorik veriler (genres) başarıyla kodlandı ve 'genres_encoded' sütunu eklendi.")

# 6. K-Means Kümeleme Algoritması

# Kullanılacak özellikler seçiliyor
features = ['popularity', 'vote_count', 'revenue']
movies_df_cluster = movies_df[features].dropna()

# Veriyi ölçeklendirme
scaler = StandardScaler()
movies_df_cluster_scaled = scaler.fit_transform(movies_df_cluster)

# KMeans kümeleme algoritması uygulanıyor
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 küme
movies_df['cluster'] = kmeans.fit_predict(movies_df_cluster_scaled)

# Kümeleme sonuçlarını görselleştirme
plt.figure(figsize=(8, 6))
sns.scatterplot(x=movies_df['popularity'], y=movies_df['revenue'], hue=movies_df['cluster'], palette='viridis', s=100)
plt.title('Filmleri Kümele - Popülerlik ve Gelir')
plt.xlabel('Popülerlik')
plt.ylabel('Gelir')
plt.legend(title="Küme", loc="best")
plt.show()

# 7. Başarı Tahmini (Sınıflandırma)


# Gelir eşik değeri (median) kullanılarak filmler başarılı (1) ve başarısız (0) olarak etiketleniyor
success_threshold = movies_df['revenue'].median()
movies_df['success'] = (movies_df['revenue'] > success_threshold).astype(int)

# Kullanılacak özellikler belirleniyor
X_classification = movies_df[['budget', 'popularity', 'vote_count', 'runtime', 'vote_average']]
y_classification = movies_df['success']

# Eksik verilerin kontrol edilmesi ve kaldırılması
X_classification = X_classification.dropna()
y_classification = y_classification.loc[X_classification.index]

# Eğitim ve test verisinin ayrılması
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42)

# Random Forest modeli oluşturma ve eğitme
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_class, y_train_class)

# Test verisi üzerinde tahmin yapma
y_pred_class = rf_model.predict(X_test_class)

# Modelin değerlendirilmesi
from sklearn.metrics import classification_report
print("\nBaşarı Tahmini Modeli (Random Forest) - Sınıflandırma Raporu:")
print(classification_report(y_test_class, y_pred_class))

# Başarı tahmin sonuçlarının görselleştirilmesi
plt.figure(figsize=(8, 6))
sns.countplot(x=y_test_class, hue=y_pred_class, palette="Set2")
plt.title('Başarı Tahmini: Gerçek ve Tahmin Edilen')
plt.xlabel('Gerçek Başarı Durumu')
plt.ylabel('Film Sayısı')
plt.legend(title='Tahmin', loc='best')
plt.show()

# 7. Başarı Tahmini (Sınıflandırma)

# Gelir eşik değeri (median) kullanılarak filmler başarılı (1) ve başarısız (0) olarak etiketleniyor
success_threshold = movies_df['revenue'].median()
movies_df['success'] = (movies_df['revenue'] > success_threshold).astype(int)

# Kullanılacak özellikler belirleniyor
X_classification = movies_df[['budget', 'popularity', 'vote_count', 'runtime', 'vote_average']]
y_classification = movies_df['success']

# Eksik verilerin kontrol edilmesi ve kaldırılması
X_classification = X_classification.dropna()
y_classification = y_classification.loc[X_classification.index]

# Eğitim ve test verisinin ayrılması
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42)

# Random Forest modeli oluşturma ve eğitme
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_class, y_train_class)

# Test verisi üzerinde tahmin yapma
y_pred_class = rf_model.predict(X_test_class)

# Modelin değerlendirilmesi
from sklearn.metrics import classification_report
print("\nBaşarı Tahmini Modeli (Random Forest) - Sınıflandırma Raporu:")
print(classification_report(y_test_class, y_pred_class))

# Başarı tahmin sonuçlarının görselleştirilmesi
plt.figure(figsize=(8, 6))
sns.countplot(x=y_test_class, hue=y_pred_class, palette="Set2")
plt.title('Başarı Tahmini: Gerçek ve Tahmin Edilen')
plt.xlabel('Gerçek Başarı Durumu')
plt.ylabel('Film Sayısı')
plt.legend(title='Tahmin', loc='best')
plt.show()

# 8. Zaman Serisi Analizi (Yıllık Gelir ve Bütçe


# 'release_date' sütunundan yıl bilgisi çıkarılıyor
movies_df['release_year'] = movies_df['release_date'].dt.year

# Yıllık gelir ve bütçe toplamı hesaplanıyor
annual_revenue_budget = movies_df.groupby('release_year')[['revenue', 'budget']].sum()

# Gelir ve bütçe trendlerini görselleştirme
plt.figure(figsize=(12, 6))
annual_revenue_budget[['revenue', 'budget']].plot(kind='line', marker='o')
plt.title('Yıllık Gelir ve Bütçe Trendleri', fontsize=16)
plt.xlabel('Yıl', fontsize=12)
plt.ylabel('Toplam Gelir ve Bütçe', fontsize=12)
plt.legend(['Gelir', 'Bütçe'])
plt.grid(True)
plt.show()

# 9. Doğal Dil İşleme (NLP)

# 9.1. Anahtar Kelime Çıkarımı (TF-IDF)
# 'overview' sütunundaki boş değerler dolduruluyor
movies_df['overview'] = movies_df['overview'].fillna('')

# TF-IDF vektörleştirme
tfidf = TfidfVectorizer(stop_words='english', max_features=10)
tfidf_matrix = tfidf.fit_transform(movies_df['overview'])

# Anahtar kelimeler belirleniyor
feature_names = np.array(tfidf.get_feature_names_out())
top_keywords = feature_names[np.argsort(tfidf_matrix.sum(axis=0)).flatten()[-10:]]
print("Anahtar Kelimeler:")
print(top_keywords)

# 9.2. Duygu Analizi (Sentiment Analysis) - TextBlob
# Metin duygu analizi için yardımcı fonksiyon
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Duygu analizi uygulanıyor
movies_df['sentiment'] = movies_df['overview'].apply(lambda x: get_sentiment(x))

# Duygu analizi sonuçlarını görselleştirme
plt.figure(figsize=(8, 6))
sns.histplot(movies_df['sentiment'], bins=50, color='skyblue')
plt.title('Filmlerin Duygu Analizi Sonuçları', fontsize=16)
plt.xlabel('Duygu Puanı', fontsize=12)
plt.ylabel('Film Sayısı', fontsize=12)
plt.show()


# 10. Gelir Tahmini (Bütçe ile Doğrusal Regresyon)

# Özellik ve hedef değişkenler seçiliyor
X_budget = movies_df[['budget']]  # Bağımsız değişken: Bütçe
y_revenue = movies_df['revenue']  # Bağımlı değişken: Gelir

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_budget, y_revenue, test_size=0.2, random_state=42)

# Doğrusal regresyon modeli oluşturuluyor
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapılıyor
y_pred_budget = linear_model.predict(X_test)

# Model performansı değerlendiriliyor
mse_budget = mean_squared_error(y_test, y_pred_budget)
r2_budget = r2_score(y_test, y_pred_budget)

# Sonuçlar yazdırılıyor
print("Bütçe ve Gelir Modeli (Doğrusal Regresyon):")
print(f"- Mean Squared Error (MSE): {mse_budget:.2f}")
print(f"- R^2 Skoru: {r2_budget:.2f}")

# Sonuçların görselleştirilmesi
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['budget'], y=y_test, label="Gerçek Değerler", color="blue")
sns.lineplot(x=X_test['budget'], y=y_pred_budget, label="Tahmin Edilen Değerler", color="red")
plt.title('Bütçe ve Gelir Arasındaki İlişki', fontsize=16)
plt.xlabel('Bütçe', fontsize=12)
plt.ylabel('Gelir', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 11. Popülerlik Tahmini (Çoklu Regresyon)

# Kullanılacak özellikler ve hedef değişken belirleniyor
X_multireg = movies_df[['budget', 'vote_count', 'runtime', 'vote_average']]
y_popularity = movies_df['popularity']

# Eksik veri kontrolü ve kaldırılması
X_multireg = X_multireg.dropna()
y_popularity = y_popularity.loc[X_multireg.index]

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_multireg, y_popularity, test_size=0.2, random_state=42)

# Çoklu regresyon modeli oluşturuluyor
multi_reg_model = LinearRegression()
multi_reg_model.fit(X_train, y_train)

# Tahminler yapılıyor
y_pred_popularity = multi_reg_model.predict(X_test)

# Modelin değerlendirilmesi
mse_popularity = mean_squared_error(y_test, y_pred_popularity)
r2_popularity = r2_score(y_test, y_pred_popularity)

# Sonuçlar yazdırılıyor
print("\nPopülerlik Tahmini Modeli (Çoklu Regresyon):")
print(f"- Mean Squared Error (MSE): {mse_popularity:.2f}")
print(f"- R^2 Skoru: {r2_popularity:.2f}")

# Sonuçların görselleştirilmesi
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_popularity, color="green", label="Tahminler")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Mükemmel Çizgi")
plt.title('Popülerlik Tahmini: Gerçek ve Tahmin Edilen Değerler', fontsize=16)
plt.xlabel('Gerçek Popülerlik', fontsize=12)
plt.ylabel('Tahmin Edilen Popülerlik', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 12. Gelir ve Popülerlik Arasındaki İlişki - Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=movies_df, x='popularity', y='revenue', color='purple')
plt.title('Gelir ve Popülerlik Arasındaki İlişki', fontsize=16)
plt.xlabel('Popülerlik', fontsize=12)
plt.ylabel('Gelir', fontsize=12)
plt.grid(True)
plt.show()

# 13. İçerik Tabanlı Tavsiye Sistemi

# genres, keywords ve overview sütunlarındaki boş değerler temizleniyor
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')
movies_df['overview'] = movies_df['overview'].fillna('')

# İçerik birleştirilerek zenginleştiriliyor
movies_df['content'] = movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['overview']

# TF-IDF vektörleştirme ile metin özellikleri çıkarılıyor
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['content'])

# Cosine similarity hesaplanıyor
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Tavsiye fonksiyonu
def get_content_recommendations(movie_index, cosine_sim_matrix, movies_df, top_n=5):
    """
    Belirli bir filmin benzerlerini bulur ve önerir.
    :param movie_index: Tavsiye alınacak filmin indexi
    :param cosine_sim_matrix: Cosine similarity matrisi
    :param movies_df: Veri seti (DataFrame)
    :param top_n: Öneri sayısı
    :return: Önerilen filmlerin başlıkları ve benzerlik puanları
    """
    # Cosine similarity sıralaması
    similarity_scores = list(enumerate(cosine_sim_matrix[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # İlk top_n benzer film (kendisi hariç)
    similar_movies = similarity_scores[1:top_n + 1]
    recommendations = [(movies_df.iloc[i]['title'], score) for i, score in similar_movies]

    return recommendations

# Örnek: 10. film için tavsiye al
movie_index = 10
recommended_movies = get_content_recommendations(movie_index, cosine_sim, movies_df)

print(f"'{movies_df.iloc[movie_index]['title']}' filmi için öneriler:")
for title, score in recommended_movies:
    print(f"- {title} (Benzerlik Puanı: {score:.2f})")

# 14. Kullanıcı Tabanlı Tavsiye Sistemi

# Ağırlıklı puan hesaplama fonksiyonu
def calculate_weighted_score(row, m=1000, C=None):
    """
    IMDB tarzı ağırlıklı puan hesaplama.
    :param row: Veri satırı
    :param m: Minimum oy sayısı
    :param C: Ortalama puan
    :return: Ağırlıklı puan
    """
    if C is None:
        C = movies_df['vote_average'].mean()
    v = row['vote_count']
    R = row['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

# Ortalama puan ve minimum oy sayısı hesaplanıyor
mean_vote_average = movies_df['vote_average'].mean()
min_votes = 1000

# Ağırlıklı puan hesaplanıyor
movies_df['weighted_score'] = movies_df.apply(calculate_weighted_score, axis=1, m=min_votes, C=mean_vote_average)

# Popüler filmler ağırlıklı puana göre sıralanıyor
popular_movies = movies_df.sort_values(by='weighted_score', ascending=False)

# En popüler 5 filmi yazdır
print("\nEn Popüler Filmler (Ağırlıklı Puan):")
print(popular_movies[['title', 'weighted_score']].head(5))

# 15. Hibrit Tavsiye Sistemi

# Hibrit puan hesaplanıyor: İçerik tabanlı ve kullanıcı tabanlı önerilerin ortalaması
movies_df['hybrid_score'] = 0.5 * movies_df['weighted_score'] + 0.5 * movies_df['vote_average']

# Hibrit puana göre sıralama
hybrid_recommendations = movies_df.sort_values(by='hybrid_score', ascending=False)

# En iyi 5 filmi yazdır
print("\nHibrit Tavsiyeler (İçerik ve Kullanıcı Tabanlı):")
print(hybrid_recommendations[['title', 'hybrid_score']].head(5))

