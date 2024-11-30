# Cinematic-Data-Analytics-and-Recommendation-Platform

![movie](https://github.com/user-attachments/assets/78507f89-5b83-4d01-b937-c1f76344046d)

This project aims to perform data exploration, segmentation, and modeling on wholesale customer data using various clustering algorithms such as K-Means, DBSCAN, and Gaussian Mixture Models (GMM). Additionally, the project leverages Principal Component Analysis (PCA) for data visualization and uses decision trees to predict customer channel preferences based on their attributes.

## Project Overview

The goal of this project is to understand wholesale customer behavior through data exploration, clustering, and prediction. By applying machine learning techniques like clustering and decision trees, this project helps businesses segment customers, detect outliers, and analyze relationships between variables. The results are visualized to make the findings more accessible and actionable.

## Technologies Used

- **Python** – Core programming language for data analysis and modeling.
- **Pandas** – Data manipulation and cleaning.
- **NumPy** – Numerical operations and array manipulation.
- **Scikit-Learn** – Machine learning algorithms (K-Means, DBSCAN, GMM, Decision Trees).
- **Matplotlib** – Plotting and visualization of data.
- **Seaborn** – Statistical data visualization.
- **PCA** – Principal Component Analysis for dimensionality reduction.

## Installation Instructions

To run the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/deliprofesor/Cinematic-Data-Analytics-and-Recommendation-Platform.git
   cd Wholesale-Customer-Segmentation
   
## Project Description

The project uses the TMDB movie dataset, which consists of 4803 rows and 20 different features for each movie. The features include movie titles, genres, budgets, revenues, popularity scores, viewer ratings, release dates, and short summaries of the movies. The dataset provides both numerical and textual data, enabling a wide range of analysis. It requires data preprocessing steps such as handling missing and outlier data to ensure accuracy in further analysis.

## Features in the Dataset:

- **budget (Budget):** The production budget of the movie (numerical).
- **genres (Genres):** The genre information of the movie. In JSON format, it can include multiple genres (e.g., Action, Adventure).
- **homepage:** The official website URL of the movie.
- **id:** Unique identifier for each movie.
- **keywords (Keywords):** Keywords that describe the movie (e.g., "space", "future").
- **original_language (Original Language):** The language code of the movie (e.g., en for English).
- **original_title:** The original title of the movie.
- **overview:** A short summary of the movie.
- **popularity (Popularity):** A numerical score indicating the movie’s viewership and popularity.
- **production_companies:** Information about the production companies (JSON format).
- **production_countries:** Countries where the movie was produced (JSON format).
- **release_date:** The release date of the movie.
- **revenue (Revenue):** The worldwide gross earnings of the movie.
- **runtime:** The length of the movie in minutes.
- **spoken_languages:** The languages spoken in the movie (JSON format).
- **status:** The current status of the movie (e.g., Released, Post Production).
- **tagline:** The slogan or promotional tagline of the movie.
- **title:** The title of the movie.
- **vote_average:** The average viewer rating for the movie.
- **vote_count:** The total number of viewer ratings for the movie.
  
## Project Aim

The core objective of this project is to analyze a movie dataset and develop various models and methods. This includes predicting movie success, examining the relationship between revenue and popularity, extracting insights from textual content using natural language processing (NLP) techniques, and developing recommendation systems to offer suggestions to users. Visualization methods are used to make the results more meaningful. The project provides a strong foundation for understanding trends in the movie industry and creating user-focused recommendation systems.

## 1. Data Loading and Cleaning
The first step in the project involves loading the movie dataset. The data was loaded from a CSV file using pandas, and missing data was examined. For instance, missing values in columns such as budget, revenue, and popularity were filled using the median value to ensure accurate analysis. Outliers in the budget column were capped using a 95% threshold, and the release_date column was converted to datetime format to clean erroneous values. This step is critical for ensuring the accuracy of subsequent analyses.

## 2. Data Analysis and Visualization

Various analysis and visualization methods were used to understand the distributions and relationships in the data. Log transformations were applied to the budget and revenue columns to normalize the distributions. The genres column was converted into numerical values to make it suitable for classification models. The K-means clustering algorithm was applied to features such as popularity, vote_count, and revenue, and the movies were grouped into 3 clusters. The clustering results were visualized with scatter plot graphs. Additionally, total revenue and budget trends were analyzed by creating a new column for the release year.

## 3. Predictive Models
Three predictive models were developed for the project:

-**Success Prediction (Random Forest):** A Random Forest model was trained to classify movies as successful (1) or unsuccessful (0) using features such as budget, popularity, vote_count, runtime, and vote_average. This model achieved an accuracy of 85%.
- **Revenue Prediction (Linear Regression):** A linear regression model was trained to predict movie revenue based on the budget. The model showed moderate success in predicting revenue.
- **Popularity Prediction (Multiple Regression):** Features like budget, vote_count, runtime, and vote_average were used to predict movie popularity. The model performed reasonably well in predicting popularity.
  
## 4. Natural Language Processing (NLP)

Text analysis was performed to extract insights from movie summaries. The TF-IDF method was used to extract important keywords from the overview column. Additionally, sentiment analysis was conducted on the summary text using TextBlob, and sentiment scores were computed. The sentiment scores were visualized with histogram charts.

## 5. Recommendation Systems
Three different recommendation systems were developed:

- **Content-Based Recommendation:** The genres, keywords, and overview columns were combined and vectorized using TF-IDF. Similar movies were recommended using the cosine similarity method.
- **User-Based Recommendation:** A system was developed to rank movies based on user ratings. An IMDB-style weighted rating system was used to recommend the top 5 most popular movies.
- **Hybrid Recommendation System:** The results of the content-based and user-based systems were combined, and the top 5 movies were recommended based on a hybrid score.
