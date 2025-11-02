# Enhancing Customer Experience and Loyalty Measurement through Sentiment Analysis and NPS: Evidence from the Luxury Hotel Industry

## Project Overview

This notebook explores customer reviews from the luxury hotel industry to enhance customer experience and loyalty measurement. By applying Natural Language Processing (NLP) techniques, sentiment analysis, and Net Promoter Score (NPS) calculation, we aim to gain insights into customer opinions across various aspects of their stay and identify key areas for improvement and loyalty building.

## Data

The dataset used in this project is collected from sources like Google Maps, Booking.com, and TripAdvisor, focusing on 5-star luxury hotels. The raw data contains various features related to reviews, including the reviewer's name, publication date, review origin, reviewer ID, star rating, and the review text.

Initial data understanding revealed missing values in several columns, particularly in `reviewContext` and `reviewDetailedRating`. Duplicate values were also identified and handled. Key features for the analysis, such as `name`, `publishedAtDate`, `reviewOrigin`, `reviewId`, `reviewerId`, `stars`, and `text`, were selected.

## Methodology

The project follows a structured approach involving data preprocessing, analysis, modeling, and interpretation:

### Data Preprocessing

- **Handling Missing and Duplicate Values**: Missing values in relevant columns were addressed, and duplicate reviews were removed to ensure data quality.
- **Lowercase Conversion**: Review text was converted to lowercase for consistency.
- **Removing Punctuation, Digits, and Special Characters**: Irrelevant characters were removed from the text.
- **Removing URLs, Emails, and Mentions**: These elements were cleaned from the review text.
- **Handling Emojis**: Emojis were replaced with their textual descriptions.
- **Multilingual Handling**: Reviews in languages other than English were translated to English to facilitate unified analysis.
- **Tokenization**: Review text was split into individual words (tokens).
- **Stopwords Removal**: Common English stopwords were removed to focus on meaningful terms.
- **Lemmatization**: Words were reduced to their base or root form to standardize the vocabulary.

### Exploratory Data Analysis (EDA)

- **N-Gram Analysis**: Unigrams, Bigrams, and Trigrams were analyzed to identify frequent word sequences in reviews, categorized by sentiment (positive and negative).
- **Word Clouds**: Visual representations of the most frequent words in positive and negative reviews were generated to provide a quick overview of prominent themes.

### Data Augmentation

- To address class imbalance, particularly between positive and negative sentiments, data augmentation techniques (specifically Antonym Augmentation using the `nlpaug` library) were applied to the minority class.

### Feature Engineering

- **Vectorization**: Review text was converted into numerical feature vectors using:
    - **Bag of Words (BoW)**: Represents text as a bag of its words, disregarding grammar and even word order, but keeping multiplicity.
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weights words based on their frequency in a document and their inverse frequency across the entire corpus, highlighting important terms.
- **Label Encoding**: The target sentiment labels (`positive`, `negative`) were converted into numerical representations for model training.

### Modeling

- **Machine Learning Models**:
    - Logistic Regression and Multinomial Naive Bayes classifiers were trained and evaluated using both BoW and TF-IDF vectorized features.
    - Performance metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC were used for evaluation.
- **Deep Learning Model (DistilBERT)**:
    - A DistilBERT model was fine-tuned for sentiment classification.
    - The text data was tokenized and prepared using the Hugging Face `transformers` library.
    - The model was trained and evaluated, and its performance was assessed using classification reports, confusion matrices, and ROC curves.

### Aspect Extraction

- Latent Dirichet Allocation (LDA) was used to identify key aspects discussed in the reviews (e.g., Staff & Service, Room & View, Food & Restaurant).
- Sentence embeddings and cosine similarity were employed to assign specific sentences or snippets from reviews to the identified aspects.

### Aspect-Based Sentiment Analysis

- Sentiment analysis (using the fine-tuned DistilBERT model) was applied to the extracted text snippets for each aspect to determine the sentiment towards specific aspects (e.g., positive sentiment towards "Food & Beverage").

### NPS Calculation

- The Net Promoter Score (NPS) was calculated based on the star ratings, classifying reviews as Promoters, Passives, or Detractors.
- The overall NPS score and the distribution of NPS categories were analyzed.
- The trend of NPS over time (quarterly) was visualized.
- Word clouds for each NPS category were generated to understand the language used by each group.

### Repeat vs One-time Reviewers Analysis

- Reviewers were categorized as "One-time" or "Repeat" based on the number of reviews they provided.
- Sentiment distribution across different aspects was analyzed separately for one-time and repeat reviewers.
- Visualizations like bar charts, radar charts, and waterfall charts were used to compare the sentiment profiles and identify key issues raised by repeat customers.

## Results and Insights

The analysis provides valuable insights into customer sentiment and loyalty in the luxury hotel industry. Key findings include:

- **Overall Sentiment**: The overall sentiment distribution and the performance of the trained models (ML and DL) provide an understanding of general customer satisfaction.
- **Aspect-Specific Sentiment**: Aspect-based sentiment analysis reveals specific areas where the hotel excels or needs improvement (e.g., high positive sentiment for "Service & Staff", potential issues with "Room & Facilities" for negative reviews).
- **NPS Score**: The calculated NPS score indicates the overall likelihood of customers to recommend the hotel. The trend analysis shows how this score has changed over time.
- **Repeat Customer Behavior**: The analysis of repeat reviewers highlights their sentiment patterns and the specific aspects they frequently mention with negative sentiment, which is crucial for understanding and addressing the concerns of loyal customers.

