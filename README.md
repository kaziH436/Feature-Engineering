# Feature-Engineering
advanced feature engineering techniques 

Notes from [what is the Fourier Transform? A visual introduction](https://www.youtube.com/watch?v=spUNpyF58BY)

The information is from [Advanced Feature Engineering Techniques for Machine Learning](https://medium.com/@sandunayanathara99/advanced-feature-engineering-techniques-for-machine-learning-0dc6a8a1c905)

## 4 Techniques 

1. Domain-Specific Knowledge

2. Time Series Feature 

    - Fourier Transform
        - Pressure (sum of all amplitudes of wave) agnanist time 
        ### Problem to solve: how to take a signal and decompose to its pure frequencies
        - Uncertainty Principle

                import numpy as np
                from scipy.fft import fft

                def extract_fft_features(signal):
                    return np.abs(fft(signal)) # Because the FFT output is complex, taking the absolute value computes the magnitude (or amplitude) of those complex numbers using sqrt((real)**2 + (imag)**2)


    - Wavelet Transform

        import numpy as np
        import pywt

        def extract_wavelet_cwt_features(signal):
            # Define widths/scales to analyze
            widths = np.arange(1, 31) # defines the scales at which the wavelet will be stretched or compressed. Smaller widths (like 1, 2, 3) compress the wavelet to look for fast, sharp, high-frequency changes. Larger widths (up to 30) stretch it out to capture slow, low-frequency trends
            
            # Perform the Continuous Wavelet Transform
            # 'cmor1.5-1.0' is a complex Morlet wavelet (common for frequency analysis)
            coefficients, frequencies = pywt.cwt(signal, widths, 'cmor1.5-1.0')
            
            # Take the absolute value (magnitude) of the complex coefficients
            return np.abs(coefficients) 




3. TF-IDF (Term Frequency-Inverse Document Frequency): For textual data, techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings (e.g., Word2Vec, GloVe) are essential for converting text into numerical features.


        from sklearn.feature_extraction.text import TfidVectorizer
        vectorizer=TfidVectorizer()
        X=vectorizer.fit_transform(list_of_text_strings)

4. Word embeddings (Word2Vec, GloVe)

        from gensim.models import Word2Vec

        # Tokenize your text strings into a list of list of words
        tokenized_text = [text.split() for text in list_of_text_strings]

        # Train the Word2Vec model
        model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1) # The model will only look at a maximum distance of 5 words to the left and 5 words to the right.
        # min_count: The model looks at your frequency list and completely deletes any word that appears fewer than 2 times. 

        # Get the vector for a specific word
        vector = model.wv["example_word"]



## Advanced Feature Transformation Techniques 

1. Polynomial Features: Polynomial feature transformation involves creating interaction terms and polynomial combinations of features, allowing models to capture non-linear relationships

        from sklearn.proprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

2. Log Transform:  useful for normalizing skewed data, for linear models

        import numpy as np
        X_log=np.log1p(X)

3. Box-Cox and Yeo-Johnson Transformations: These transformations are used to stabilize variance and make the data more normally distributed.

        from scipy.stats import boxcon
        X_boxcon,-=boxcox(X)


## Advanced Feature Selection Techniques

1. Recursive Feature Elimination (RFE)
RFE is a recursive process that iteratively removes the least important features, using a model’s feature importance.

        from sklearn.feature_selection import RFE
        from sklearn.feature_selection import LogisticRegression

        model=LogisticRegression()
        rfe=RFE(model,n_featurs_to_select=10)
        X_rfe=rfe.fit_transform(X,y)

2. Feature Importance from Tree-Based Models: Tree-based models, such as Random Forests and Gradient Boosting, provide feature importances, which can be used to select the most relevant features.

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X,y)
        importances=model.feature_importances_ #contains the importance of each feature.


3. L1 Regularization (Lasso)
Lasso regression can be used for feature selection by penalizing the absolute size of coefficients, effectively shrinking some to zero.

        from sklearn.linear_model import Lasso
        model=Lasso(alpha=0.1)
        model.fit(X,y)
        coefficients=model.coef_




# Feature Scaling
Ensure no feature is overshawdowing the other features in the datasets 

## Standardization


## Normalization
This works when the machine learning model doesn't have outliers so there isn't anything to skew the data
Min-max scaling: 0-1 range

X_new = (X-X_min)/(X_max-X_min)
