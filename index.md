
# My Portfolio
---
# Hi, I’m Anirban Gangopadhyay 👋

Welcome to my portfolio! I am passionate about leveraging machine learning, deep learning, and AI technologies to solve real-world problems. Over the years, I have worked on several projects spanning various domains, from predictive analytics to advanced gesture recognition systems.

Below, you'll find some of the projects I have worked on. Each project demonstrates my proficiency in using state-of-the-art machine learning techniques, along with a wide range of tools and technologies.

Here are the projects I have worked on:

## Stack Overflow Tag Prediction with GRU and Bidirectional GRU Models using GloVe Embedding
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/AnirbanG-git/stack_overflow_tag_prediction)

This project focuses on the development of machine learning models to predict tags for Stack Overflow questions, facilitating more efficient content categorization and navigation. By leveraging a dataset comprised of Stack Overflow questions, the initiative seeks to employ advanced neural network models, specifically GRU and Bidirectional GRU, enhanced with GloVe embeddings for effective text representation.

The methodology encompasses initial data preprocessing, focusing on the most frequent tags, followed by the exploration of neural network architectures for the task. Utilizing Python and key libraries such as TensorFlow, Keras, NLTK, and Scikit-learn, the project aims at achieving high precision, recall, and F1 scores in multi-class text classification.

Performance analysis reveals that while both GRU and Bidirectional GRU models show promising results, they exhibit trade-offs between precision and recall, and certain challenges in category-specific performance, particularly with 'html' questions. These insights underscore the potential for further model optimization and exploration of additional architectures.

Future directions include experimenting with diverse models and hyperparameters to enhance accuracy and extending the model's capability to predict a broader array of tags. 

## Distracted Driver Multi-Action Classification Using CNN and MobileNet
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/AnirbanG-git/DistractedDriverClassification)

This project is centered on the development of a convolutional neural network (CNN) model to classify images of drivers into ten distinct categories based on their behaviors, such as texting, talking on the phone, driving safely, and more. Utilizing a dataset of driver images, the goal is to accurately identify and classify various driving behaviors to enhance road safety measures.

By organizing the dataset into training, testing, and validation sets, the project employs a methodical approach to data preprocessing and model training. A key part of the methodology involves leveraging transfer learning from MobileNet, combined with a GlobalAveragePooling2D layer and dense layers with dropout regularization to construct a robust CNN model.

Technological tools and libraries such as Python, TensorFlow, Keras, Scikit-learn, and data visualization libraries like Matplotlib and Seaborn are integral to this project, enabling efficient model development and performance evaluation.

The CNN model achieved a commendable accuracy of 86.16% on the validation set, showcasing its potential in accurately classifying distracted driving behaviors. This achievement underscores the model's effectiveness, though it also highlights opportunities for further refinement and enhancement to improve accuracy and robustness.

Future directions for this project include exploring additional data preprocessing and augmentation techniques, experimenting with different CNN architectures and hyperparameters, and extending the model's capabilities to encompass a wider range of driver behaviors and scenarios.

## Quora Spam Question Filtering with GRU and Bidirectional GRU Models using GloVe Embedding
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/AnirbanG-git/QuoraSpamDetection)

This project tackles the challenge of identifying and filtering spam questions on Quora using advanced NLP techniques and deep learning models. With the objective of enhancing content quality and user experience on the platform, the project employs GloVe embeddings for text representation and deep learning models, specifically GRU and Bidirectional GRU, to differentiate between spam and legitimate questions.

Utilizing a dataset of Quora questions, the project undergoes rigorous data preprocessing and exploratory data analysis (EDA) to visualize data distributions and features effectively. The methodology involves using class weights to address the dataset's imbalance— a common challenge in spam detection tasks.

Key technologies and libraries such as Python, TensorFlow, Keras, NLTK, and Scikit-learn play crucial roles in building and training the models. Data visualization tools like Matplotlib and Seaborn are used to analyze model performance and data characteristics, aiding in the iterative process of model refinement.

The outcome of the project is promising, with the GRU model achieving an accuracy of 94.10% and the Bidirectional GRU model achieving 90.60%. Despite the high accuracy, the precision for the positive class (spam) highlights an area for improvement, suggesting further exploration into model architectures and class imbalance strategies.

Future work will focus on investigating additional model architectures, hyperparameter tuning, and advanced techniques for handling class imbalance, aiming to improve the precision for spam detection without compromising overall accuracy.


## AI-Powered Laptop Recommendation Chatbot - ShopAssist
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/AnirbanG-git/shopassist)

ShopAssist is an intelligent chatbot developed to assist users in selecting the best laptops based on their preferences and requirements. The chatbot leverages OpenAI's GPT models with dynamic function calling to deliver personalized recommendations, enhancing user interaction and decision-making.

### Key Features:
- **Interactive Conversations**: Engages users in a conversation to understand their needs, such as GPU intensity, portability, multitasking, and budget.
- **Personalized Laptop Recommendations**: Dynamically suggests the top 3 laptops based on the user’s specific requirements, utilizing a custom-built function calling API.
- **Moderation Layer**: Incorporates a moderation feature to ensure the chatbot stays focused on relevant topics, improving response accuracy.
- **Asynchronous CSV Generation**: Built-in admin interface to generate and update the laptop catalog CSV in the background, ensuring real-time data management.

### Strategic Insights:
- **Function Calling API**: Integrated OpenAI’s Function Calling API to trigger dynamic responses and execute tailored functions, providing highly accurate and relevant recommendations.

### Technologies Used:
- **Python**: Core language for developing the application.
- **Flask**: Web framework used for managing routes and user sessions.
- **Pandas**: Utilized for data manipulation and CSV handling.
- **OpenAI API**: Powers the chatbot with GPT models and Function Calling.

### Future Enhancements:
- **Model Expansion**: Incorporating additional models to improve recommendation accuracy.
- **User Profiles**: Implementing persistent user profiles to enable users to return and continue their sessions.


## Gesture Recognition for Smart TVs
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/AnirbanG-git/GestureRecognition)

This project focuses on the development of a gesture recognition feature for smart TVs, enabling users to control the TV without a remote by using five specific gestures. These gestures correspond to commands such as increasing or decreasing volume, jumping forward or backward in a video, and pausing the video.

### Dataset Description:
- **Data Source**: The dataset consists of video frames categorized into five gesture classes. Each gesture is split into 30 frames, and the data is organized into training and validation sets. CSV files provide information on the video subfolders, gesture names, and numeric labels.
- **Data Preprocessing**: Involves shuffling data, cropping images, resizing to standard dimensions, normalizing pixel values, and optionally augmenting the images with random transformations.

### Model and Methods:
- **DataLoader Class**: Handles loading and shuffling training and validation data.
- **DataGenerator Class**: Generates batches of preprocessed image sequences and corresponding labels for training and validation.
- **ModelManager Class**: Builds, compiles, and trains the gesture recognition model dynamically based on different architectures.
- **Trainer Class**: Manages the training process, including data preparation, model building, and performance tracking.
- **CustomLearningRateScheduler Class**: Adjusts the learning rate during training based on a predefined schedule.

### Model Architectures:
- Each model is defined in a separate architecture file (e.g., `model_15.py`), using the `build_model` function. The CNN+GRU architecture with transfer learning from MobileNet (model_15.py) achieved the best performance.
- **Best Model**: A combination of 2D CNN for feature extraction from video frames and GRU (Gated Recurrent Unit) for sequence processing. The model achieved:
  - **Training Accuracy**: 99%
  - **Validation Accuracy**: 98%
  
This model is highly effective for real-time gesture recognition in smart TVs.

### Ablation:
Ablation studies were conducted using stratified sampling to reduce the dataset size while preserving the label distribution. This experiment provided insights into model performance on smaller subsets of data.

### Technologies Used:
- **Python Version**: 3.8.10
- **NumPy Version**: 1.19.4
- **Skimage Version**: 0.19.2
- **TensorFlow Version**: 2.7.0
- **Matplotlib Version**: 3.5.0
- **Scikit-learn Version**: 0.24.1

### Conclusion:
The CNN+GRU model with transfer learning is highly accurate in recognizing gestures for controlling smart TVs. Future work could explore further fine-tuning and testing with larger, more diverse datasets to ensure robustness across different user inputs and environments.

## BoomBikes: Leveraging Predictive Analytics to Revitalize Post-Pandemic Bike-Sharing Demand
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/AnirbanG-git/BoomBikeSharing)

This project employs multiple linear regression analysis to forecast post-pandemic demand for BoomBikes, a bike-sharing service impacted by COVID-19. Analyzing a dataset reflecting American market trends, we identified key demand influencers, including weather conditions, seasonality, and temporal shifts. This insight helps BoomBikes align their strategies with environmental cues and market dynamics to boost customer engagement and revenue.

**Strategic Insights:**
- **Weather and Seasonality:** High demand correlates with warmer temperatures and clear skies. Seasonal changes significantly affect bike rentals, highlighting the need for targeted bike availability and marketing in warmer months.
- **Demand Growth:** An upward trend in bike rentals suggests increasing popularity. It emphasizes the potential for strategic expansion and enhanced promotional efforts.

**Model Findings:**
- Developed two models; the latter, incorporating binned variables for temperature and weather conditions, showed superior accuracy. This model's success underscores the importance of nuanced factors like weather and season in demand prediction.

**Recommendations:**
-ocus resources on peak demand periods, identified as warmer seasons with clear weather.
-Leverage the growing trend in bike-sharing to expand market presence and customer base.

**Technologies Used:**

Python and libraries such as pandas, numpy, matplotlib, and seaborn facilitated the data analysis, supported by the Anaconda platform.

## Melanoma Detection Using Multiclass Classification
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/AnirbanG-git/SkinCancerDetectionCNN)

### Overview
This project aims to build a reliable multiclass classification model to detect melanoma, a deadly skin cancer, using TensorFlow. By leveraging images from the International Skin Imaging Collaboration (ISIC), the model assists in reducing the manual effort required for diagnosis by dermatologists.

### Dataset
The dataset consists of 2357 images, representing various skin diseases. Key classes include:
- **Actinic Keratosis**
- **Basal Cell Carcinoma**
- **Dermatofibroma**
- **Melanoma**
- **Nevus**
- **Pigmented Benign Keratosis**
- **Seborrheic Keratosis** (fewest samples)
- **Squamous Cell Carcinoma**
- **Vascular Lesion**

*Note*: The dataset is highly imbalanced, particularly with fewer samples for Seborrheic Keratosis.

### Models Developed
1. **Model 1: Vanilla CNN**
   - Initial model showed signs of overfitting, with a 45% discrepancy between training and validation accuracies.

2. **Model 2: Augmented CNN using Keras**
   - Introducing a data augmentation layer in Keras reduced overfitting, bringing down the accuracy gap to 5%, but with a slight decrease in overall accuracy.

3. **Model 3: Augmented CNN using Augmentor**
   - Refinements using the Augmentor library significantly reduced overfitting, achieving ~80% accuracy. While the model performs well, further improvements are possible.

### Technologies Used
Since the versions of libraries often change, here are the specific versions used in this project:
- **Python**: 3.10.14
- **TensorFlow**: 2.16.1
- **Augmentor**: 0.2.12
- **NumPy**: 1.26.4
- **Matplotlib**: 3.8.4
- **Seaborn**: 0.12.2
- **PIL**: 10.3.0
- **Anaconda**: 23.5.2

### How to Use
Details on how to set up, train, and evaluate the models are provided in the [README](https://github.com/AnirbanG-git/SkinCancerDetectionCNN#readme) section of the GitHub repository. This includes instructions for setting up the environment, data preprocessing, and model training.

### Acknowledgments
A big thank you to the **International Skin Imaging Collaboration (ISIC)** for providing the dataset used in this project.


## Lending Club Loan Default Prediction
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/AnirbanG-git/LendingClubCaseStudy)

This project applies Exploratory Data Analysis (EDA) to predict loan defaults for the Lending Club, aiding in refining loan approval processes and minimizing credit losses. Through detailed analysis of a dataset containing loan applicant information, we identified key predictors of loan default, enabling more informed lending decisions.

**Key Insights:**

- **Predictive Variables:** Loan term, grade, interest rate, and annual income emerged as significant predictors of default. High-interest rates and loan amounts, particularly in the 'Medium-High' and 'Very High' categories, indicated increased default risk.
- **Binning Impact:** Categorizing continuous variables (e.g., loan amount, interest rate) improved predictive accuracy. Specifically, loans with terms of 60 months, higher loan amounts, and interest rates above 12% showed a higher likelihood of default.
- **Geographic Trends:** Loan defaults varied by geography, with states like Florida and California showing higher default rates, suggesting the importance of location in risk assessment.

**Recommendations:**

- **Risk-Based Pricing:** Adjust interest rates for higher-risk loan categories to mitigate potential losses.
- **Loan Term Promotion:** Encourage shorter loan terms (36 months) due to their lower default rates.
- **Enhanced Scrutiny:** Apply more rigorous checks for loans with high interest rates (>12%) and monitor high installment loans closely.
- **Income and Employment Verification:** Strengthen verification processes for these factors, especially for borrowers in lower income brackets.
- **Geographic Strategy**: Tailor risk management strategies to regions with historically higher default rates.

**Technologies:**

Utilized Python along with libraries like pandas, numpy, matplotlib, and seaborn for data analysis and visualization.





