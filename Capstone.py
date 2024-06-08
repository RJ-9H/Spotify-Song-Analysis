#Capstone project 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import seaborn as sns
import scipy
from scipy.special import expit
spotifyData = pd.read_csv(r"C:\Users\rjhaf\Downloads\spotify52kData.csv")
seed = 13857844
np.random.seed(13857844)

#Cleaning/prep
spotifyData.dropna(inplace = True)

#setting alpha 
alpha = .05

#Correlation Matrix
correlation_matrix = spotifyData.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap = 'coolwarm')
plt.title("Correlation Matrix")
plt.show()


#Consider the 10 song features duration, danceability, energy, loudness, speechiness,
#acousticness, instrumentalness, liveness, valence and tempo. Is any of these features
#reasonably distributed normally? If so, which one? [Suggestion: Include a 2x5 figure with
#histograms for each feature) 
songFeatures = spotifyData[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns
axes = axes.flatten()  # Flatten the array of axes for easy iteration

ksNormal = []
for i, ax in enumerate(axes):
    featureName = songFeatures.columns[i]
    normalDist = np.random.normal(loc = np.mean(songFeatures.iloc[:, i]), scale = np.std(songFeatures.iloc[:, i]), size = 100000)
    ax.hist(songFeatures.iloc[:, i], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel(featureName)
    ax.set_ylabel('Frequency')
    
    #Conduct a KS test with generated normal distribution
    test = scipy.stats.kstest(songFeatures.iloc[: , i], normalDist)
    ksNormal.append(test)
plt.tight_layout()
plt.show()

for i, test in enumerate(ksNormal):
    print(f"Feature {i + 1} KS Test Statistic: {test.statistic}, p-value: {test.pvalue}")
#Looking at the results of the ks-test and setting an alpha of 1% we see
#that we can reject the null and none of these distributions are distributed normally

#2 Is there a relationship between song length and popularity of a song? If so, if the relationship
#positive or negative? [Suggestion: Include a scatterplot]
duration = spotifyData[['duration']]
popularity = spotifyData[['popularity']]
plt.scatter(duration, popularity)
plt.xlabel('Duration(ms)')
plt.ylabel('Popularity')
plt.show()
correlation_coef, p_value = scipy.stats.pearsonr(spotifyData['duration'], spotifyData['popularity'])
corr_spearmann, p_spearman = scipy.stats.spearmanr(spotifyData['duration'], spotifyData['popularity'])
print('Pearson correlation:', correlation_coef)
print('Spearmann correlation:', corr_spearmann)


#3 Are explicitly rated songs more popular than songs that are not explicit? [Suggestion: Do a
#suitable significance test, be it parametric, non-parametric or permutation]

explicit = spotifyData[spotifyData["explicit"]==True]
not_explicit = spotifyData[spotifyData["explicit"]==False]
plt.hist(spotifyData['popularity'], bins = 30, color='skyblue', edgecolor='black')
plt.xlabel('Popularity')
plt.ylabel('frequency')
plt.show()

#Spike at 0 since so it is not reasonable to reduce it to sample means, the data is not categorical 
#and we are comparing which has higher rating (so we want to compare medians) we use Mann Whitney U test

statistic, pvalue = scipy.stats.mannwhitneyu(explicit['popularity'], not_explicit['popularity'], alternative = 'greater')
print('Mann-Whitney U test p value:', pvalue)
if pvalue < alpha:
    print('Reject the null. There is siginficant difference between explicity and not explicit popularity')
    if explicit['popularity'].median() > not_explicit['popularity'].median():
        print('Explcit songs are rated higher overall with a median of {0} while nonexplicit songs had a median rating {1}'.format(explicit['popularity'].median(), not_explicit['popularity'].median()))
    else:
        print('Nonexplcit songs are rated higher overall with a median of {1} while explicit songs had a median rating {0}'.format(explicit['popularity'].median(), not_explicit['popularity'].median()))
else:
    print('Fail to reject the null')
    
#4 Are songs in major key more popular than songs in minor key? [Suggestion: Do a suitable
#significance test, be it parametric, non-parametric or permutation]

#This will have the same process as 3, from the distribution of the popularity histogram we will use the Mann Whitney U test

major = spotifyData[spotifyData["mode"]==1]
minor = spotifyData[spotifyData["mode"]==0]
keyStatistic, p_value = scipy.stats.mannwhitneyu(minor["popularity"], major["popularity"], alternative='greater')
print('Mann Whitney U test for major/minor p-value:', p_value)
if p_value < alpha:
    print('Reject the null. There is siginficant difference between explicity and not explicit popularity')
    if major['popularity'].median() > minor['popularity'].median():
        print('Songs in a major key are rated higher overall with a median of {0} while songs in a minor key had a median rating {1}'.format(major['popularity'].median(), minor['popularity'].median()))
    else:
        print('Songs in a minor key are rated higher overall with a median of {1} while songs in a major key had a median rating {0}'.format(major['popularity'].median(), minor['popularity'].median()))
else:
    print('Fail to reject the null')
    
#5 Energy is believed to largely reflect the “loudness” of a song. Can you substantiate (or refute)
#that this is the case? [Suggestion: Include a scatterplot]
energy = spotifyData[['energy']]
loudness = spotifyData[['loudness']]
plt.scatter(energy, loudness)
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.title('Energy - Loudness')
plt.show()
#Noticeable trend between energy and loudness but non-linear so we will calculate a spearmann coefficient
spearman_corr, spearman_p = scipy.stats.spearmanr(energy, loudness)
print("Spearman correlation coefficient:", spearman_corr)
print("P-value:", spearman_p)

#6 Which of the 10 individual (single) song features from question 1 predicts popularity best?
#How good is this “best” model?
R2 = []
MSE = []
features = []

for i in range(songFeatures.shape[1]):   
    x = songFeatures.iloc[:, i].values.reshape(-1,1)
    y = spotifyData[['popularity']]
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= .2, random_state = seed) 
    modelReg = LinearRegression().fit(X_train, y_train)
    y_pred = modelReg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    R2.append(r2)
    MSE.append(mse)
    features.append(songFeatures.columns[i])
plt.figure(figsize=(10, 6))
plt.bar(features, R2)
plt.xlabel('Features')
plt.ylabel('R^2 Score')
plt.title('R^2 Score for Different Features')
plt.xticks(rotation=45)
plt.show()
#The best predictor out of all the attributes is instrumentalness at .022 which is still incredibly low
#Overall not a good predictor 

#7 Building a model that uses *all* of the song features from question 1, how well can you
#predict popularity now? How much (if at all) is this model improved compared to the best
#model in question 6). How do you account for this?

#part 1 perforaming a linear regression
x = songFeatures.values
y = spotifyData[['popularity']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed) 

# Train linear regression model
modelReg = LinearRegression().fit(X_train, y_train)

# Make predictions
y_pred = modelReg.predict(X_test)

# Calculate R^2
r2all = r2_score(y_test, y_pred)
print("R^2 Score:", r2all)

#Part 2 Ridge regression 
ridgealpha = 1000 #changed alpha between 10, 100, 1000 and 10,000 little to no change from orginial
ridge_model = Ridge(alpha).fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

r2_ridge = r2_score(y_test, y_pred)
print("R^2 Score (Ridge Regression):", r2_ridge)
#Part 3 Lasso 
lassoalpha = 1000 #changed alpha between 10, 100, 1000 and 10,000 little to no change from orginial
Lasso_model = Lasso(alpha).fit(X_train, y_train)
y_pred = Lasso_model.predict(X_test)

r2_Lasso = r2_score(y_test, y_pred)
print("R^2 Score (LASSO):", r2_Lasso)

#8 When considering the 10 song features above, how many meaningful principal components
#can you extract? What proportion of the variance do these principal components account for? 
featureData = songFeatures.to_numpy()
zscoredData = scipy.stats.zscore(featureData) #Z-scoring the data to conduct a pca

pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_ #eigenvectors associated with eigenvalues above

rotatedData = pca.fit_transform(zscoredData) * -1 #(polarity)
varExplained = eigVals/sum(eigVals)*100


print("variance explained for each component:")
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))

#Scree plot 
numQuestions = 10
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigVals, color='gray')
plt.plot([0,numQuestions],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()
#Observe there are 3 principle components with eigenvalue 1 
var3components = varExplained[0]+ varExplained[1]+ varExplained[2]
print('Variance accounted for by the 3 components:{:.2f}'.format(var3components))

#9 Can you predict whether a song is in major or minor key from valence? If so, how good is this
#prediction? If not, is there a better predictor? [Suggestion: It might be nice to show the logistic
#regression once you are done building the model]
mode = spotifyData[['mode']]
valence = spotifyData[['valence']]
plt.scatter(valence, mode, edgecolor = 'black')
plt.xlabel('Valence')
plt.ylabel('Mode')
plt.show()
#Visually there appears to be no indication of valence being a good predictor of key(mode)
X_train, X_test, y_train, y_test = train_test_split(valence, mode, test_size=0.2, random_state=seed)
modelLogistic = LogisticRegression().fit(X_train, y_train)
y_pred = modelLogistic.predict(X_test)

#now we want to show the sigmodal function 
x_input = np.linspace(-50, 50, 1000).reshape(-1, 1)
y_out = x_input* modelLogistic.coef_ + modelLogistic.intercept_
y_prob = expit(y_out)
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(x_input, y_prob, color = 'red')
plt.xlabel('Valence')
plt.ylabel('Mode')
plt.show()

#Now we want to look at the AUROC so we first need to look at the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')


#3924 with false positive and 6476 true positive.
#0 false negatives and 0 true negatives which means the logistic regression is not good at all
y_prob = modelLogistic.predict_proba(X_test)[:, 1]  # Probability of being in the major key

# Compute AUROC
auroc = roc_auc_score(y_test, y_prob)
print("AUROC Score:", auroc)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auroc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Which is a better predictor of whether a song is classical music – duration or the principal
#components you extracted in question 8? [Suggestion: You might have to convert the
#qualitative genre label to a binary numerical label (classical or not)]

#First we need to convert the necessary data for genre into binary 
spotifyData['classical'] = (spotifyData['track_genre'] == 'classical').astype(int)

#Now we will look at duration as a predictor.
classical = spotifyData[['classical']]
duration = spotifyData[['valence']]
plt.scatter(valence, mode, edgecolor = 'black')
plt.xlabel('Duration')
plt.ylabel('Classical')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(duration, classical, test_size=0.2, random_state=seed)
modelLogistic = LogisticRegression().fit(X_train, y_train)
y_pred = modelLogistic.predict(X_test)

#now we want to show the sigmodal function 
x_input = np.linspace(-50, 50, 1000).reshape(-1, 1)
y_out = x_input* modelLogistic.coef_ + modelLogistic.intercept_
y_prob = expit(y_out)
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(x_input, y_prob, color = 'red')
plt.xlabel('Duration')
plt.ylabel('Classical')
plt.show()

#Now we want to look at the AUROC so we first need to look at the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')



y_prob = modelLogistic.predict_proba(X_test)[:, 1]  #

# Compute AUROC
auroc = roc_auc_score(y_test, y_prob)
modelAccuracy = accuracy_score(y_pred, y_test)

print("AUROC Score:", auroc)
print("Model Accuracy:", modelAccuracy)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auroc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Now we will use the 3 Principle components as predictos.
pca1 = rotatedData[:, 0]*-1
pca2 = rotatedData[:, 1]*-1
pca3 = rotatedData[:, 2]*-1
principalComponents = np.column_stack((pca1, pca2, pca3))


X_train, X_test, y_train, y_test = train_test_split(principalComponents, classical, test_size=0.2, random_state=seed)
modelLogistic = LogisticRegression().fit(X_train, y_train)
y_pred = modelLogistic.predict(X_test)

x_input = np.linspace(-50, 50, 1000).reshape(-1, 1)
y_out = x_input* modelLogistic.coef_ + modelLogistic.intercept_
y_prob = expit(y_out)

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')



y_prob = modelLogistic.predict_proba(X_test)[:, 1]  #

# Compute AUROC
auroc = roc_auc_score(y_test, y_prob)
modelAccuracy = accuracy_score(y_pred, y_test)
print("AUROC Score:", auroc)
print("Model Accuracy:", modelAccuracy)
# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auroc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


#Bonus look at the time signature, tempo and key to see how it informs danceability
danceFeatures = spotifyData[['time_signature', 'key',  'tempo']]
x = danceFeatures.values
y = spotifyData[['danceability']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed) 

# Train linear regression model
modelReg = LinearRegression().fit(X_train, y_train)

# Make predictions
y_pred = modelReg.predict(X_test)

# Calculate R^2
r2all = r2_score(y_test, y_pred)
print("R^2 Score:", r2all)
#Now lets look at the individual plots 
R2_dance = []
MSE_dance = []
features_dance = []

for i in range(danceFeatures.shape[1]):   
    x = danceFeatures.iloc[:, i].values.reshape(-1,1)
    y = spotifyData[['danceability']]
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= .2, random_state = seed) 
    modelReg = LinearRegression().fit(X_train, y_train)
    y_pred = modelReg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    R2_dance.append(r2)
    MSE_dance.append(mse)
    features_dance.append(danceFeatures.columns[i])

plt.figure(figsize=(10, 6))
plt.bar(features_dance, R2_dance)
plt.xlabel('Dance Features')
plt.ylabel('R^2 Score')
plt.title('R^2 Score for Different Features')
plt.xticks(rotation=45)
plt.show()

#Observe how none of these variables are good predictors of dance as the coefficient of determination
#is low for all 3 variables and the use of all of them combined.

