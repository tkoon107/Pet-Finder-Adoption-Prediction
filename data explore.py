  import os
import pandas as pd
import plotly as py
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r"D:\Project Data\Pet Finder")
train = pd.read_csv(r"./train/train.csv").drop(axis=1, columns ="Description")
test = pd.read_csv(r"./test/test.csv").drop(axis=1, columns ="Description")

train['dataset'] = 'train'
test['dataset'] = 'test'

full_data = pd.concat([train, test])

#full_data['Type_name'] = 'dog'
#dataset.loc[full_data['Type'] == 2, "Type_name"] = "cat"
#full_data["Type"] = full_data["Type"].apply(lambda x: 'dog' if x == 1 else 'cat')

full_data.head()
full_data.describe()

%matplotlib qt


pd.qcut(full_data['Age'], 5).unique()

sns.set_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])

'''
Data dictionary

PetID - Unique hash ID of pet profile
AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
Type - Type of animal (1 = Dog, 2 = Cat)
Name - Name of pet (Empty if not named)
Age - Age of pet when listed, in months
Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
Quantity - Number of pets represented in profile
Fee - Adoption fee (0 = Free)
State - State location in Malaysia (Refer to StateLabels dictionary)
RescuerID - Unique hash ID of rescuer
VideoAmt - Total uploaded videos for this pet
PhotoAmt - Total uploaded photos for this pet
Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.
'''

#spearman rank correlation between varialbes
full_data.corr(method='spearman')


full_data['AdoptionSpeed'].

#Data is compared against the class label in each case

sns.pairplot(iris, hue='species', size=1.5);

plt.subplot(1,3,1)

%matplotlib inline
type_plot = sns.countplot(data = full_data, x='Type', hue = 'AdoptionSpeed')
type_plot.set_xticklabels(["Dog", "Cat"])
age_plot =sns.relplot(data=full_data, x="Age", y='AdoptionSpeed')

  fee_plot = sns.scatterplot(data=full_data, x="Fee", y='AdoptionSpeed')
full_data[['Fee', 'AdoptionSpeed']].groupby('AdoptionSpeed').mean()

health_plot = sns.countplot(data=full_data, hue="AdoptionSpeed", x="Health")
health_plot.set_xticklabels(["Health","Minor Injury","Serious Injury", "Not Specified"])

ster_plot = sns.countplot(data=full_data, hue="AdoptionSpeed", x="Sterilized")
ster_plot.set_xticklabels(["Yes","No","Not Sure"])

vacc_plot = sns.countplot(data=full_data, hue="AdoptionSpeed", x="Vaccinated")
vacc_plot.set_xticklabels(["Yes","No","Not Sure"])
#comments: Quicker rate of adoptions for not vaccinated

gender = sns.countplot(data=full_data, hue="AdoptionSpeed", x='Gender')
gender_plot.set_xticklabels(["Male","Female","Not Specified"])
#comments: Males typically adopted quicker

#Both extreme classes 0 and 4 have around 3 photos, but class 4 the longest time before adoption has the least amount of photos per posting
photo_amt = sns.barplot(x = full_data['AdoptionSpeed'], y = full_data['PhotoAmt'])

#Name wordclouds

#top breed rankings cats/dogs
