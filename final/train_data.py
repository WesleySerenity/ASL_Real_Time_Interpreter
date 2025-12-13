import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load the dataset
dataset_pickle_path = "./data.pickle"

with open(dataset_pickle_path, "rb") as dataset_file:
    loaded_dataset = pickle.load(dataset_file)



# convert to numpy
#makes it easier to train
feature_matrix = np.asarray(loaded_dataset["data"])
label_vector   = np.asarray(loaded_dataset["labels"])
# quick train/test split
test_fraction = 0.20
x_train, x_test, y_train, y_test = train_test_split(
    feature_matrix,
    label_vector,
    test_size=test_fraction,
    shuffle=True,
    stratify=label_vector
)

# model of choice
# Random Forest (good baseline)
# (fast for a web app)
rf_model = RandomForestClassifier()

# train
rf_model.fit(x_train, y_train)
#test
test_predictions = rf_model.predict(x_test)

test_accuracy = accuracy_score(y_test, test_predictions)



print("{:.2f}% of samples were classified correctly !".format(test_accuracy * 100))


#save model so the Flask app can load it later
model_output_path = "model.p"

with open(model_output_path, "wb") as model_file:
    pickle.dump({"model": rf_model}, model_file)
