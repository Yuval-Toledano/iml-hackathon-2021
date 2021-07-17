import pickle
from preprocessing import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

classifier_classes = [RandomForestClassifier, DecisionTreeClassifier, ExtraTreeClassifier,
                      SVC, Perceptron, PassiveAggressiveClassifier, LogisticRegression]

CLASSIFIER_PATH = "classifier.weights"


def load_data():
    train_filename = "Data/train_dataset.pkl"
    # validate_filename = "Data/validate_dataset.pkl"
    test_filename = "Data/test_dataset.pkl"

    with open(train_filename, "rb") as fh:
        train_x, train_y = pickle.load(fh)

    # with open(validate_filename, "rb") as fh:
    #     validate_x, validate_y = pickle.load(fh)

    with open(test_filename, "rb") as fh:
        test_x, test_y = pickle.load(fh)

    return train_x, train_y, test_x, test_y


def create_tree(cls, train_x, train_y):
    classifier = cls()
    classifier.fit(train_x, train_y)
    return classifier


def score(classifier, test_x, test_y):
    print(classifier.score(test_x, test_y))
    pass


def save_classifier(vote_clf):
    with open(CLASSIFIER_PATH, "wb") as fh:
        pickle.dump(vote_clf, fh)


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data()
    train_x = clean_data(train_x)
    test_x = clean_data(test_x)

    # cv = KFold(n_splits=15)
    rand_forest_classifier = RandomForestClassifier(n_estimators=68, max_depth=13)
    # rand_forest_classifier.fit(train_x, train_y)
    # scores = cross_val_score(rand_forest_classifier, train_x, train_y, scoring='accuracy', cv=cv)
    # print(scores)
    # print(scores.mean())
    # print(scores.std())
    # print(f"RandomForestClassifier, score={rand_forest_classifier.score(validate_x, validate_y)}")

    decision_tree_classifier = AdaBoostClassifier(
        DecisionTreeClassifier(criterion="entropy", max_depth=8, min_samples_split=300),
        n_estimators=10
    )
    # decision_tree_classifier.fit(train_x, train_y)
    # print(f"DecisionTreeClassifier, score={decision_tree_classifier.score(validate_x, validate_y)}")

    logistic_reg_classifier = LogisticRegression(multi_class="multinomial", max_iter=800)
    # logistic_reg_classifier.fit(train_x, train_y)
    # print(f"LogisticRegression, score={decision_tree_classifier.score(validate_x, validate_y)}")

    print("Voting calssifier fitting")
    vote_clf = VotingClassifier(estimators=[('rf', rand_forest_classifier), ('dt', decision_tree_classifier),
                                            ('lr', logistic_reg_classifier)], voting='soft')
    vote_clf.fit(train_x, train_y)
    print(f"voting committee, score={vote_clf.score(test_x, test_y)}")

    save_classifier(vote_clf)
