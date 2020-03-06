import os
import pickle
import datetime
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split


def load_datasets():
    dataset = load_files('../../../sentiment/review_polarity/txt_sentoken',
                         shuffle=False)
    docs_traindev, docs_test, y_traindev, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42
    )

    test = (docs_test, y_test)

    docs_train, docs_dev, y_train, y_dev = train_test_split(
        docs_traindev, y_traindev, test_size=0.2, random_state=42
    )

    train = docs_train, y_train
    dev = docs_dev, y_dev

    return train, dev, test



def load_datasets_unlabeled_test():
    dataset = load_files('../review_polarity_competition/reviews_sentoken',
                         shuffle=False)

    docs_train, docs_dev, y_train, y_dev = train_test_split(
            dataset.data, dataset.target,
            test_size=0.10, random_state=42
    )

    dirname = "../review_polarity_competition/test_reviews_sentoken"
    test = []
    # I do this to keep the files in numeric order
    for fname in range(len(os.listdir(dirname))):
        fname = str(fname) + ".txt"
        with open(os.path.join(dirname, fname)) as fd:
            test.append(fd.read())

    train = docs_train, y_train
    dev = docs_dev, y_dev

    return train, dev, test


def get_best_params(results, length=2):
    results_df = results.copy()

    results_sort = results_df.sort_values(['acc', 'f1'], ascending=False)
    results_sort_dict = results_sort.to_dict(orient='records')

    best_result_list = results_sort_dict[:length]

    # Eliminamos 'acc' y 'f1' del dict
    for best_result in best_result_list:
        del best_result['acc']
        del best_result['f1']

    filter_clf = lambda d: {
                    k[5:]: v for k, v in d.items() if k.startswith('clf__')
                }

    filter_vect = lambda d: {
                    k[6:]: v for k, v in d.items() if k.startswith('vect__')
                }

    best_params_vect_clf = [(filter_vect(best_result),
                             filter_clf(best_result))
                            for best_result in best_result_list]

    return best_params_vect_clf


def save_csv_results(fname, labels):
    current_time = datetime.datetime.now()
    date = current_time.strftime("%Y-%m-%d")

    fname = "{}_{}".format(date, fname)

    with open('Results_CSV/' + fname, 'w') as f:
        f.write("Id,Category\n")
        for i, l in enumerate(labels):
            f.write(str(i) + ".txt," + str(l) + "\n")


def save_pickle_model(model, filename):
    current_time = datetime.datetime.now()
    date = current_time.strftime("%Y-%m-%d")

    filename = "{}_{}".format(date, filename)

    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_pickle_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model
