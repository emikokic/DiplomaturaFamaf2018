from sklearn import metrics


class Evaluator:

    def evaluate(self, model, X, y_true):
        y_pred = model.predict(X)
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')

        return {'acc': acc, 'f1': f1}

    def print_short_eval(self, model, X, y_true):
        res = self.evaluate(model, X, y_true)
        print('Accuracy = {acc:2.2f}  |  Macro F1 = {f1:2.2f}'.format(**res))

    def print_eval(self, model, X, y_true):
        y_pred = model.predict(X)
        acc = metrics.accuracy_score(y_true, y_pred)

        print('Accuracy = {:2.2f}\n'.format(acc))
        print(metrics.classification_report(y_true, y_pred,
                                            target_names=['neg', 'pos']))

        cm = metrics.confusion_matrix(y_true, y_pred)
        print(cm)
