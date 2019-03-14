import itertools


class Multiclass:
    def __init__(self):
        self.cols_name = ['obscene', 'insult', 'toxic', 'severe_toxic', 'identity_hate', 'threat']

    def transform_label(self, df) -> list:
        """

        :param df: dataframe containing multi labels
        :return: new label vector with classes representing combination of labels
        """
        results = [str(classe) for classe in range(1, len(self.cols_name) + 1)]
        classes = [str(classe) for classe in range(1, len(self.cols_name) + 1)]
        ynew = []
        for x in range(2, len(self.cols_name) + 1):
            results.extend([''.join(comb) for comb in itertools.combinations(classes, x)])
        sample_size = df.shape[0]
        for sample in range(0, sample_size):
            y = ''
            for idx, label in enumerate(self.cols_name):
                if df[label][sample] != 0:
                    y = ''.join([y, str(idx + 1)])
            if y:
                ynew.append(y)
            else:
                ynew.append('0')
        print(len(ynew))
        for classe in results:
            print('On a {} commentaires dans la classe {}'.format(ynew.count(classe), classe))
        return ynew