from grakel import graph_from_networkx, NeighborhoodHash
from sklearn.svm import SVC

from kaggle_data_challenge.utils.load import load_my_data
import pandas as pd

def main():
    train_data, train_labels, test_data = load_my_data()

    train_data_grakel = list(graph_from_networkx(train_data, node_labels_tag="labels", edge_labels_tag="labels"))
    test_data_grakel = list(graph_from_networkx(test_data, node_labels_tag="labels", edge_labels_tag="labels"))

    for graph in train_data_grakel:
        attributes = graph[1]
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        graph[1] = new_attributes

        edge_attributes = graph[2]
        new_edge_attributes = {key: edge_attributes[key][0] for key in edge_attributes.keys()}
        graph[2] = new_edge_attributes

    for graph in test_data_grakel:
        attributes = graph[1]
        new_attributes = {key: attributes[key][0] for key in attributes.keys()}
        graph[1] = new_attributes

        edge_attributes = graph[2]
        new_edge_attributes = {key: edge_attributes[key][0] for key in edge_attributes.keys()}
        graph[2] = new_edge_attributes

    kernel_computer = NeighborhoodHash(n_jobs=16, R=4, nh_type="count_sensitive")
    K_train = kernel_computer.fit_transform(train_data_grakel)
    K_test = kernel_computer.transform(test_data_grakel)

    clf = SVC(kernel='precomputed', probability=True)
    clf.fit(K_train, train_labels)

    y_pred = clf.predict_log_proba(K_test)
    y_pred_logit = y_pred[:, 1] - y_pred[:, 0]
    Yte = {'Prediction': y_pred_logit}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv('test_pred.csv', index_label='Id')


if __name__ == '__main__':
    main()