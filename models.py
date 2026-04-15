from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

def get_knn_model():
    return KNeighborsClassifier(n_neighbors=5)

def get_dt_model():
    return DecisionTreeClassifier(criterion='entropy', random_state=42)

def get_kmeans_model():
    return KMeans(n_clusters=2, random_state=42)