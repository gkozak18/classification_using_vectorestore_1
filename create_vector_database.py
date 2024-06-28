import faiss
import numpy as np
from pandas import read_excel, read_csv, DataFrame
from openai import OpenAI
from time import time
import config

client = OpenAI()

def get_embedding(data, model="text-embedding-3-small"):
   return client.embeddings.create(input = data, model=model).data[0].embedding

def create_faiss_index(data, dimension=1536):
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array([get_embedding(d) for d in data])
    index.add(vectors)
    return index

def create_index_from_loaded_embeddings(file_name):
    df = read_csv(file_name)
    classes = df.values.T[0]
    embeddings = df.values.T[1]
    for i, embd in enumerate(embeddings):
        arr = embd[1:-1].split(", ")
        embeddings[i] = list(map(float, arr))
    index = faiss.IndexFlatL2(len(embeddings[0]))
    vectors = np.array([i for i in embeddings])
    #print(embeddings[0])
    #print(vectors.shape)
    index.add(vectors)
    return classes, index

def search_faiss_index(index, query_vector, k=1):
    distances, indices = index.search(np.array([query_vector]), k)
    return indices

def search_by_query(index, query, k=1):
    #print(query)
    query_vector = get_embedding(data=query)
    index_of_found_vector = search_faiss_index(index, query_vector, k=k)
    return index_of_found_vector

def load_classes(vonbis=(0, 10)):
    # Size of Full Dataset is 38741
    von = vonbis[0]
    bis = vonbis[1]
    data = list(read_excel("part_categories_list.xlsx").values.T[0])#[von:bis]
    #data = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[1])
    return data

def embed_and_save_to_file(data, file_name, model="text-embedding-3-small"):
    t1 = time()
    n = len(data)
    embedded_data = []
    for i in range(n // 1000 + 1 * (n % 1000 > 0)):
        if (i + 1) * 1000 < n - 1:
            current_embedded_data = client.embeddings.create(input = data[i*1000:(i+1)*1000], model=model).data
            print((i+1)*1000)
        else:
            current_embedded_data = client.embeddings.create(input = data[i*1000:n], model=model).data
            print(n)
        for j, embd in enumerate(current_embedded_data):
            current_embedded_data[j] = embd.embedding
        embedded_data += current_embedded_data
        #print(len(current_embedded_data))
    print(len(embedded_data))
    #for i in data:
    #    embedded_item = get_embedding(i)
    #    embedded_data.append(embedded_item)
    data_for_dataframe = {
        "classes": data,
        "embeddings": embedded_data
    }
    df = DataFrame(data_for_dataframe)
    df.to_csv(file_name, index=False)
    t2 = time()
    print("time:", t2 - t1)


def predict_classes(index, ind_classes, obj, k=1):
    found_ind = list(search_by_query(index, obj, k=k)[0])
    found_classes = []
    for i in found_ind:
        found_classes.append(ind_classes[i])
    #print(*results, sep="\n")
    return (obj, found_classes)


def test1(ind_classes, index, k=1):
    if len(ind_classes) == 0:
        ind_classes = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[1])
        index = create_faiss_index(data=classes)
    #print("Classes:", classes)
    obj_classes = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[1])
    objects = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[0])
    #print("Objects:", objects)
    results = []
    res_count = 0
    for obj_ind, obj in enumerate(objects):
        found_ind = list(search_by_query(index, obj, k=k)[0])
        found_classes = []
        for i in found_ind:
            found_classes.append(ind_classes[i])
        obj_class = obj_classes[obj_ind]
        if obj_class in found_classes:
            res_count += 1
            print("TRUE!!!", obj_class, found_classes)
        else:
            print("FALSE((", obj_class, found_classes)
        results.append((obj_class, found_classes))
    print("Count:", res_count)
    #print(*results, sep="\n")


def create_embedding_files():
    # vonbis = (0, 38741)
    vonbis = (0, 100)
    data = load_classes(vonbis)
    print(len(data))
    #embs = get_embedding(data=data)
    #print(embs)
    #index = create_faiss_index(data=data)
    #print(list(search_by_query(index, "Truck Bed Side Rail Anchor", k=4)[0]))
    embed_and_save_to_file(data, f"embedded_classes_{vonbis[0]}_{vonbis[1]}.csv")


if __name__ == "__main__":
    filename = "embedded_classes_0_38741.csv"
    classes, index = create_index_from_loaded_embeddings(filename)
    print(len(classes))
    #print(list(search_by_query(index, "Truck Bed Side Rail Anchor", k=4)[0]))
    test1(classes, index, k=20)