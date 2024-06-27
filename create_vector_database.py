import faiss
import numpy as np
from pandas import read_excel, read_csv, DataFrame
from openai import OpenAI
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

def load_classes():
    data = list(read_excel("part_categories_list.xlsx").values.T[0])[:10]
    #data = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[1])
    return data

def embed_and_save_to_file(data, file_name):
    embedded_data = []
    for i in data:
        embedded_item = get_embedding(i)
        embedded_data.append(embedded_item)
    data_for_dataframe = {
        "classes": data,
        "embeddings": embedded_data
    }
    df = DataFrame(data_for_dataframe)
    df.to_csv(file_name, index=False)


def test1():
    classes = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[1])
    #print("Classes:", classes)
    objects = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[0])
    #print("Objects:", objects)
    index = create_faiss_index(data=classes)
    results = []
    res_count = 0
    for obj_ind, obj in enumerate(objects):
        found_ind = list(search_by_query(index, obj, k=3)[0])
        if obj_ind in found_ind:
            res_count += 1
        results.append((obj_ind, found_ind))
    print("Count:", res_count)
    #print(*results, sep="\n")


if __name__ == "__main__":
    data = load_classes()
    print(len(data))
    #embs = get_embedding(data=data)
    #print(embs)
    #index = create_faiss_index(data=data)
    #print(list(search_by_query(index, "Truck Bed Side Rail Anchor", k=4)[0]))
    #test1()
    #embed_and_save_to_file(data, "embedded_classes3000.csv")
    #classes, index = create_index_from_loaded_embeddings("embedded_classes38741.csv")
    #print(classes)
    #print(list(search_by_query(index, "Truck Bed Side Rail Anchor", k=4)[0]))