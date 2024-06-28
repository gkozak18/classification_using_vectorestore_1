import config
from create_vector_database import create_index_from_loaded_embeddings, predict_classes
from pandas import read_excel

from langchain.prompts import ChatPromptTemplate

template = """You are a smart classification model which gets as an input a product title and a list of categories in which you need
to clasify the product title. You just need to answer the name of the class without any additional words.
Product title: {title}
List of categories: {classes}"""

prompt = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

choose_classes_chain = (
    prompt
    | ChatOpenAI(temperature = 0)
    | StrOutputParser()
)

def choose_class(title, classes):
    res_class = choose_classes_chain.invoke({
        "title": title,
        "classes": classes
    })
    return res_class

def test2():
    filename = "embedded_classes_0_38741.csv"
    classes, index = create_index_from_loaded_embeddings(filename)
    print(len(classes))

    objects = objects = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[0])
    obj_classes = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[1])
    res_count_yes = 0
    res_count_true = 0
    for i, obj in enumerate(objects):
        title, found_classes = predict_classes(index, classes, obj, k=50)
        res = choose_class(title, found_classes)
        if obj_classes[i] in found_classes:
            print("Yees!!!", obj_classes[i], found_classes)
            res_count_yes += 1
        else:
            print("No((", obj_classes[i], found_classes)
        if res == obj_classes[i]:
            print("True!!!", title, res)
            res_count_true += 1
        else:
            print("False((", title, res)
    print("Result Count Yes:", res_count_yes)
    print("Result Count True:", res_count_true)


if __name__ == "__main__":
    test2()
