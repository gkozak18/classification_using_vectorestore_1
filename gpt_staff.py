import config
from config import llm_gpt4o, llm_gpt35
from create_vector_database import create_index_from_loaded_embeddings, predict_classes
from pandas import read_excel

from langchain.prompts import ChatPromptTemplate

template = """You are a smart classification model which gets as an input a product title and a list of categories in which you need
to clasify the product title. You just need to answer the name of the class without any additional words. List of categories is build the way 
that the items on the begging is more probable tha classes then the itmes closer to end.
Product title: {title}
List of categories: {classes}"""

add = """And one common problem with classification is 
when the class with longer name is choosen instead of the class with smaller but more accurate name. But you also should not forget 
about specific ditailes which distinguish a class from others."""

prompt = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

choose_classes_simple_chain = (
    prompt
    | llm_gpt4o
    | StrOutputParser()
)

def simple_choose_class(title, classes):
    res_class = choose_classes_simple_chain.invoke({
        "title": title,
        "classes": classes
    })
    return res_class

def test2(k=1):
    filename = "embedded_classes_0_38741.csv"
    classes, index = create_index_from_loaded_embeddings(filename)
    print(len(classes))

    objects = objects = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[0])
    obj_classes = list(read_excel("reference_tilte_to_category_mapping_(50).xlsx").values.T[1])
    res_count_yes = 0
    res_count_true = 0
    for i, obj in enumerate(objects):
        title, found_classes = predict_classes(index, classes, obj, k=k)
        res = simple_choose_class(title, found_classes)
        print("Number:", i)
        if obj_classes[i] in found_classes:
            print("YEES, THERE'S A RIGHT CANDIDATE!!!", "REAL CLASS:", obj_classes[i], "CANDIDATES FROM DB:", found_classes)
            res_count_yes += 1
        else:
            print("NO, THERE'S NO RIGHT CANDIDATE((", "REAL CLASS", obj_classes[i], "CANDIDATES FROM DB:", found_classes)
        if res == obj_classes[i]:
            print("TRUE RESULT!!!", "TITLE:", title, "RES:", res)
            res_count_true += 1
        else:
            print("FALSE RESULT((", "TITLE:", title, "RES:", res)
    print("Result Count of Right Candidates in the List:", res_count_yes)
    print("Result Count of True Results:", res_count_true)


if __name__ == "__main__":
    test2(k=13)
