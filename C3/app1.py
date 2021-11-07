import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()
# import znanych nam bibliotek

filename = "model2.sv"
model = pickle.load(open(filename, 'rb'))


# otwieramy wcześniej wytrenowany model
def main():
    st.set_page_config(page_title="Czy jesteś zdrowy?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    with overview:
        st.title("Czy jesteś zdrowy?")

    with right:
        symptoms_slider = st.slider("Objawy", value=1, min_value=1, max_value=5)
        age_slider = st.slider("Wiek", value=50, min_value=1, max_value=100)
        disease_slider = st.slider("Choroby współistniejące", value=0, min_value=0, max_value=5)
        height_slider = st.slider("Wzrost (cm)", min_value=0, max_value=300)

    data = [[symptoms_slider, age_slider, disease_slider, height_slider]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.header("Czy dana osoba jest chora? {0}".format("Tak" if survival[0] == 1 else "Nie"))
        st.subheader("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()

## Źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic), zastosowanie przez Adama Ramblinga
