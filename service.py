import catboost as cb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np


def corr_res(val):
    if (val >= 0.3) and (val < 0.5):
        return "умеренная"
    elif (val >= 0.5) and (val < 0.7):
        return "заметная"
    elif (val >= 0.7) and (val < 0.9):
        return "высокая"
    elif (val >= 0.9):
        return "весьма высокая"
    else:
        return "слабая"


def age_vectorize(val):
    age = np.zeros((3, ))
    if val == "от 35 до 50":
        age[0] = 1
    if val == "от 50 до 65":
        age[1] = 1
    if val == "от 65":
        age[2] = 1
    return age

def loans_vectorize(val):
    arr = np.zeros((4, ))
    if val == "от 2 до 4":
        arr[0] = 1
    if val == "от 4 до 6":
        arr[1] = 1
    if val == "от 6 до 8":
        arr[2] = 1
    if val == "свыше 8":
        arr[3] = 1
    return arr


data = pd.read_csv("data/credit_scoring.csv")

with open("cols_map.json", "r", encoding="utf-8") as json_file:
    col_map = json.load(json_file)
col_df = pd.DataFrame(columns=["col", "description"])
col_df["col"] = col_map.keys()
col_df["description"] = col_map.values()

st.set_page_config(
    page_title="Кредитный скоринг",
    layout="wide"
)
st.title("Кредитный скоринг")
placeholder = st.empty()

with placeholder.container():
    st.write("## Краткая справка по архивным данным")

    st.write("Исходные данные")
    st.table(data.head(5))

    raws, columns, memory_usage = st.columns(3)
    raws.metric(
        label="Кол-во строк",
        value=data.shape[0],
        delta=-22081,
        delta_color="inverse"
    )
    columns.metric(
        label="Кол-во столбцов",
        value=data.shape[1],
        delta="+4",
        delta_color="inverse"
    )
    memory_usage.metric(
        label="Потребление памяти",
        value=f"{round(data.memory_usage().sum() / 1024 ** 2, 1)}Мб",
        delta="-7.9Мб",
        delta_color="inverse"
    )

    selector = st.selectbox("Выберите признак", data.columns)
    st.table(col_df[col_df["col"] == selector])

    st.write("Зависимость переменных")
    val, res = st.columns(2)
    val.metric(
        label="Значение",
        value=(
            round(data[[selector, "SeriousDlqin2yrs"]].corr().values[0, 1], 5)
            if selector not in ["RealEstateLoansOrLines", "GroupAge"] else "Категория"
        )
    )
    res.metric(
        label="Описание",
        value=(
            corr_res(round(data[[selector, "SeriousDlqin2yrs"]].corr().values[0, 1]))
            if selector not in ["RealEstateLoansOrLines", "GroupAge"] else "Нельзя посчитать"
        )
    )

    bar1, bar2, pie1 = st.columns(3)
    with bar1:
        st.markdown("Распределение целевой переменной")
        fig = plt.figure(figsize=(4, 4.15))
        plt.bar(data["SeriousDlqin2yrs"].value_counts().index, data["SeriousDlqin2yrs"].value_counts().values)
        st.pyplot(fig)

    with bar2:
        st.markdown("Распределение пропусков в данных")
        fig = plt.figure(figsize=(4, 7.3))
        plt.barh(data.isna().sum().index, data.isna().sum().values)
        st.pyplot(fig)

    with pie1:
        st.markdown("Распределение типов данных")
        fig = plt.figure(figsize=(3, 4))
        plt.pie(data.dtypes.value_counts().values, labels=data.dtypes.value_counts().index)
        st.pyplot(fig)

    bar4, bar5 = st.columns(2)
    with bar4:
        st.markdown("Распределение возрастных групп")
        fig = plt.figure(figsize=(4, 1))
        plt.barh(data["GroupAge"].value_counts().index, data["GroupAge"].value_counts().values)
        st.pyplot(fig)

    with bar5:
        st.markdown("Распределение кол-ва кредитов")
        fig = plt.figure(figsize=(4, 1))
        plt.barh(data["RealEstateLoansOrLines"].value_counts().index, data["RealEstateLoansOrLines"].value_counts().values)
        st.pyplot(fig)

    st.write("## Воспользуйтесь сервисом")

    radio1, radio2 = st.columns(2)
    with radio1:
        radio1 = st.radio(
            "Возраст",
            ("от 21 до 35", "от 35 до 50", "от 50 до 65", "от 65")
        )

    with radio2:
        radio2 = st.radio(
            "Кол-во кредитов",
            ("до 2", "от 2 до 4", "от 4 до 6", "от 6 до 8", "свыше 8")
        )

    st.write("Два года назад была ли просрочка")
    radio3, radio4, radio5 = st.columns(3)
    with radio3:
        radio3 = st.radio(
            "От 30 до 59 дней",
            ("Да", "Нет")
        )

    with radio4:
        radio4 = st.radio(
            "ОТ 60 до 89 дней",
            ("Да", "Нет")
        )

    with radio5:
        radio5 = st.radio(
            "Свыше 90 дней",
            ("Да", "Нет")
        )

    num1 = st.number_input("Введите общий баланс средств")
    num2 = st.number_input("Введите ежемесячный доход")
    num3 = st.number_input("Введите коэффициент, отражающий месячные расходы делённые на месяные доходы")
    num4 = st.number_input("Введите кол-во открытых кредитных продуктов")
    num5 = st.number_input("Введите кол-во иждивенцев ")


    full_arr = np.asarray([
        num1,
        1 if radio3 == "Да" else 0,
        num3,
        num2,
        num4,
        1 if radio5 == "Да" else 0,
        1 if radio4 == "Да" else 0,
        num5
    ])
    full_arr = np.hstack([full_arr, loans_vectorize(radio2), age_vectorize(radio1)])

    clf = cb.CatBoostClassifier()
    clf.load_model("weights/clf")
    prob = clf.predict_proba(full_arr)[1]
    if (prob >= 0.5) and (prob < 0.75):
        st.warning(f"С веротяностью {round(prob * 100, 2)} у клиента будет  просрочка в 90 или 90+ дней", icon="⚠️")
    elif prob >= 0.75:
        st.error(f"С веротяностью {round(prob * 100, 2)} у клиента будет  просрочка в 90 или 90+ дней", icon="🚨")
    else:
        st.info(f"С веротяностью {round(prob * 100, 2)} у клиента будет  просрочка в 90 или 90+ дней", icon="ℹ️")
