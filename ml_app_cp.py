import streamlit as st
import numpy as np

# load ML package
import joblib
import os

attribute_info = """
                - Levy
                - Manufacturer
                - Fuel Type
                - Engine Volume
                - Mileage
                - Cylinders
                - Gear Box Type
                - Doors
                - Color
                - Airbags
                - Car Age
                - Leather Interior
                """

df_manufacturer = {
    "SKODA": 46101.0,
    "SSANGYONG": 29997.600490196077,
    "JAGUAR": 27949.64285714286,
    "MASERATI": 26813.5,
    "JEEP": 26112.354838709678,
    "LAND ROVER": 23912.75,
    "HYUNDAI": 21504.12881854987,
    "PORSCHE": 20604.2,
    "MINI": 19118.666666666668,
    "MITSUBISHI": 18910.363636363636,
    "BMW": 18042.659498207886,
    "HONDA": 16123.089514066496,
    "VOLVO": 15994.333333333334,
    "MERCEDES-BENZ": 15543.265938069217,
    "CHEVROLET": 15303.871007371008,
    "KIA": 15153.442567567568,
    "TOYOTA": 15152.942263279445,
    "MERCURY": 14740.0,
    "VOLKSWAGEN": 14598.163090128755,
    "PEUGEOT": 14113.0,
    "FORD": 14040.396061269146,
    "MAZDA": 13887.68,
    "LINCOLN": 13590.0,
    "RENAULT": 13429.8,
    "LEXUS": 13351.2466367713,
    "SUBARU": 13265.32380952381,
    "AUDI": 13180.860215053763,
    "SCION": 12936.5,
    "BUICK": 12845.5,
    "CITROEN": 12795.2,
    "ACURA": 12011.2,
    "NISSAN": 11416.311320754718,
    "INFINITI": 11368.5,
    "FIAT": 11064.973684210527,
    "VAZ": 11051.5,
    "CHRYSLER": 10682.75,
    "OPEL": 9539.412698412698,
    "DODGE": 9039.75,
    "DAEWOO": 6977.444444444444,
    "SUZUKI": 6332.090909090909,
    "GMC": 6085.875,
    "CADILLAC": 4547.5,
}

df_fuel_type = {
    "Diesel": 23474.326286398085,
    "Plug-in Hybrid": 20708.416666666668,
    "Petrol": 16569.341518872214,
    "LPG": 14918.835195530726,
    "Hybrid": 12170.808485562759,
    "CNG": 11850.142857142857,
}

df_gear_box_type = {
    "Tiptronic": 21097.235555555555,
    "Automatic": 17249.66915153158,
    "Variator": 15462.823308270677,
    "Manual": 14657.260683760684,
}

df_doors = {"5": 18867.96153846154, "4": 17665.179342298565, "2": 13559.53125}
df_color = {
    "Brown": 19200.24705882353,
    "Yellow": 18861.79661016949,
    "Grey": 18793.72281275552,
    "White": 18184.712688172043,
    "Carnelian red": 17967.73611111111,
    "Beige": 17798.764705882353,
    "Black": 17406.61740331492,
    "Silver": 17279.314300960512,
    "Blue": 16708.623973727423,
    "Orange": 16391.434782608696,
    "Golden": 16189.148936170213,
    "Red": 15525.904605263158,
    "Sky blue": 15267.021276595744,
    "Purple": 13215.857142857143,
    "Green": 12594.630434782608,
    "Pink": 9968.42857142857,
}

df_leather_interior = {"Yes": True, "No": False}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


# @st.cache_data  # -> this is for local host used
@st.cache  # -> this is for deploy used
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def run_ml_app():
    st.subheader("ML Section")  # <h4> html

    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    manufacturer = st.selectbox(
        "Manufacturer",
        [
            "LEXUS",
            "CHEVROLET",
            "FORD",
            "HONDA",
            "HYUNDAI",
            "TOYOTA",
            "MERCEDES-BENZ",
            "PORSCHE",
            "VOLKSWAGEN",
            "NISSAN",
            "DAEWOO",
            "BMW",
            "SSANGYONG",
            "GMC",
            "KIA",
            "INFINITI",
            "AUDI",
            "OPEL",
            "SUBARU",
            "MITSUBISHI",
            "CITROEN",
            "RENAULT",
            "DODGE",
            "FIAT",
            "MAZDA",
            "JEEP",
            "ACURA",
            "MINI",
            "JAGUAR",
            "BUICK",
            "CHRYSLER",
            "LAND ROVER",
            "SUZUKI",
            "LINCOLN",
            "CADILLAC",
            "VAZ",
            "MASERATI",
            "SKODA",
            "VOLVO",
            "MERCURY",
            "SCION",
            "PEUGEOT",
        ],
    )
    fuel_type = st.selectbox(
        "Fuel Type",
        ["Hybrid", "Petrol", "Diesel", "Plug-in Hybrid", "LPG", "CNG"],
    )

    gear_box_type = st.selectbox(
        "Gear Box Type", ["Automatic", "Tiptronic", "Manual", "Variator"]
    )
    doors = st.selectbox("Doors", ["4", "5", "2"])
    color = st.selectbox(
        "Color",
        [
            "Silver",
            "Black",
            "White",
            "Blue",
            "Grey",
            "Sky blue",
            "Red",
            "Green",
            "Yellow",
            "Beige",
            "Orange",
            "Brown",
            "Carnelian red",
            "Golden",
            "Purple",
            "Pink",
        ],
    )
    leather_interior = st.selectbox("Leather Interior", ["Yes", "No"])

    levy = st.number_input("Levy", 1, 2000)
    engine_volume = st.number_input("Engine Volume", 1, 4)
    mileage = st.number_input("Mileage", 10, 350500)
    cylinders = st.number_input("Cylinders", 2, 16)
    airbags = st.number_input("Airbags", 0, 16)
    car_age = st.number_input("Car Age", 4, 24)

    with st.expander("Your Selected Options"):
        result = {
            "manufacturer": manufacturer,
            "fuel_type": fuel_type,
            "gear_box_type": gear_box_type,
            "doors": doors,
            "color": color,
            "leather_interior": leather_interior,
            "levy": levy,
            "engine_volume": engine_volume,
            "mileage": mileage,
            "cylinders": cylinders,
            "airbags": airbags,
            "car_age": car_age,
        }

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in [
            "LEXUS",
            "CHEVROLET",
            "FORD",
            "HONDA",
            "HYUNDAI",
            "TOYOTA",
            "MERCEDES-BENZ",
            "PORSCHE",
            "VOLKSWAGEN",
            "NISSAN",
            "DAEWOO",
            "BMW",
            "SSANGYONG",
            "GMC",
            "KIA",
            "INFINITI",
            "AUDI",
            "OPEL",
            "SUBARU",
            "MITSUBISHI",
            "CITROEN",
            "RENAULT",
            "DODGE",
            "FIAT",
            "MAZDA",
            "JEEP",
            "ACURA",
            "MINI",
            "JAGUAR",
            "BUICK",
            "CHRYSLER",
            "LAND ROVER",
            "SUZUKI",
            "LINCOLN",
            "CADILLAC",
            "VAZ",
            "MASERATI",
            "SKODA",
            "VOLVO",
            "MERCURY",
            "SCION",
            "PEUGEOT",
        ]:
            res = get_value(i, df_manufacturer)
            encoded_result.append(res)
        elif i in ["Hybrid", "Petrol", "Diesel", "Plug-in Hybrid", "LPG", "CNG"]:
            res = get_value(i, df_fuel_type)
            encoded_result.append(res)

        elif i in ["Automatic", "Tiptronic", "Manual", "Variator"]:
            res = get_value(i, df_gear_box_type)
            encoded_result.append(res)

        elif i in ["4", "5", "2"]:
            res = get_value(i, df_doors)
            encoded_result.append(res)

        elif i in [
            "Silver",
            "Black",
            "White",
            "Blue",
            "Grey",
            "Sky blue",
            "Red",
            "Green",
            "Yellow",
            "Beige",
            "Orange",
            "Brown",
            "Carnelian red",
            "Golden",
            "Purple",
            "Pink",
        ]:
            res = get_value(i, df_color)
            encoded_result.append(res)

        elif i in ["Yes", "No"]:
            res = get_value(i, df_leather_interior)
            encoded_result.append(res)

    ## prediction section
    st.subheader("Car Prediction Result")

    # Decode
    single_sample = np.array(encoded_result).reshape(1, -1)
    st.write(single_sample)

    model = load_model("RF_model.pkl")

    prediction = model.predict(single_sample)
    # pred_proba = model.predict_proba(single_sample)
    st.write(prediction)
    # st.write(pred_proba)
    pred_score = {"Car Price": round(prediction[0], 3)}
    st.write(pred_score)
