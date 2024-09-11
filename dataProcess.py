def data_adjust(data):
    data["Star color"] = data["Star color"].str.lower().str.replace(" ", "").str.replace("-", "")
    data["Star color"] = data["Star color"].str.replace("yellowishwhite", "yellowwhite").str.replace("whiteyellow", "yellowwhite").str.replace(
        "whitish", "white")
    return data