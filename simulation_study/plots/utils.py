def get_right_palette_for_subset(cols):
    palette_dict = {"SCOPE": "royalblue",
                    "EM": "orange",
                    "0": "darkgreen",
                    "KP, EM": "firebrick",
                    "MCMC": "m"}
    palette_return = []
    keywords = list(palette_dict.keys())
    for col in cols:
        for k in keywords:
            if k in col:
                palette_return.append(palette_dict[k])
    return palette_return