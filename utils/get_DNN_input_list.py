def get_DNN_input_list(run2, dnn_input_variables):
    """Create the DNN input variables"""
    column_list = []
    for x, y in dnn_input_variables.values():
        if run2:
            if ":" in x:
                coll, pos = x.split(":")
                column_list.append(f"{coll}Run2_{y}_{pos}")
            elif x != "events":
                column_list.append(f"{x}Run2_{y}")
            elif "sigma" in y:
                column_list.append(f"{x}_{y}Run2")
            else:
                column_list.append(f"{x}_{y}")
        else:
            if ":" in x:
                coll, pos = x.split(":")
                column_list.append(f"{coll}_{y}_{pos}")
            else:
                column_list.append(f"{x}_{y}")

    return column_list

if __name__ == "__main__":
    from configs.HH4b_common.dnn_input_variables import sig_bkg_dnn_input_variables

    columns = get_DNN_input_list(False, sig_bkg_dnn_input_variables)
    for var in columns:
        print(var)