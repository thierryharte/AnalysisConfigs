def plot_regions_names(cat, namesuffix=""):
    cat_plot_name=cat.replace("Run2", r" $D_{HH}")
    cat_plot_name=cat.replace("_transform", "")
    if "SPREAD" in cat:
        cat_plot_name = (
            cat_plot_name.split("_SPREAD")[0]
        )
    cat_plot_name=cat_plot_name.replace("_", " ")
    cat_plot_name=cat_plot_name.replace("control region", "CR")
    cat_plot_name=cat_plot_name.replace("signal region", "SR")
    
    cat_plot_name=cat_plot_name + namesuffix
    return cat_plot_name
    
#TODO: add a function to plot the columns names
def plot_columns_names():
    return