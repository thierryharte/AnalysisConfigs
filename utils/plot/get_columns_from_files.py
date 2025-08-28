from coffea.util import load
import numpy as np

def get_columns_from_files(inputfiles, filter_lambda=None, debug=False):
    cat_col = {}
    total_datasets_list = []
    # get the columns
    for inputfile in inputfiles:
        accumulator = load(inputfile)
        samples = list(accumulator["columns"].keys())
        if debug: print(f"inputfile {inputfile}")
        for sample in samples:
            if debug: print(f"sample {sample}")
            datasets = list(accumulator["columns"][sample].keys())
            for dataset in datasets:
                if dataset not in total_datasets_list:
                    total_datasets_list.append(dataset)
                if debug: print(f"dataset {dataset}")
                categories = list(accumulator["columns"][sample][dataset].keys())
                for category in categories:
                    if debug: print(f"category {category}")
                    if category not in cat_col:
                        cat_col[category] = {}
                    columns = list(
                        accumulator["columns"][sample][dataset][category].keys()
                    )
                    for i, column in enumerate(columns):
                        #filter with lamda function
                        if filter_lambda is not None:
                            if not filter_lambda(column):
                                if debug: print(f"Skipping column {column} due to filter")
                                continue
                        column_array = accumulator["columns"][sample][dataset][
                            category
                        ][column].value
                        
                        if column == "weight" and dataset  in accumulator["sum_genweights"]:
                            column_array = column_array / accumulator["sum_genweights"][dataset]
                        
                        if column not in cat_col[category]:
                            cat_col[category][column] = column_array
                        else:
                            cat_col[category][column] = np.concatenate(
                                (cat_col[category][column], column_array)
                            )
                        
                        if i == 0:
                            if debug: print(
                                f"column {column}",
                                column_array.shape,
                                cat_col[category][column].shape,
                            )
                
    return cat_col, total_datasets_list
