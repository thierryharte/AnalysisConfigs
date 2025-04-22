import configs.HH4b_common.custom_cuts_common as cuts


def define_single_category(category_name):
    """
    Define a single category for the analysis.
    """
    cut_list=[]
    # number of b jets
    if "4b" in category_name:
        cut_list.append(cuts.hh4b_4b_region)
    if "2b" in category_name:
        cut_list.append(cuts.hh4b_2b_region)
        
    # mass cuts
    if "VR1" not in category_name:
        if "control" in category_name:
            if "Run2" not in category_name:
                cut_list.append(cuts.hh4b_control_region)
            else:
                cut_list.append(cuts.hh4b_control_region_run2)
        if "signal" in category_name:
            if "Run2" not in category_name:
                cut_list.append(cuts.hh4b_signal_region)
            else:
                cut_list.append(cuts.hh4b_signal_region_run2)
    if "VR1" in category_name:
        if "control" in category_name:
            if "Run2" not in category_name:
                cut_list.append(cuts.hh4b_VR1_control_region)
            else:
                cut_list.append(cuts.hh4b_VR1_control_region_run2)
        if "signal" in category_name:
            if "Run2" not in category_name:
                cut_list.append(cuts.hh4b_VR1_signal_region)
            else:
                cut_list.append(cuts.hh4b_VR1_signal_region_run2)

    # blind region
    if "blind" in category_name:
        cut_list.append(cuts.blinded)
    
    category_item = {
        category_name: cut_list
    }
    return category_item


def define_categories(bkg_morphing_dnn=False, blind=False, spanet=False, vbf_ggf_dnn=False, run2=False, vr1=False):
    """
    Define the categories for the analysis.
    """
    categories_dict={}
    if not vr1:
        if spanet:
            categories_dict |= define_single_category("4b_control_region")
            categories_dict |= define_single_category("2b_control_region_preW")
            categories_dict |= define_single_category("4b_signal_region" + "_blind" * blind)
            categories_dict |= define_single_category("2b_signal_region_preW"+ "_blind" * blind)
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_control_region_postW")
                categories_dict |= define_single_category("2b_signal_region_postW" + "_blind" * blind)
        if run2:
            categories_dict |= define_single_category("4b_control_regionRun2")
            categories_dict |= define_single_category("2b_control_region_preWRun2")
            categories_dict |= define_single_category("4b_signal_region" + "_blind" * blind+"Run2")
            categories_dict |= define_single_category("2b_signal_region_preW" + "_blind" * blind + "Run2")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_control_region_postWRun2")
                categories_dict |= define_single_category("2b_signal_region_postW" + "_blind" * blind + "Run2")
    else:
        if spanet:
            categories_dict |= define_single_category("4b_VR1_control_region")
            categories_dict |= define_single_category("2b_VR1_control_region_preW")
            categories_dict |= define_single_category("4b_VR1_signal_region")
            categories_dict |= define_single_category("2b_VR1_signal_region_preW")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_VR1_control_region_postW")
                categories_dict |= define_single_category("2b_VR1_signal_region_postW")
        if run2:
            categories_dict |= define_single_category("4b_VR1_control_regionRun2")
            categories_dict |= define_single_category("2b_VR1_control_region_preWRun2")
            categories_dict |= define_single_category("4b_VR1_signal_regionRun2")
            categories_dict |= define_single_category("2b_VR1_signal_region_preWRun2")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_VR1_control_region_postWRun2")
                categories_dict |= define_single_category("2b_VR1_signal_region_postWRun2")
    
    return categories_dict    
            
# print(define_categories(bkg_morphing_dnn=True, blind=True, spanet=True, vbf_ggf_dnn=False, run2=True, vr1=False))
            
    #         categories_dict = categories_dict | {
    #             "4b_control_region": [hh4b_4b_region, hh4b_control_region],
    #             "2b_control_region_preW": [hh4b_2b_region, hh4b_control_region],
    #             "4b_signal_region": [hh4b_4b_region, hh4b_signal_region],
    #             "2b_signal_region_preW": [hh4b_2b_region, hh4b_signal_region],
    #             # "4b_region": [hh4b_4b_region],
    #             # "2b_region": [hh4b_2b_region],
    #         }
    #         if bkg_morphing_dnn:
    #             categories_dict = categories_dict | {
    #             "2b_control_region_postW": [hh4b_2b_region, hh4b_control_region],
    #             "2b_signal_region_postW": [hh4b_2b_region, hh4b_signal_region],
    #             }
    #     if run2:
    #         categories_dictRun2 = {
    #             "4b_control_regionRun2": [hh4b_4b_region, hh4b_control_region_run2],
    #             "2b_control_region_preWRun2": [hh4b_2b_region, hh4b_control_region_run2],
    #             "4b_signal_regionRun2": [hh4b_4b_region, hh4b_signal_region_run2],
    #             "2b_signal_region_preWRun2": [hh4b_2b_region, hh4b_signal_region_run2],
    #         }
    #         if bkg_morphing_dnn:
    #             categories_dictRun2 = categories_dictRun2 | {
    #                 "2b_control_region_postWRun2": [hh4b_2b_region, hh4b_control_region_run2],
    #                 "2b_signal_region_postWRun2": [hh4b_2b_region, hh4b_signal_region_run2],
    #             }
    #         categories_dict = categories_dict | categories_dictRun2
            
            
            
            
            
    # else:
    #     if spanet:
    #         categories_dict = categories_dict | {
    #             "4b_VR1_control_region": [hh4b_4b_region, hh4b_VR1_control_region],
    #             "2b_VR1_control_region_preW": [hh4b_2b_region, hh4b_VR1_control_region],
    #             "4b_VR1_signal_region": [hh4b_4b_region, hh4b_VR1_signal_region],
    #             "2b_VR1_signal_region_preW": [hh4b_2b_region, hh4b_VR1_signal_region],
    #         }
    #         if bkg_morphing_dnn:
    #             categories_dict = categories_dict | {
    #                 "2b_VR1_control_region_postW": [hh4b_2b_region, hh4b_VR1_control_region],
    #                 "2b_VR1_signal_region_postW": [hh4b_2b_region, hh4b_VR1_signal_region],
    #             }
    #     if run2:
    #         categories_dictRun2 = {
    #             "4b_VR1_control_regionRun2": [hh4b_4b_region, hh4b_VR1_control_region_run2],
    #             "2b_VR1_control_region_preWRun2": [
    #                 hh4b_2b_region,
    #                 hh4b_VR1_control_region_run2,
    #             ],
    #             "4b_VR1_signal_regionRun2": [hh4b_4b_region, hh4b_VR1_signal_region_run2],
    #             "2b_VR1_signal_region_preWRun2": [
    #                 hh4b_2b_region,
    #                 hh4b_VR1_signal_region_run2,
    #             ],
    #         }
    #         if bkg_morphing_dnn:
    #             categories_dictRun2 = categories_dictRun2 | {
    #                 "2b_VR1_control_region_postWRun2": [
    #                     hh4b_2b_region,
    #                     hh4b_VR1_control_region_run2,
    #                 ],
    #                 "2b_VR1_signal_region_postWRun2": [
    #                     hh4b_2b_region,
    #                     hh4b_VR1_signal_region_run2,
    #                 ],
    #             }
    #         categories_dict = categories_dict | categories_dictRun2