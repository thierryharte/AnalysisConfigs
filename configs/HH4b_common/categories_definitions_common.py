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
        if "Run2" not in category_name:
            cut_list.append(cuts.blinded)
        else:
            cut_list.append(cuts.blindedRun2)
    
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
            categories_dict |= define_single_category("4b_signal_region" + "_blind") if blind else {}
            categories_dict |= define_single_category("4b_signal_region" )
            categories_dict |= define_single_category("2b_signal_region_preW"+ "_blind") if blind else {}
            categories_dict |= define_single_category("2b_signal_region_preW")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_control_region_postW")
                categories_dict |= define_single_category("2b_signal_region_postW" + "_blind") if blind else {}
                categories_dict |= define_single_category("2b_signal_region_postW" )
        if run2:
            categories_dict |= define_single_category("4b_control_regionRun2")
            categories_dict |= define_single_category("2b_control_region_preWRun2")
            categories_dict |= define_single_category("4b_signal_region" + "_blind"+"Run2") if blind else {}
            categories_dict |= define_single_category("4b_signal_region" +"Run2")
            categories_dict |= define_single_category("2b_signal_region_preW" + "_blind" + "Run2") if blind else {}
            categories_dict |= define_single_category("2b_signal_region_preW"  + "Run2")
            if bkg_morphing_dnn:
                categories_dict |= define_single_category("2b_control_region_postWRun2")
                categories_dict |= define_single_category("2b_signal_region_postW" + "_blind" + "Run2") if blind else {}
                categories_dict |= define_single_category("2b_signal_region_postW"  + "Run2")
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
            