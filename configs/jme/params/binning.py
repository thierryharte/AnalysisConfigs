# bins = {
#     "ElectronGood_pt" : {
#         '2016_PreVFP' : [29, 40, 50, 60, 70, 80, 100, 130, 500],
#         '2016_PostVFP': [29, 40, 50, 60, 70, 80, 100, 130, 500],
#         '2017': [30, 35, 40, 50, 60, 70, 80, 90, 100, 130, 200, 500],
#         '2018': [30, 35, 40, 50, 60, 70, 80, 90, 100, 130, 200, 500],
#     },
#     "ElectronGood_etaSC" : {
#         '2016_PreVFP' : [-2.5, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.5],
#         '2016_PostVFP': [-2.5, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.5],
#         '2017': [-2.5, -2.0, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.0, 2.5],
#         '2018': [-2.5, -2.0, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.0, 2.5],
#     }
# }


# eta_bins = [
#     0.000,
#     0.087,
#     # 0.174,
#     # 0.261,
#     # 0.348,
#     # 0.435,
#     # 0.522,
#     # 0.609,
#     # 0.696,
#     # 0.783,
#     # 0.879,
#     # 0.957,
#     # 1.044,
#     # 1.131,
#     # 1.218,
#     # 1.305,
#     # 1.392,
#     # 1.479,
#     # 1.566,
#     # 1.653,
#     # 1.740,
#     # 1.830,
#     # 1.930,
#     # 2.043,
#     # 2.172,
#     # 2.322,
#     # 2.500,
#     # 2.650,
#     # 2.853,
#     # 2.964,
#     # 3.139,
#     # 3.314,
#     # 3.489,
#     # 3.664,
#     # 3.839,
#     # 4.013,
#     # 4.191,
#     # 4.363,
#     # 4.538,
#     # 4.716,
#     # 4.889,
#     # 5.191,
# ]

# # concatenate the eta bins with the opposite values
# eta_bins = [-i for i in eta_bins[::-1] if i != 0.0] + eta_bins


eta_bins = [
    # -5.191,
    # -4.889,
    # -4.716,
    # -4.538,
    # -4.363,
    # -4.191,
    # -4.013,
    # -3.839,
    # -3.664,
    # -3.489,
    # -3.314,
    # -3.139,
    # -2.964,
    # -2.853,
    # -2.65,
    # -2.5,
    # -2.322,
    # -2.172,
    # -2.043,
    # -1.93,
    # -1.83,
    # -1.74,
    # -1.653,
    # -1.566,
    # -1.479,
    # -1.392,
    # -1.305,
    # -1.218,
    # -1.131,
    # -1.044,
    # -0.957,
    # -0.879,
    # -0.783,
    # -0.696,
    # -0.609,
    # -0.522,
    # -0.435,
    # -0.348,
    # -0.261,
    # -0.174,
    # -0.087,
    0.0,
    0.087,
    # 0.174,
    # 0.261,
    # 0.348,
    # 0.435,
    # 0.522,
    # 0.609,
    # 0.696,
    # 0.783,
    # 0.879,
    # 0.957,
    # 1.044,
    # 1.131,
    # 1.218,
    # 1.305,
    # 1.392,
    # 1.479,
    # 1.566,
    # 1.653,
    # 1.74,
    # 1.83,
    # 1.93,
    # 2.043,
    # 2.172,
    # 2.322,
    # 2.5,
    # 2.65,
    # 2.853,
    # 2.964,
    # 3.139,
    # 3.314,
    # 3.489,
    # 3.664,
    # 3.839,
    # 4.013,
    # 4.191,
    # 4.363,
    # 4.538,
    # 4.716,
    # 4.889,
    # 5.191,
]

pt_bins = [
    15.0,
    17.0,
    20.0,
    23.0,
    27.0,
    30.0,
    35.0,
    40.0,
    45.0,
    57.0,
    72.0,
    90.0,
    120.0,
    150.0,
    200.0,
    300.0,
    400.0,
    550.0,
    750.0,
    1000.0,
    1500.0,
    2000.0,
    2500.0,
    3000.0,
    3500.0,
    4000.0,
    4500.0,
    5000.0,
]
