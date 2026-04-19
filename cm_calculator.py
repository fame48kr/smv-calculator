"""CM (공임) 계산 로직 - ERP 품셈표 logic.xlsx 기반"""

FACTORIES = {
    'VINA KOREA':           {'code':'00001','country':'VN','BEP_AMT':1139.14,'FAC_WORKHOUR':570,'FAC_SEWER':25.49,'FAC_ETC':21.53,'E_FAC_EFFC':0.68,'E_FAC_EFFC_MAX':0.74,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':2.55,'E_HUM_MNG_AMT':4.85},
    'YAKJIN VIETNAM':       {'code':'00003','country':'VN','BEP_AMT':1041.12,'FAC_WORKHOUR':570,'FAC_SEWER':26.74,'FAC_ETC':19.31,'E_FAC_EFFC':0.64,'E_FAC_EFFC_MAX':0.74,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':2.38,'E_HUM_MNG_AMT':4.20},
    'MICHIGAN HAIDUONG':    {'code':'00007','country':'VN','BEP_AMT':1230.89,'FAC_WORKHOUR':570,'FAC_SEWER':27.57,'FAC_ETC':20.24,'E_FAC_EFFC':0.70,'E_FAC_EFFC_MAX':0.74,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':2.71,'E_HUM_MNG_AMT':4.90},
    'YAKJIN SAIGON':        {'code':'00010','country':'VN','BEP_AMT':1639.54,'FAC_WORKHOUR':540,'FAC_SEWER':28.34,'FAC_ETC':17.20,'E_FAC_EFFC':0.68,'E_FAC_EFFC_MAX':0.74,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':4.00,'E_HUM_MNG_AMT':6.43},
    'JS VINA LTD':          {'code':'00050','country':'VN','BEP_AMT':1247.59,'FAC_WORKHOUR':570,'FAC_SEWER':31.27,'FAC_ETC':20.23,'E_FAC_EFFC':0.60,'E_FAC_EFFC_MAX':0.74,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':2.55,'E_HUM_MNG_AMT':4.20},
    'YAKJIN CAMBODIA':      {'code':'00004','country':'KH','BEP_AMT':1470.12,'FAC_WORKHOUR':570,'FAC_SEWER':35.00,'FAC_ETC':23.40,'E_FAC_EFFC':0.63,'E_FAC_EFFC_MAX':0.74,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':2.65,'E_HUM_MNG_AMT':4.62},
    'YTC CORPORATION':      {'code':'00009','country':'KH','BEP_AMT':1465.06,'FAC_WORKHOUR':570,'FAC_SEWER':30.79,'FAC_ETC':25.29,'E_FAC_EFFC':0.66,'E_FAC_EFFC_MAX':0.74,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':2.75,'E_HUM_MNG_AMT':5.62},
    'YAKJIN JAYA':          {'code':'00006','country':'ID','BEP_AMT':1177.25,'FAC_WORKHOUR':540,'FAC_SEWER':32.10,'FAC_ETC':18.60,'E_FAC_EFFC':0.65,'E_FAC_EFFC_MAX':0.778,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':2.60,'E_HUM_MNG_AMT':4.30},
    'PT. YAKJIN JAYA 2':    {'code':'00143','country':'ID','BEP_AMT':1420.85,'FAC_WORKHOUR':540,'FAC_SEWER':37.89,'FAC_ETC':22.83,'E_FAC_EFFC':0.64,'E_FAC_EFFC_MAX':0.74,'E_FAC_EFFC_ADD':0.03,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':2.60,'E_HUM_MNG_AMT':4.45},
    'PT. SEMARANG GARMENT': {'code':'00154','country':'ID','BEP_AMT':1463.21,'FAC_WORKHOUR':540,'FAC_SEWER':44.14,'FAC_ETC':33.28,'E_FAC_EFFC':0.63,'E_FAC_EFFC_MAX':0.70,'E_FAC_EFFC_ADD':0.00,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.00,'E_HUM_SEW_AMT':2.10,'E_HUM_MNG_AMT':3.68},
    'PT. BATANG APPAREL':   {'code':'00155','country':'ID','BEP_AMT':1428.60,'FAC_WORKHOUR':540,'FAC_SEWER':44.80,'FAC_ETC':34.57,'E_FAC_EFFC':0.60,'E_FAC_EFFC_MAX':0.70,'E_FAC_EFFC_ADD':0.00,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.00,'E_HUM_SEW_AMT':2.00,'E_HUM_MNG_AMT':3.54},
    'YAKJIN GUATEMALA':     {'code':'00070','country':'GT','BEP_AMT':2429.12,'FAC_WORKHOUR':540,'FAC_SEWER':34.74,'FAC_ETC':20.91,'E_FAC_EFFC':0.70,'E_FAC_EFFC_MAX':0.77,'E_FAC_EFFC_ADD':0.00,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':4.85,'E_HUM_MNG_AMT':8.45},
    'SUB.CON GUATEMALA':    {'code':'00060','country':'GT','BEP_AMT':2429.12,'FAC_WORKHOUR':540,'FAC_SEWER':34.74,'FAC_ETC':20.91,'E_FAC_EFFC':0.70,'E_FAC_EFFC_MAX':0.77,'E_FAC_EFFC_ADD':0.00,'E_FAC_EFFC_ADD_STD':6.5,'FAC_INLAND':0.09,'E_HUM_SEW_AMT':4.85,'E_HUM_MNG_AMT':8.45},
}

COUNTRY_FLAGS = {'VN': '🇻🇳 Vietnam', 'KH': '🇰🇭 Cambodia', 'ID': '🇮🇩 Indonesia', 'GT': '🇬🇹 Guatemala'}

WASH_OPTIONS = {
    'NO G/W & G/D': {'add_wash': 0,    'add_cost': 0.007},
    'Normal G/W':   {'add_wash': 5,    'add_cost': 0.012},
    'Special G/W':  {'add_wash': 8,    'add_cost': 0.012},
    'Normal G/D':   {'add_wash': 15,   'add_cost': 0.037},
    'Special G/D':  {'add_wash': 18,   'add_cost': 0.037},
}

RAMP_UP_RATES = {
    1:    [0.30, 0.60, 0.75, 0.85, 1.0],
    5:    [0.40, 0.65, 0.80, 0.95, 1.0],
    10:   [0.45, 0.70, 0.85, 1.00, 1.0],
    20:   [0.50, 0.75, 0.90, 1.00, 1.0],
    40:   [0.55, 0.80, 0.95, 1.00, 1.0],
}

# DS_SMV011 봉제 LOSS율 (수량구간별)
LOSS_SEW_TABLE = [(500,0.13),(3000,0.06),(10000,0.024),(30000,0.014),(50000,0.007),(float('inf'),0.005)]
LOSS_WASH_TABLE = [(500,0.13),(3000,0.06),(10000,0.024),(30000,0.014),(50000,0.007),(float('inf'),0.005)]  # Spec G/W
LOSS_DYE_TABLE =  [(500,0.10),(3000,0.045),(10000,0.023),(30000,0.013),(50000,0.007),(float('inf'),0.004)]

# DS_SMV007 수량 가중치
QTY_WEIGHT_TABLE = [(500,0.4),(1000,0.55),(3000,0.7),(5000,0.8),(10000,0.9),(30000,1.0),(50000,1.0),(100000,1.05),(300000,1.08),(float('inf'),1.1)]


def _lookup(table, qty):
    for threshold, val in table:
        if qty <= threshold:
            return val
    return table[-1][1]


def get_ramp_rates(lines):
    for key in sorted(RAMP_UP_RATES.keys()):
        if lines <= key:
            return RAMP_UP_RATES[key]
    return RAMP_UP_RATES[40]


def calc_loss_total(qty, has_grp=False, has_emb=False, wash_option='NO G/W & G/D'):
    loss_sew = _lookup(LOSS_SEW_TABLE, qty)
    loss_grp = 0.02 if has_grp else 0
    loss_emb = 0.02 if has_emb else 0
    if wash_option in ('Normal G/W', 'Special G/W'):
        loss_wash = _lookup(LOSS_WASH_TABLE, qty)
    else:
        loss_wash = 0
    if wash_option in ('Normal G/D', 'Special G/D'):
        loss_dye = _lookup(LOSS_DYE_TABLE, qty)
    else:
        loss_dye = 0
    total = ((1 + loss_sew) * (1 + loss_grp) * (1 + loss_emb) * (1 + loss_wash) * (1 + loss_dye) - 1)
    return {'sew': loss_sew, 'grp': loss_grp, 'emb': loss_emb, 'wash': loss_wash, 'dye': loss_dye, 'total': total}


def calc_fac_effc(factory, tot_smv, qty):
    base = factory['E_FAC_EFFC']
    smv_add = factory['E_FAC_EFFC_ADD'] if tot_smv >= factory['E_FAC_EFFC_ADD_STD'] else 0
    qty_weight = _lookup(QTY_WEIGHT_TABLE, qty)
    return min((base + smv_add) * qty_weight, factory['E_FAC_EFFC_MAX'])


def calc_dt_working(daily_prod, qty_order, lines):
    rates = get_ramp_rates(lines)
    ramp_pcs = [daily_prod * r for r in rates[:4]]
    ramp_total = sum(ramp_pcs)
    if ramp_total >= qty_order:
        cumulative = 0
        for i, pcs in enumerate(ramp_pcs):
            if cumulative + pcs >= qty_order:
                frac = (qty_order - cumulative) / pcs
                return i + frac
            cumulative += pcs
    return 4 + max(0, qty_order - ramp_total) / daily_prod


def calculate_cm(factory_name, tot_smv, qty_ord, lines=1, wash_option='NO G/W & G/D', has_grp=False, has_emb=False):
    fac = FACTORIES[factory_name]
    wash = WASH_OPTIONS[wash_option]
    add_wash_pct = wash['add_wash']

    loss = calc_loss_total(qty_ord, has_grp, has_emb, wash_option)
    qty_order = qty_ord * (1 + loss['total'])

    fac_effc = calc_fac_effc(fac, tot_smv, qty_ord)
    daily_prod = (fac['FAC_WORKHOUR'] * fac['FAC_SEWER'] * fac_effc) / tot_smv
    dt_working = calc_dt_working(daily_prod, qty_order, lines)

    wash_factor = 1 + add_wash_pct * 0.01
    net_cm = fac['BEP_AMT'] * wash_factor / daily_prod - fac['FAC_INLAND']
    working_cm = dt_working * fac['BEP_AMT'] * wash_factor / qty_ord - fac['FAC_INLAND']

    return {
        'factory': factory_name,
        'FAC_EFFC': fac_effc,
        'DAILY_PROD': daily_prod,
        'DT_WORKING': dt_working,
        'LOSS': loss,
        'QTY_ORDER': qty_order,
        'NET_CM': net_cm,
        'WORKING_CM': working_cm,
        'BEP_AMT': fac['BEP_AMT'],
        'FAC_INLAND': fac['FAC_INLAND'],
    }
