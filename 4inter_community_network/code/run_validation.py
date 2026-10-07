"""Reproducible, local-only medical identification validation on unchanged inputs.

Run with the existing jupyter Python environment (geopandas/scipy/pyarrow).
Stages: prepare, extract, analyze, report, all. Outputs never replace source files.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr, pearsonr
from rules import PURPOSES, normalize_name, time_flags, eligible, distinct_weekdays, split_neighborhood_flow

ROOT = Path(r'D:\多功能网络构建')
OUT = Path(__file__).resolve().parents[1]
RESULT = OUT / 'results'
REPORT = OUT / 'reports'
CACHE = OUT / 'local_records'
RAW = ROOT / '2018年11月非通勤出行/深圳2018年11月常住人口非通勤出行数据.csv'
EXTERNAL = Path(r'D:\download\全市各医院年度门诊工作量（医院数据包含下属社康）_2920001503656.csv')
POI = ROOT / '代码/出行轨迹/result/grid_poi归一化.csv'
PRIOR = ROOT / '代码/出行轨迹/result/时间分布概率_归一化平滑.csv'
GRID = ROOT / 'SZ_grid_500m/SZ_grid_500m/SZ_grid_500m.shp'
HOSP = ROOT / 'poi/poi/medical/医院_床位.shp'
CHC = ROOT / 'poi/poi/medical/社康完整版.shp'
CENTERS = ROOT / '等时圈/等时圈wgs84拆分/Shenzhen_人口重心/深圳社区_联通人口重心result.shp'
COMM = ROOT / 'shp/深圳社区objectid.shp'
ISO = ROOT / '等时圈/等时圈wgs84拆分/等时圈wgs84拆分/foot等时圈.shp'
POP = ROOT / '2018年11月非通勤出行/网格_人口_社区/网格_人口_社区/深圳各社区人口.shp'
HOME = ROOT / '2018年11月非通勤出行/用户居住地结果.csv'
ATTR = ROOT / '2018年11月非通勤出行/个体属性数据.csv'

def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x), allow_nan=False), encoding='utf-8')

def save(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')

def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {'n': len(x), 'spearman': None, 'pearson': None}
    return {'n': len(x), 'spearman': float(spearmanr(x, y).statistic), 'pearson': float(pearsonr(x, y).statistic)}

def file_manifest(paths):
    rows = []
    for p in paths:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        rows.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()})
    return rows

def prepare():
    ext = pd.read_csv(EXTERNAL, dtype={'组织机构代码': str})
    metrics = ['门诊人次', '急诊人次', '总诊疗人次']
    for col in metrics:
        ext[col] = pd.to_numeric(ext[col], errors='raise')
        assert (ext[col] >= 0).all() and ext[col].notna().all()
    key = ['年度', '组织机构代码']
    conflict = ext.groupby(key)[['机构名称'] + metrics].nunique().gt(1).any(axis=1)
    assert not conflict.any(), 'Conflicting annual observations require source reconciliation.'
    unique = ext.sort_values('地址', na_position='last').drop_duplicates(key).copy()
    assert set(unique['年度']) == {2018}
    unique['match_name'] = unique['机构名称'].map(normalize_name)
    assert not unique['match_name'].duplicated().any()
    save(unique, RESULT / 'external_2018_deduplicated.csv')
    dump(REPORT / 'external_input_audit.json', {
        'raw_rows': len(ext), 'unique_institutions': len(unique), 'duplicate_rows_removed': len(ext)-len(unique),
        'duplicate_metric_conflicts': int(conflict.sum()), 'missing_address_rows': int(ext['地址'].isna().sum()),
        'outpatient_total_unique': int(unique['门诊人次'].sum()),
        'total_not_outpatient_plus_emergency': int((unique['总诊疗人次'] != unique['门诊人次']+unique['急诊人次']).sum()),
        'comparison': 'November identified visits vs annual 2018 outpatient rank/share; no absolute scaling',
        'parent_scope': 'hospital data includes subordinate CHCs'})
    grid = gpd.read_file(GRID).to_crs(32650)
    assert grid.Tid.is_unique
    grid.geometry = grid.geometry.centroid
    grid = grid[['Tid', 'geometry']]
    hospital = gpd.read_file(HOSP).to_crs(32650)
    chc = gpd.read_file(CHC).to_crs(32650)
    assert hospital.geometry.is_valid.all() and chc.geometry.is_valid.all()
    # Collapse only exact duplicate identifiers/geometries, never fuzzy branches.
    hospital = hospital.drop_duplicates('Uid').reset_index(drop=True)
    hospital['facility_id'] = 'H:' + hospital.Uid.astype(str)
    hospital['facility_name'] = hospital['Name']
    hospital['parent_name'] = hospital['Name']
    hospital['kind'] = 'hospital'
    chc['facility_id'] = ['C:' + str(i) for i in range(len(chc))]
    chc['facility_name'] = chc['社康机']
    chc['parent_name'] = chc['上级举']
    chc['kind'] = 'CHC'
    facilities = gpd.GeoDataFrame(pd.concat([hospital[['facility_id','facility_name','parent_name','kind','geometry']],
                                              chc[['facility_id','facility_name','parent_name','kind','geometry']]], ignore_index=True), crs=32650)
    lookup = unique.set_index('match_name')['组织机构代码']
    facilities['match_name'] = facilities.parent_name.map(normalize_name)
    facilities['external_code'] = facilities.match_name.map(lookup)
    save(pd.DataFrame(facilities.drop(columns='geometry')), RESULT / 'facility_parent_crosswalk.csv')
    # No bed/CHC weight mixing, and no unsupported gravity exponent.
    buffered = facilities[['facility_id','geometry']].copy()
    buffered.geometry = buffered.geometry.buffer(500)
    candidates = gpd.sjoin(grid, buffered, how='inner', predicate='intersects')
    candidates['distance_m'] = [geom.distance(facilities.geometry.iloc[int(i)]) for geom,i in zip(candidates.geometry,candidates.index_right)]
    candidates = pd.DataFrame(candidates[['Tid','facility_id','distance_m']]).drop_duplicates(['Tid','facility_id'])
    candidates = candidates.merge(facilities.drop(columns='geometry'), on='facility_id', validate='many_to_one')
    candidates.to_parquet(CACHE / 'candidates.parquet', index=False)
    save(candidates, RESULT / 'grid_facility_candidates_500m.csv')
    centers = gpd.read_file(CENTERS)
    assert centers.OBJECTID.is_unique and centers.SQCODE.is_unique
    nodes = centers[['OBJECTID','SQCODE','SQNAME']].sort_values('OBJECTID')
    save(nodes, RESULT / 'node_order.csv')
    comm = gpd.read_file(COMM).to_crs(32650)
    comm = comm[['SQCODE','geometry']].merge(nodes[['SQCODE','OBJECTID']], on='SQCODE', validate='one_to_one')
    joined = gpd.sjoin(grid, comm[['OBJECTID','geometry']], how='inner', predicate='intersects')
    multiplicity = joined.groupby('Tid').OBJECTID.nunique()
    ambiguous = multiplicity[multiplicity > 1].index
    # Administrative boundary ambiguity is reported/excluded, not arbitrarily assigned.
    joined = joined[~joined.Tid.isin(ambiguous)].drop_duplicates('Tid')
    save(pd.DataFrame(joined[['Tid','OBJECTID']]), RESULT / 'grid_community.csv')
    iso = gpd.read_file(ISO).to_crs(32650)
    assert iso.objectid.is_unique and set(iso.objectid).issubset(set(nodes.OBJECTID))
    target = gpd.sjoin(grid, iso[['objectid','geometry']], how='inner', predicate='intersects')
    save(pd.DataFrame(target[['Tid','objectid']]).drop_duplicates(), RESULT / 'grid_neighborhood.csv')
    # Facility representative-point membership for location-ambiguity audits only.
    fp = facilities[['facility_id','geometry']].copy()
    fp.geometry = fp.geometry.representative_point()
    fm = gpd.sjoin(fp, iso[['objectid','geometry']], how='inner', predicate='intersects')
    save(pd.DataFrame(fm[['facility_id','objectid']]).drop_duplicates(), RESULT / 'facility_neighborhood.csv')
    population = gpd.read_file(POP).drop(columns=['geometry','OBJECTID'])
    population = nodes.merge(population, on='SQCODE', validate='one_to_one', suffixes=('','_population'))
    save(population, RESULT / 'population_crosswalk.csv')
    dump(REPORT / 'spatial_input_audit.json', {
        'grid_count':len(grid), 'hospital_AOIs':len(hospital), 'CHC_points':len(chc),
        'ambiguous_admin_grid_ids': [int(x) for x in ambiguous], 'canonical_nodes':len(nodes),
        'missing_neighborhood_ids': sorted(set(nodes.OBJECTID)-set(iso.objectid)),
        'matched_hospital_AOIs': int(facilities.query("kind == 'hospital'").external_code.notna().sum()),
        'matched_CHCs':int(facilities.query("kind == 'CHC'").external_code.notna().sum()),
        'coordinates':'500m grid centroid proxy; metric operations EPSG:32650',
        'matching':'NFKC/whitespace normalization, exact names only; no automatic fuzzy aliases'})
    manifest_paths = [RAW, EXTERNAL, POI, PRIOR, HOME, ATTR]
    for p in [GRID,HOSP,CHC,CENTERS,COMM,ISO,POP]:
        manifest_paths.extend(sorted(p.parent.glob(p.stem + '.*')))
    dump(REPORT / 'source_manifest.json', file_manifest(manifest_paths))
    print('prepare completed', flush=True)

def classifier_tables():
    poi = pd.read_csv(POI).set_index('Tid')
    assert poi.index.is_unique
    assert np.isfinite(poi[PURPOSES].values).all() and (poi[PURPOSES].values >= 0).all()
    prior = pd.read_csv(PRIOR, encoding='gb18030')
    piv = prior.pivot(index=['日期属性','发布时刻（小时）'], columns='地点大类功能', values='W_平滑')[PURPOSES]
    assert len(piv) == 48 and piv.notna().all().all() and (piv.values >= 0).all()
    labels_table = np.zeros((2,24,int(poi.index.max())+1),dtype=bool)
    scores_table = np.zeros(labels_table.shape,dtype=float)
    for weekend in (False,True):
        for hour in range(24):
            weights = poi[PURPOSES].values * piv.loc[('休息日' if weekend else '工作日', hour)].values
            totals = weights.sum(axis=1)
            maxima = weights.max(axis=1)
            medical = (weights.argmax(axis=1) == PURPOSES.index('medical')) & (maxima > 0)
            relative = np.divide(weights[:,2], totals, out=np.zeros_like(totals), where=totals>0)
            labels_table[int(weekend),hour,poi.index.values] = medical
            scores_table[int(weekend),hour,poi.index.values] = relative
    return labels_table,scores_table

def extract(chunksize=500000):
    tables = classifier_tables()
    cand = pd.read_parquet(CACHE / 'candidates.parquet')
    # Frequent presence uses unambiguous facility/100m proxy; employment uses hospitals.
    h100 = cand[(cand.kind == 'hospital') & (cand.distance_m <= 100)]
    f100 = cand[cand.distance_m <= 100]
    one = f100.groupby('Tid').facility_id.nunique()
    unique_hospital = f100[f100.Tid.isin(one[one==1].index)].set_index('Tid').facility_id
    hosp_grids = set(h100.Tid)
    grid_ids = set(pd.read_csv(POI).Tid)
    counters = Counter()
    age = Counter()
    work_parts, presence_parts = [], []
    usecols=['nid','home_tid','start_tid','start_ptype','end_tid','end_ptype','stime','etime','date','age']
    numeric = ['nid','home_tid','start_tid','end_tid','start_ptype','end_ptype','date']
    source_offset = 0
    for number, raw in enumerate(pd.read_csv(RAW, usecols=usecols, chunksize=chunksize)):
        raw['source_row'] = np.arange(source_offset, source_offset+len(raw), dtype=np.int64)
        source_offset += len(raw)
        counters['raw_rows'] += len(raw)
        for col in numeric:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
        valid_ids = raw[numeric].notna().all(axis=1) & raw[numeric].mod(1).eq(0).all(axis=1)
        counters['invalid_ID_or_type_rows'] += int((~valid_ids).sum())
        raw = raw.loc[valid_ids].copy()
        raw[numeric] = raw[numeric].astype('int64')
        a = pd.to_datetime(raw.stime, errors='coerce')
        flags = time_flags(raw.stime, raw.etime)
        in_month = a.ge(pd.Timestamp('2018-11-01')) & a.lt(pd.Timestamp('2018-12-01'))
        counters['invalid_or_outside_arrival_month'] += int((~in_month).sum())
        counters['invalid_time_rows'] += int(flags.invalid_time.sum())
        counters['missing_destination_grid_rows'] += int((~raw.end_tid.isin(grid_ids)).sum())
        counters['start_home_type_inconsistent'] += int(((raw.start_ptype==1)&raw.start_tid.ne(raw.home_tid)).sum())
        for t,v in raw.start_ptype.value_counts().items():counters['start_ptype_'+str(t)] += int(v)
        for t,v in raw.end_ptype.value_counts().items():counters['end_ptype_'+str(t)] += int(v)
        for prefix in ('start','end'):
            w = raw.loc[raw[prefix+'_ptype'].eq(2), ['nid',prefix+'_tid']].rename(columns={prefix+'_tid':'work_tid'}).drop_duplicates()
            if len(w):work_parts.append(w)
        presence = raw.loc[in_month & flags.weekday & ~flags.invalid_time, ['nid','end_tid']].copy()
        presence['facility_id'] = presence.end_tid.map(unique_hospital)
        presence['day'] = flags.loc[presence.index, 'day']
        presence = presence.dropna(subset=['facility_id'])[['nid','facility_id','day']].drop_duplicates()
        if len(presence):presence_parts.append(presence)
        med = pd.Series(False, index=raw.index)
        relative = pd.Series(0.0, index=raw.index)
        sel = in_month & raw.end_tid.ge(0) & raw.end_tid.lt(tables[0].shape[2])
        ti = raw.loc[sel,'end_tid'].to_numpy()
        hi = a.loc[sel].dt.hour.to_numpy(dtype=int)
        wi = a.loc[sel].dt.dayofweek.ge(5).to_numpy(dtype=int)
        med.loc[sel] = tables[0][wi,hi,ti]
        relative.loc[sel] = tables[1][wi,hi,ti]
        selected = raw.loc[med].copy()
        selected['medical_relative_score'] = relative.loc[med]
        selected = pd.concat([selected, flags.loc[med]], axis=1)
        selected.to_parquet(CACHE / ('medical_%04d.parquet' % number), index=False)
        counters['medical_argmax_rows'] += len(selected)
        counters['medical_invalid_time_rows'] += int(selected.invalid_time.sum())
        age.update(selected.age.fillna('unknown').astype(str))
        if number % 5 == 0:print('extract rows',source_offset,'medical',counters['medical_argmax_rows'],flush=True)
    work = pd.concat(work_parts, ignore_index=True).drop_duplicates() if work_parts else pd.DataFrame(columns=['nid','work_tid'])
    work['hospital_buffer_proxy'] = work.work_tid.isin(hosp_grids)
    work.to_parquet(CACHE / 'work_grid_evidence.parquet', index=False)
    presence = pd.concat(presence_parts, ignore_index=True) if presence_parts else pd.DataFrame(columns=['nid','facility_id','day'])
    freq = distinct_weekdays(presence)
    freq.to_parquet(CACHE / 'weekday_presence.parquet',index=False)
    counters['users_with_work_evidence'] = int(work.nid.nunique())
    counters['users_with_work_hospital_proxy'] = int(work.loc[work.hospital_buffer_proxy,'nid'].nunique())
    counters['frequent_hospital_pairs'] = int(freq.working_days.ge(10).sum())
    dump(REPORT / 'raw_input_audit.json', dict(counters))
    dump(REPORT / 'medical_age_codes.json', dict(age))
    dump(CACHE / 'extraction_complete.json', {'parts':number+1,'raw_rows':source_offset, 'medical_rows':counters['medical_argmax_rows']})
    print('extract completed',dict(counters),flush=True)

def facility_presence(chunksize=500000):
    """Rebuild all-facility weekday evidence independently from destination stays."""
    c=pd.read_parquet(CACHE/'candidates.parquet').query('distance_m <= 100')
    multiplicity=c.groupby('Tid').facility_id.nunique()
    lookup=c[c.Tid.isin(multiplicity[multiplicity==1].index)].set_index('Tid').facility_id
    parts=[]
    for i,d in enumerate(pd.read_csv(RAW,usecols=['nid','end_tid','stime','etime'],chunksize=chunksize)):
        d=d[d.end_tid.isin(lookup.index)].copy()
        f=time_flags(d.stime,d.etime)
        a=pd.to_datetime(d.stime,errors='coerce')
        d=d.loc[f.weekday&~f.invalid_time&a.ge('2018-11-01')&a.lt('2018-12-01')].copy()
        d['facility_id']=d.end_tid.map(lookup)
        d['day']=f.loc[d.index,'day']
        parts.append(d[['nid','facility_id','day']].drop_duplicates())
        if i%20==0:print('facility presence scanned chunks',i+1,flush=True)
    freq=distinct_weekdays(pd.concat(parts,ignore_index=True))
    freq.to_parquet(CACHE/'weekday_presence.parquet',index=False)
    dump(REPORT/'facility_presence_audit.json',{'rule':'unique facility within 100 m of grid centroid, >=10 distinct working days',
        'frequent_pairs':int(freq.working_days.ge(10).sum()),
        'frequent_CHC_pairs':int((freq.working_days.ge(10)&freq.facility_id.str.startswith('C:')).sum()),
        'status':'location proxy, not confirmed employment; ambiguous multi-facility grids excluded from frequency attribution'})

def population_validation(source_lookup, ids):
    home = pd.read_csv(HOME, usecols=['nid','home_tid'])
    for col in home:
        home[col] = pd.to_numeric(home[col],errors='coerce')
    home = home.dropna().drop_duplicates()
    conflicts = home.groupby('nid').home_tid.nunique()
    ambiguous = conflicts[conflicts>1].index
    home = home[~home.nid.isin(ambiguous)].drop_duplicates('nid')
    home['OBJECTID'] = home.home_tid.map(source_lookup)
    counts = home.groupby('OBJECTID').nid.nunique().rename('identified_home_users')
    pop = pd.read_csv(RESULT / 'population_crosswalk.csv').set_index('OBJECTID').join(counts).reindex(ids)
    pop.identified_home_users = pop.identified_home_users.fillna(0)
    pop['population_proxy_weight'] = pop['七普扩'].div(pop.identified_home_users.replace(0,np.nan))
    save(pop.reset_index(),RESULT / 'community_representativeness.csv')
    age = pd.read_csv(ATTR, usecols=['nid','age']).drop_duplicates()
    conflicting_age = age.groupby('nid').age.nunique().gt(1)
    age = age[~age.nid.isin(conflicting_age[conflicting_age].index)].drop_duplicates('nid')
    save(age.age.fillna('unknown').astype(str).value_counts().rename_axis('age_code').reset_index(name='users'), RESULT / 'age_code_distribution.csv')
    ratio = pop['七普扩'].div(pop['联通人'].replace(0,np.nan))
    dump(REPORT / 'representativeness.json', {
        'home_users':len(home),'ambiguous_home_users_excluded':len(ambiguous),
        'home_users_without_admin_match':int(home.OBJECTID.isna().sum()),
        'home_vs_operator_population':correlation(pop.identified_home_users,pop['联通人']),
        'home_vs_seventh_census_expansion_proxy':correlation(pop.identified_home_users,pop['七普扩']),
        'population_operator_ratio_min':float(ratio.min()),'population_operator_ratio_max':float(ratio.max()),
        'population_source':'七普扩 is a supplied expanded population field; independent census provenance not established; 2020/2018 timing differs',
        'age_status':'Encoded classes; no age-code dictionary found. No invented age cutoffs or elderly adjustment.',
        'operator_market_share':'No verified 2018 Shenzhen percentage; no 20% multiplier.'})
    return pop.population_proxy_weight

def analyze():
    complete = json.loads((CACHE/'extraction_complete.json').read_text('utf-8'))
    files = [CACHE / ('medical_%04d.parquet'%i) for i in range(complete['parts'])]
    visits = pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    assert len(visits)==complete['medical_rows']
    # Apply the current outpatient bound even when extraction parts are cached.
    visits['long_stay'] = visits.minutes.ge(480)
    # Do not count duplicate source observations as separate visits.
    visit_key = ['nid','end_tid','stime','etime']
    duplicates = int(visits.duplicated(visit_key).sum())
    # Ambiguous home/origin values are excluded instead of taking an arbitrary duplicate.
    duplicates_all = visits[visits.duplicated(visit_key,keep=False)]
    conflicts = duplicates_all.groupby(visit_key,dropna=False)[['home_tid','start_tid']].nunique().gt(1).any(axis=1)
    ambiguous_keys = conflicts[conflicts].index
    key_index = pd.MultiIndex.from_frame(visits[visit_key])
    ambiguous_rows = key_index.isin(ambiguous_keys)
    visits = visits.loc[~ambiguous_rows].drop_duplicates(visit_key).reset_index(drop=True)
    work = pd.read_parquet(CACHE/'work_grid_evidence.parquet')
    visits['work_hospital_proxy'] = visits.nid.isin(work.loc[work.hospital_buffer_proxy,'nid'])
    work_pairs = pd.MultiIndex.from_frame(work[['nid','work_tid']])
    visits['at_work_grid'] = pd.MultiIndex.from_frame(visits[['nid','end_tid']]).isin(work_pairs)
    visits['work_known'] = visits.nid.isin(work.nid)
    visits['at_home_grid'] = visits.end_tid.eq(visits.home_tid)
    candidates = pd.read_parquet(CACHE/'candidates.parquet')
    h100 = candidates.query('distance_m <= 100')
    unique_hosp = h100.groupby('Tid').facility_id.nunique()
    unique_hosp = h100[h100.Tid.isin(unique_hosp[unique_hosp==1].index)].set_index('Tid').facility_id
    visits['hospital_presence_proxy_id'] = visits.end_tid.map(unique_hosp)
    freq = pd.read_parquet(CACHE/'weekday_presence.parquet').query('working_days >= 10')
    frequent_pairs = pd.MultiIndex.from_frame(freq[['nid','facility_id']])
    visits['frequent_presence'] = pd.MultiIndex.from_frame(visits[['nid','hospital_presence_proxy_id']]).isin(frequent_pairs)
    inside = candidates.query("kind == 'hospital' and distance_m <= 0.000001")
    visits['hospital_AOI_grid_proxy'] = visits.end_tid.isin(inside.Tid)
    for radius in (200,300,500):
        visits['candidate_'+str(radius)] = visits.end_tid.isin(candidates.loc[candidates.distance_m<=radius,'Tid'])
    visits['keep_main'] = eligible(visits,30) & visits.candidate_300
    visits.to_parquet(CACHE/'medical_screened_records.parquet',index=False)
    visits.loc[visits.keep_main].to_parquet(CACHE/'outpatient_records.parquet',index=False)
    nodes = pd.read_csv(RESULT/'node_order.csv')
    ids = nodes.OBJECTID.to_numpy()
    source_lookup = pd.read_csv(RESULT/'grid_community.csv').set_index('Tid').OBJECTID.to_dict()
    target_lookup = pd.read_csv(RESULT/'grid_neighborhood.csv').groupby('Tid').objectid.apply(lambda x:sorted(set(x))).to_dict()
    attraction_path = ROOT/'LDA/result/isochrone_medical_level.csv'
    attraction_frame = pd.read_csv(attraction_path)
    assert attraction_frame.objectid.is_unique
    assert np.isfinite(attraction_frame.medical_level).all() and attraction_frame.medical_level.ge(0).all()
    attraction = attraction_frame.set_index('objectid').medical_level.to_dict()
    assert set(t for ts in target_lookup.values() for t in ts).issubset(attraction)
    save(attraction_frame, RESULT/'neighborhood_attraction_scores.csv')
    dump(REPORT/'attraction_allocation.json',{
        'source':file_manifest([attraction_path])[0], 'score_column':'medical_level',
        'score_definition':'Existing whole-isochrone sum of dominant medical-topic grid probabilities, from notebook cell 47; attraction proxy, not physical capacity.',
        'external_overlap_rule':'score_j / sum(candidate_circle_scores)',
        'internal_priority':True,'single_target_rule':'all flow to the only target',
        'all_zero_overlap_rule':'unallocated mass, no equal-weight fallback',
        'zero_score_circles':attraction_frame.loc[attraction_frame.medical_level.eq(0),'objectid'].tolist()})
    masks = {'baseline_corrected_time_medical':pd.Series(True,index=visits.index)}
    for radius in (200,300,500):
        for minimum in (30,45,60):
            masks['r%d_d%d'%(radius,minimum)] = eligible(visits,minimum)&visits['candidate_'+str(radius)]
    masks['work_nearby_retained'] = eligible(visits,30,exclude_work=False)&visits.candidate_300
    masks['strict_grid_proxy'] = (eligible(visits,60)&visits.minutes.le(240)&visits.hospital_AOI_grid_proxy
                                  &~visits.at_home_grid&~visits.at_work_grid&visits.work_known)
    for buffer in (0,100,200):
        tids = candidates.query("kind == 'hospital'").loc[lambda x:x.distance_m<=buffer,'Tid']
        masks['AOI_buffer_%d_grid_proxy'%buffer] = eligible(visits,30)&visits.end_tid.isin(tids)
    mass, matrices, hospital_rows = [], {}, []
    ext = pd.read_csv(RESULT/'external_2018_deduplicated.csv',dtype={'组织机构代码':str})
    for name,mask in masks.items():
        subset = visits.loc[mask]
        for origin in ('home_tid','start_tid'):
            grouped = subset.groupby([origin,'end_tid']).size().reset_index(name='count')
            low_support=float(grouped.loc[grouped['count']<=3,'count'].sum())
            grouped=grouped[grouped['count']>3]
            matrix, audit = split_neighborhood_flow(grouped,source_lookup,target_lookup,ids,attraction)
            audit['grid_OD_count_le_3_mass']=low_support
            assert np.isclose(sum(audit.values()),len(subset))
            matrices[(name,origin)] = matrix
            np.savetxt(RESULT/('flow_%s_%s.csv'%(name,origin)),matrix,delimiter=',',fmt='%.8f')
            mass.append({'scenario':name,'origin':origin,'retained_records':len(subset),**audit})
        # Fractional candidate weights are an uncertainty benchmark, not probabilities of care.
        if name.startswith('r'):
            radius = int(name.split('_')[0][1:])
            c = candidates.loc[candidates.distance_m<=radius].copy()
            c['weight'] = 1/c.groupby('Tid').facility_id.transform('size')
            grid_counts = subset.groupby('end_tid').size()
            c['allocated_visits'] = c.Tid.map(grid_counts).fillna(0)*c.weight
            hospital = c.groupby('external_code').allocated_visits.sum()
            comparison = ext.copy()
            comparison['identified_fractional_visits'] = comparison['组织机构代码'].map(hospital).fillna(0)
            comparison['has_matched_facility'] = comparison['组织机构代码'].isin(c.external_code.dropna())
            comparison['matched_hospital_AOI'] = comparison['组织机构代码'].isin(c.loc[c.kind.eq('hospital'),'external_code'].dropna())
            comparison['scenario'] = name
            matched = comparison[comparison.has_matched_facility]
            covered = comparison[comparison.matched_hospital_AOI]
            hospital_rows.append({'scenario':name,**correlation(matched['identified_fractional_visits'],matched['门诊人次']),
                                 'hospital_AOI_covered_only':correlation(covered['identified_fractional_visits'],covered['门诊人次']),
                                 'allocated_mass':float(c.allocated_visits.sum()),
                                 'external_matched_mass':float(c.loc[c.external_code.notna(),'allocated_visits'].sum()),
                                 'unmatched_mass':float(c.loc[c.external_code.isna(),'allocated_visits'].sum())})
            if name=='r300_d30':
                matched_total=comparison.identified_fractional_visits.sum()
                external_total=matched['门诊人次'].sum()
                comparison['identified_share_matched'] = comparison.identified_fractional_visits/matched_total if matched_total else np.nan
                comparison['external_share_matched'] = comparison['门诊人次'].where(comparison.has_matched_facility)/external_total if external_total else np.nan
                save(comparison,RESULT/'hospital_external_validation.csv')
                save(c.drop(columns=['match_name']),RESULT/'facility_allocated_visits.csv')
            assert np.isclose(c.allocated_visits.sum(),len(subset))
        print('analyze',name,len(subset),flush=True)
    save(pd.DataFrame(mass),RESULT/'scenario_flow_mass.csv')
    dump(REPORT/'hospital_correlations.json',hospital_rows)
    stability=[]
    for origin in ('home_tid','start_tid'):
        base=matrices[('r300_d30',origin)]
        for (name,which),m in matrices.items():
            if which!=origin:continue
            union=(base>0)|(m>0)
            off=~np.eye(len(ids),dtype=bool)
            stability.append({'scenario':name,'origin':origin,
                'OD_positive_union':correlation(base[union],m[union]),
                'community_outflow':correlation((base*off).sum(axis=1),(m*off).sum(axis=1)),
                'community_external_outflow_fraction':correlation(np.divide((base*off).sum(axis=1),base.sum(axis=1),out=np.full(len(ids),np.nan),where=base.sum(axis=1)>0),np.divide((m*off).sum(axis=1),m.sum(axis=1),out=np.full(len(ids),np.nan),where=m.sum(axis=1)>0))})
    dump(REPORT/'sensitivity_correlations.json',stability)
    weights = population_validation(source_lookup,ids)
    base = matrices[('r300_d30','home_tid')]
    # Rows with no population denominator remain unweighted; they are separately flagged.
    expanded = base * weights.fillna(1).values[:,None]
    np.savetxt(RESULT/'flow_r300_d30_home_tid_population_proxy.csv',expanded,delimiter=',',fmt='%.8f')
    dump(REPORT/'population_weight_sensitivity.json',{
        'label':'supplied 七普扩 sensitivity, not independently validated census correction',
        'undefined_weight_nodes':ids[weights.isna()].tolist(),
        'outflow_correlation':correlation(base.sum(axis=1),expanded.sum(axis=1)),
        'unsupported_elderly_destination_recovery':False})
    # Compare with frozen downstream input without replacing it.
    old = np.loadtxt(ROOT/'26上实验/gat/data/flow_matrix.csv',delimiter=',')
    assert old.shape==base.shape
    dump(REPORT/'original_flow_comparison.json',{'original_sum':float(old.sum()),
         'adjusted_sum':float(base.sum()),'OD_positive_union':correlation(old[(old>0)|(base>0)],base[(old>0)|(base>0)]),
         'interpretation':'Raw-record time semantics and screening differ; not an isolated threshold effect.'})
    flags = ['invalid_time','overnight','long_stay','peak_arrival','work_hospital_proxy','frequent_presence','at_home_grid','at_work_grid','work_known','hospital_AOI_grid_proxy','candidate_200','candidate_300','candidate_500','keep_main']
    dump(REPORT/'screening_summary.json',{'medical_rows_before_dedup':complete['medical_rows'],
        'duplicate_medical_observations':duplicates,'ambiguous_duplicate_rows_excluded':int(ambiguous_rows.sum()),
        'medical_unique_records':len(visits),'flag_counts_overlapping':{k:int(visits[k].sum()) for k in flags},
        'strict_grid_proxy_records':int(masks['strict_grid_proxy'].sum()),
        'true_station_AOI_strict_subset_available':False})
    print('analyze completed',flush=True)

def report():
    raw=json.loads((REPORT/'raw_input_audit.json').read_text('utf-8'))
    screened=json.loads((REPORT/'screening_summary.json').read_text('utf-8'))
    external=json.loads((REPORT/'external_input_audit.json').read_text('utf-8'))
    hospital=json.loads((REPORT/'hospital_correlations.json').read_text('utf-8'))
    main=next(x for x in hospital if x['scenario']=='r300_d30')
    stability=json.loads((REPORT/'sensitivity_correlations.json').read_text('utf-8'))
    mass=pd.read_csv(RESULT/'scenario_flow_mass.csv')
    main_mass=mass[(mass.scenario=='r300_d30')&(mass.origin=='home_tid')].iloc[0]
    location=json.loads((REPORT/'location_ambiguity.json').read_text('utf-8')) if (REPORT/'location_ambiguity.json').exists() else None
    table=['| 设置 | 保留记录 | OD 排序相关 | 行政社区对外流出份额相关 |','|---|---:|---:|---:|']
    for name in ['r200_d30','r300_d30','r300_d45','r300_d60','r500_d30','strict_grid_proxy']:
        s=next(x for x in stability if x['scenario']==name and x['origin']=='home_tid')
        n=int(mass[(mass.scenario==name)&(mass.origin=='home_tid')].iloc[0].retained_records)
        rho=s['OD_positive_union']['spearman'];share=s['community_external_outflow_fraction']['spearman']
        table.append(f'| {name} | {n:,} | {rho:.4f} | {share:.4f} |')
    text=f'''# 疑似门诊识别调整与输入验证

## 实际执行结果

- 原始记录：{raw['raw_rows']:,}；按原有 POI×时间先验 argmax 识别医疗记录：{raw['medical_argmax_rows']:,}。
- 去重后医疗记录：{screened['medical_unique_records']:,}；300 m 候选半径、30 min 下限主设置保留：{screened['flag_counts_overlapping']['keep_main']:,}。
- 外部 CSV {external['raw_rows']} 行，机构—年度去重后 {external['unique_institutions']} 家；重复门诊数值无冲突。
- 主设置机构匹配比较：n={main['n']}，Spearman={main['spearman']}。这是部分设施覆盖下的代理一致性，不是识别准确率。
- 居住来源主矩阵流量：{main_mass.assigned_mass:,.3f}；格网OD计数≤3排除 {main_mass.grid_OD_count_le_3_mass:,.0f}；目的生活圈未匹配 {main_mass.missing_target_mass:,.0f}。
- 外部重叠候选圈得分全零而无法加权分配的流量：{main_mass.zero_attraction_mass:,.0f}。

## 门诊记录敏感性

基准为 r300_d30，OD 排序仅在两方案正流量并集上计算；对外流出份额为该行政社区指向其他生活圈的流量/全部已分配流量。未设任意“通过”阈值。

{chr(10).join(table)}

## 当前执行口径

1. 原始数据字典明确 stime/etime 为终点驻留开始/结束；以同条记录相减计算停留，不将其视为出行时间。目的推断保留原有六类 POI 得分×时间先验及 argmax，仅用正确的到达小时。
2. 门诊峰值窗口 08:00–11:00、13:00–16:00，来源 Guo et al. (2026)。本项目具体实现为到达时刻位于半开区间；这项边界/到达判定是操作化定义，论文未给出该细节。
3. 停留≥30 min，45/60 min 敏感性；排除 ≥8 h 或跨日记录，最终保留时长小于8 h。长停留/跨夜为住院样停留排除规则，不确认临床住院身份。
4. 沿用项目 ptype=2 工作地口径；工作格网中心落入任一医院 AOI+100 m 缓冲区的用户排除。无工作地证据不等于无工作地。
5. 同一用户在同一唯一候选设施100 m格网代理范围出现≥10个不同工作日，标记频繁出现，并排除该用户在该设施的候选记录。医院按AOI距离、社康按点距离；这是疑似工作活动的保守代理，不确认职业身份。此为 TXT 适配，非 Xia 原文阈值；非通勤记录只能提供部分工作地/出现证据。
6. 候选设施为医院 AOI 和社康点；200/300/500 m 是格网中心到 AOI/点的距离。各候选等分只用于位置不确定性与外部匹配检验，不称为校准概率或贝叶斯—重力模型。半径敏感性是既有医疗类别条件下的候选覆盖检验，未重估六类 POI 分类。
7. 对外部验证仅做规范化名称的精确匹配；社康按上级举办医院合并。未自动合并相似院名/分院。单月识别量与全年门诊量比较排序/份额，不换算绝对全年人次。
8. OD 输出分别保留居住格网和本次出发格网作为来源，沿用格网OD计数>3条件。
   目的格网位于来源生活圈则全量归入；仅命中一个外部圈则全量归入该圈；命中多个外部圈时按各圈整体 medical_level / 候选圈得分之和分配。
   得分读取 LDA/result/isochrone_medical_level.csv，是既有圈内医疗主题格网概率汇总的吸引力代理。
   候选圈得分全零则单列无法分配量，不等分。
   各方案核验低计数排除量、未匹配量、零权重无法分配量与已分配量守恒。
   没有覆盖原始模型矩阵。
   

## 验证边界

- 500 m 格网中心不等同基站实际坐标；严格规则文件只称“格网代理子集”，不称真正的医院 AOI 高置信患者样本。
- 七普扩是供给文件中的人口扩样字段；目前未确立独立普查来源，扩样敏感性不作为独立验证，也不能恢复缺失老年人的目的地选择。年龄编码缺少对应表，未按编码大小推定年龄。
- 医院/社康设施数据的年份与 2018 年可比性未完全确认，名称匹配和设施覆盖造成验证偏差；医院门诊 CSV 不是所有医疗活动的真值标签。
- 无法通过这些聚合坐标识别陪同者；不能据此断言陪同不改变 OD 结构。
- 高频出现规则也可能排除高频门诊患者；只能作为疑似工作活动筛选，不能作为职业真值。严格格网代理子集还要求有工作地证据，样本覆盖与全量不同。
- 非医疗 POI 同生活圈检验使用可追溯的本地POI目录作为代理目录，不声称重建了原模型的逐停留后验候选。结果见 location_ambiguity.json；“至少一对共享”与“所有组合共享”分开报告。同圈不证明重叠生活圈下的 OD 分配权重相同。
- ΔS/韧性排序未在本次输入验证中重新计算；不能据此报告下游模型已经稳定。

## 文件

- `results/scenario_flow_mass.csv`：各方案保留数及 OD 守恒；`reports/sensitivity_correlations.json`：OD、社区流出及对外流出份额的 Spearman。
- `results/hospital_external_validation.csv`：医院/社康合并后的外部匹配；`results/facility_parent_crosswalk.csv`：待核对名称与未匹配项。
- `results/flow_r300_d30_home_tid.csv`、`results/flow_r300_d30_start_tid.csv`：666×666 候选输入；行列顺序见 `node_order.csv`。
- `local_records/medical_screened_records.parquet`：本地逐记录标记，含用户标识，不用于上传。
- `local_records/outpatient_records.parquet`：仅包含最终保留的疑似门诊记录。
- `reports/source_manifest.json`：输入文件路径、大小与 SHA-256；所有原始数据保留。
- `reports/outpatient_validation_reference_14649.md`：参照所提供论文的门诊量、到达时刻和空间敏感性辅助检验。

## 文献依据

- Guo Q, Ren H, Shao X, Yang W (2026). https://doi.org/10.3389/fpubh.2026.1841112 — 提供峰值门诊时间和100 m医院缓冲区设置；原文停留>1 h，本项目下限另按TXT。
- Xia J et al. (2026). https://doi.org/10.1038/s44360-026-00100-6 — 提供30/45/60 min检验；其GPS数据精度及跨两月出现天数规则与本项目不同。
'''
    (REPORT/'validation_report.md').write_text(text,encoding='utf-8')
    dump(REPORT/'run_status.json',{'status':'completed_input_validation_and_screening',
        'target':'suspected_outpatient_only',
        'unperformed':['true_station_AOI_strict_subset','downstream_deltaS_resilience'],
        'original_model_inputs_overwritten':False})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','prepare','extract','presence','analyze','report'],default='all')
    parser.add_argument('--chunksize',type=int,default=500000)
    args=parser.parse_args()
    from grid_outpatient import main as grid_main
    grid_main(args.stage,args.chunksize)
    return
    for p in (RESULT,REPORT,CACHE):p.mkdir(parents=True,exist_ok=True)
    if args.stage in ('all','prepare'):prepare()
    if args.stage in ('all','extract'):extract(args.chunksize)
    if args.stage=='presence':facility_presence(args.chunksize)
    if args.stage in ('all','analyze'):analyze()
    if args.stage=='all':
        from audit_location_ambiguity import main as audit_location
        from audit_model_inputs import main as audit_model
        audit_location()
        audit_model()
        from validate_outpatient_reference import main as validate_outpatient
        validate_outpatient()
    if args.stage in ('all','report'):report()
    if args.stage=='all':
        from verify_outputs import main as verify
        verify()

if __name__=='__main__':main()
