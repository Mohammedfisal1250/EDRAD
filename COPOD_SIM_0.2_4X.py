from rpy2.robjects import pandas2ri
pandas2ri.activate()
import matplotlib.pyplot as plt
import numpy as np
import pysam
import os
import gc
import pandas as pd
from numba import njit
import rpy2.robjects as robjects
import datetime
import math
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from scipy.stats import norm
import scipy
from sklearn import preprocessing
from sklearn.cluster import KMeans
from os.path import exists
from pathlib import Path
import os.path
from os.path import exists as file_exists
from sklearn.metrics import euclidean_distances
import rpy2.situation
for row in rpy2.situation.iter_info():
    print(row)
import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pyod.models.base import BaseDetector
from sklearn.preprocessing import scale
from pyod.models.copod import COPOD # Import COPOD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

def read_bam(file):
    sam_file = pysam.AlignmentFile(file, "rb",ignore_truncation=True)
    chr_list = sam_file.references
    return chr_list

def read_ref(file, chr_num, ref):
    if os.path.exists(file):
        print("Read reference file: " + str(file))
        with open(file, 'r') as f:
            line = f.readline()
            for line in f:
                lines = line.strip()
                ref[chr_num] += lines
    else:
        print("Warning: can not open " + str(file) + '\n')
    return ref

def bins(ref, bin_size, chr_len, file):
    chr_tag = np.full(23, 0)
    chr_list = np.arange(23)
    max_num = int(chr_len.max() / bin_size) + 1
    init_rd = np.full((23, max_num), 0.0)
    # read bam file and get bin rd
    print("Read bam file: " + str(file))
    sam_file = pysam.AlignmentFile(file, "rb",ignore_truncation=True)
    for line in sam_file:
        idx = int(line.pos / bin_size)
        if idx > int(chr_len.max()// bin_size):
            continue
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                init_rd[int(chr)][idx] += 1
                chr_tag[int(chr)] = 1
    chr_list = chr_list[chr_tag > 0]
    chr_num = len(chr_list)
    rd_list = [[] for _ in range(chr_num)]
    pos_list = [[] for _ in range(chr_num)]
    init_gc = np.full((chr_num, max_num), 0)
    pos = np.full((chr_num, max_num), 0)
    # initialize bin_data and bin_head
    count = 0
    for i in range(len(chr_list)):
        chr = chr_list[i]
        bin_num = int(chr_len[chr] / bin_size) + 1
        for j in range(bin_num):
            pos[i][j] = j
            cur_ref = ref[chr][j * bin_size:(j + 1) * bin_size]
            N_count = cur_ref.count('N') + cur_ref.count('n')
            if N_count == 0:
                gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
            else:
                gc_count = 0
                init_rd[chr][j] = -1000000
                count = count + 1
            init_gc[i][j] = int(round(gc_count / bin_size, 3) * 1000)
        # delete
        cur_rd = init_rd[chr][:bin_num]
        cur_gc = init_gc[i][:bin_num]
        cur_pos = pos[i][:bin_num]
        cur_rd = cur_rd / 1000
        index = cur_rd >= 0
        rd = cur_rd[index]
        GC = cur_gc[index]
        cur_pos = cur_pos[index]
        rd[rd == 0] = mode_rd(rd)
        rd = gc_correct(rd, GC)
        pos_list[i].append(cur_pos)
        rd_list[i].append(rd)
    del init_rd, init_gc, pos
    gc.collect()
    return rd_list, pos_list, chr_list

def mode_rd(rd):
    new_rd = np.full(len(rd), 0)
    for i in range(len(rd)):
        new_rd[i] = int(round(rd[i], 3) * 1000)
    count = np.bincount(new_rd)
    count_list = np.full(len(count) - 49, 0)
    for i in range(len(count_list)):
        count_list[i] = np.mean(count[i:i + 50])
    mode_min = np.argmax(count_list)
    mode_max = mode_min + 50
    mode = (mode_max + mode_min) / 2
    mode = mode / 1000
    return mode

def gc_correct(rd, gc):
    # correcting gc bias
    bin_count = np.bincount(gc)
    global_rd_ave = np.mean(rd)
    for i in range(len(rd)):
        if bin_count[gc[i]] < 2:
            continue
        mean = np.mean(rd[gc == gc[i]])
        rd[i] = global_rd_ave * rd[i] / mean
    return rd

def normalize(scores):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

def calc_scores(data):
    models = [

        ('COPOD', COPOD(contamination=0.5))


    ]
    
    outlier_scores = []
    for name, model in models:
        model.fit(data)
        scores = model.decision_function(data)
        outlier_scores.append(scores)
        
    outlier_scores = np.array(outlier_scores)
    
    # Using PCA for dimensionality reduction
    pca = PCA(n_components=1)
    combined_pca = pca.fit_transform(outlier_scores.T).flatten()
    normalized_scores = normalize(combined_pca)
    
    return normalized_scores

def scaling_rd(rd, mode):
    posit_rd = rd[rd > mode]
    neg_rd = rd[rd < mode]
    if len(posit_rd) < 50:
        mean_max_rd = np.mean(posit_rd)
    else:
        sort = np.argsort(posit_rd)
        max_rd = posit_rd[sort[-50:]]
        mean_max_rd = np.mean(max_rd)
    if len(neg_rd) < 50:
        mean_min_rd = np.mean(neg_rd)
    else:
        sort = np.argsort(neg_rd)
        min_rd = neg_rd[sort[:50]]
        mean_min_rd = np.mean(min_rd)
    scaling = mean_max_rd / (mode + mode - mean_min_rd)
    for i in range(len(rd)):
        if rd[i] < mode:
            rd[i] /= scaling
    return rd

def seg_rd(rd, bin_head, seg_start, seg_end, seg_count):
    seg_rd = np.full(len(seg_count), 0.0)
    for i in range(len(seg_rd)):
        seg_rd[i] = np.mean(rd[seg_start[i]:seg_end[i]])
        seg_start[i] = bin_head[seg_start[i]] * binSize + 1
        if seg_end[i] == len(bin_head):
            seg_end[i] = len(bin_head) - 1
        seg_end[i] = bin_head[seg_end[i]] * binSize + binSize
    return seg_rd, seg_start, seg_end

def write_data_file(chr, seg_start, seg_end, seg_count, outlier_scores,label,index):
    output = open(p_value_file, "w")
    output.write("Chr Num " + '\t' + " Start Position " + '\t' + " End Position " + '\t' + "  all_rd " + '\t\t' + " outlier_scores " + '\t\t' +"label"+'\t\t' +"index"+'\n')
    for i in range(len(outlier_scores)):
        output.write(str(chr[i]) + '\t ' + str(seg_start[i]) + ' \t ' + str(seg_end[i]) + ' \t ' + str(seg_count[i]) + ' \t ' + str(outlier_scores[i]) + ' \t ' +str(label[i])+' \t ' +str(index[i])+ '\n')

def write_dataset_file(seg_start, seg_count,label):
    output_dataset = open(dataset_file, "w")
    output_dataset.write("seg_start " + '\t' + " all_rd " + '\t' +"label"+'\n')

    for i in range(len(outlier_scores)):
        output_dataset.write( str(seg_start[i]) + '\t ' + str(seg_count[i]) + ' \t ' +str(label[i])+ '\n')


def write_cnv_file(chr, cnv_start, cnv_end, cnv_type, cn, filename):
    output = open(filename, "w")
    for i in range(len(cnv_type)):
        if cnv_type[i] == 2:
            output.write("Chr" + str(chr[i]) + '\t' + str(cnv_start[i]) + '\t' + str(
                cnv_end[i]) + '\t' + str("gain") + '\t' + str(cn[i]) + '\n')
        else:
            output.write("Chr" + str(chr[i]) + '\t' + str(cnv_start[i]) + '\t' + str(
                cnv_end[i]) + '\t' + str("loss") + '\t' + str(cn[i]) + '\n')

def read_seg_file(num_col, num_bin):
    seg_start = []
    seg_end = []
    seg_count = []
    seg_len = []
    with open("seg", 'r') as f:
        for line in f:
            line_str_list = line.strip().split('\t')
            start = (int(line_str_list[0]) - 1) * num_col + int(line_str_list[2]) - 1
            end = (int(line_str_list[0]) - 1) * num_col + int(line_str_list[3]) - 1
            if start < num_bin:
                if end > num_bin:
                    end = num_bin - 1
                seg_start.append(start)
                seg_end.append(end)
                seg_count.append(float(line_str_list[5]))
                seg_len.append(int(line_str_list[4]))
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    return seg_start, seg_end, seg_count, seg_len

def calculating_copy_number(mode, cnv_rd, cnv_type):
    cn = np.full(len(cnv_type), 0)
    index = cnv_type == 1
    lossRD = cnv_rd[index]
    if len(lossRD) > 2:
        data = np.c_[lossRD, lossRD]
        del_type = KMeans(n_clusters=2, random_state=9).fit_predict(data)
        cnv_type[index] = del_type
        if np.mean(lossRD[del_type == 0]) < np.mean(lossRD[del_type == 1]):
            homo_rd = np.mean(lossRD[del_type == 0])
            hemi_rd = np.mean(lossRD[del_type == 1])
            for i in range(len(cn)):
                if cnv_type[i] == 0:
                    cn[i] = 0
                elif cnv_type[i] == 1:
                    cn[i] = 1
        else:
            hemi_rd = np.mean(lossRD[del_type == 0])
            homo_rd = np.mean(lossRD[del_type == 1])
            for i in range(len(cn)):
                if cnv_type[i] == 1:
                    cn[i] = 0
                elif cnv_type[i] == 0:
                    cn[i] = 1
        purity = 2 * (homo_rd - hemi_rd) / (homo_rd - 2 * hemi_rd)
        for i in range(len(cnv_type)):
            if cnv_type[i] == 2:
                cn[i] = int(2 * cnv_rd[i] / (mode * purity) - 2 * (1 - purity) / purity)
    return cn

def boxplot(outlier_scores):
    four = pd.Series(outlier_scores).describe()
    Q1 = four['25%']
    Q3 = four['75%']
    IQR = Q3 - Q1
    upper = Q3 + 0.75 * IQR
    lower = Q1 - 0.75 * IQR
    return upper

def combining_cnv(seg_chr, seg_start, seg_end, seg_count,outlier_scores, upper, mode):
    index = outlier_scores > upper
    print("index=",index)
    CNV_chr = seg_chr[index]
    CNV_start = seg_start[index]
    CNV_end = seg_end[index]
    CNV_RD = seg_count[index]
    type = np.full(len(CNV_RD), 1)
    for i in range(len(CNV_RD)):
        if CNV_RD[i] > mode:
            type[i] = 2
    for i in range(len(CNV_RD) - 1):
        if CNV_end[i] + 1 == CNV_start[i + 1] and type[i] == type[i + 1]:
            CNV_start[i + 1] = CNV_start[i]
            type[i] = 0
    index = type != 0
    CNV_RD = CNV_RD[index]
    CNV_chr = CNV_chr[index]
    CNV_start = CNV_start[index]
    CNV_end = CNV_end[index]
    CNV_type = type[index]
    return CNV_chr, CNV_start, CNV_end, CNV_RD, CNV_type
def label_score(outlier_scores, upper):
    label = np.full(len(outlier_scores), 0)
    print (label)
    for i in range(len(outlier_scores)):
        if outlier_scores[i] > upper:
            label[i]=1
        else:
            label[i]=0
    return label

# get params
for z in range(1,51):
    num = z;
    start = datetime.datetime.now()
    bam = "/mnt/e/DATA/Sim_chr21_0.2_4x/sim"+ str(z) +"_4_4100_read.sort.bam";
    ref_path = "/mnt/e/CNV-LOF"
    binSize = 1000
    path = os.path.abspath('.')
    p_value_file="/mnt/e/my_method/COPOD_SIM/SIM_0.2_4X/sim_0.2_4x_chr_" + str(z) + "_SCORE.txt"
    outfile ="/mnt/e/my_method/COPOD_SIM/SIM_0.2_4X/sim_0.2_4x_chr_" + str(z) + "_RESULT.txt"
    dataset_file = "/mnt/e/my_method/COPOD_SIM/SIM_0.2_4X/sim_0.2_4x_chr_" + str(z) + "_dataset.txt"
    col = 50
    print (path)
    seg_path = path + str("/seg")

    ref = [[] for i in range(23)]
    refList = read_bam(bam)
    for i in range(len(refList)):
        chr = refList[i]
        chr_num = chr.strip('chr')
        if chr_num.isdigit():
            chr_num = int(chr_num)
            reference = ref_path + '/chr' + str(chr_num) + '.fa'
            ref = read_ref(reference, chr_num, ref)

    chrLen = np.full(23, 0)
    for i in range(1, 23):
        chrLen[i] = len(ref[i])
    RDList, PosList, chrList = bins(ref, binSize, chrLen, bam)
    all_chr = []
    all_rd = []
    all_start = []
    all_end = []
    modeList = np.full(len(chrList), 0.0)
    for i in range(len(chrList)):
        RD = np.array(RDList[i][0])
        pos = np.array(PosList[i][0])
        num_bin = len(RD)
        modeList[i] = mode_rd(RD)
        scale_rd = scaling_rd(RD, modeList[i])
        print("segment count...")
        v = robjects.FloatVector(scale_rd)
        m = robjects.r['matrix'](v, ncol=col)
        robjects.r.source("CBS_data.R")
        robjects.r.CBS_data(m, seg_path)
        num_col = int(num_bin / col) + 1
        seg_start, seg_end, seg_count, seg_len = read_seg_file(num_col, num_bin)
        seg_count = np.array(seg_count)
        seg_count = seg_count[:-1]
        seg_start = seg_start[:-1]
        seg_end = seg_end[:-1]
        seg_count, seg_start, seg_end = seg_rd(RD, pos, seg_start, seg_end, seg_count)
        all_rd.extend(seg_count)
        all_start.extend(seg_start)
        all_end.extend(seg_end)
        all_chr.extend(chrList[i] for j in range(len(seg_count)))

    all_chr = np.array(all_chr)
    all_start = np.array(all_start)
    all_end = np.array(all_end)
    all_rd = np.array(all_rd)
    for i in range(len(all_rd)):
        if np.isnan(all_rd[i]).any():
            all_rd[i] = (all_rd[i - 1] + all_rd[i + 1]) / 2

    print("Calculating scores...")
    print("all_rd=", all_rd)
    data1 = all_rd.reshape(-1, 1)
    data = StandardScaler().fit_transform(data1)
    print("data size=", data.size)
    print("data=", data)
    data_filename = "/mnt/e/my_method/COPOD_SIM/SIM_0.2_4X/sim_0.2_4x_chr_" + str(z) + "_data.csv"
    df = pd.DataFrame(data, columns=['RD_Scores'])  # Assuming 'data' is a 2D array with one column
    df.to_csv(data_filename, index=False)
    print(f"Data saved to {data_filename}")
    outlier_scores = calc_scores(data)
    print("outlier_scores size =", outlier_scores.size)

    upper = boxplot(outlier_scores)
    label = label_score(outlier_scores, upper)
    print ("label=",label)
    index = outlier_scores > upper
    mode = np.mean(modeList)
    write_data_file(all_chr, all_start, all_end, all_rd, outlier_scores,label,index)
    write_dataset_file( all_start,seg_count,label)
    print("upper=",upper)
    df3= pd.DataFrame(outlier_scores)
    df3.plot.box(title='Boxplot outlier_scores');
    plt.show()
    CNV_chr, CNV_start, CNV_end, CNV_rd, CNV_type = combining_cnv(all_chr, all_start, all_end, all_rd, outlier_scores, upper,mode)
    cn = calculating_copy_number(mode, CNV_rd, CNV_type)
    write_cnv_file(CNV_chr, CNV_start, CNV_end, CNV_type, cn, outfile)
    end = datetime.datetime.now()
    print("running time: " + str((end - start).seconds) + " seconds")