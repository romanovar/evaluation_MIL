import numpy as np
import matplotlib.pyplot as plt


def plot_line_graph(line1, label1, line2, label2, line3, label3, line4, label4, line5, label5,  line6, label6,
                    x_axis_data, x_label, results_path, fig_name, text_string):

    fig = plt.figure()
    fig.text(0.5, 1, text_string, horizontalalignment='center', verticalalignment='center', fontsize=9)
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    plt.plot(x_axis_data, line1, 'g', label = label1)
    plt.plot(x_axis_data, line2, 'b', label=label2)
    if line3 is not None:
        plt.plot(x_axis_data, line3, ':k', label=label3)
        if line4 is not None:
            plt.plot(x_axis_data, line4, ':r', label=label4)
            if line5 is not None:
                plt.plot(x_axis_data, line5, ':b', label=label5)
                if line6 is not None:
                    plt.plot(x_axis_data, line6, 'r--', label=label6)
    # plt.tick_params(labelsize=2)
    plt.rc('legend', fontsize=7)
    plt.ylabel('score')
    plt.xlabel(x_label)
    plt.legend()
    plt.show()
    fig.savefig(
        results_path +  fig_name + '.jpg',
        bbox_inches='tight')
    plt.close(fig)
    plt.clf()



def overlap_coefficient(n00, n10, n01, n11 ):
    min_n01_n10 = np.minimum(n10, n01)
    return n11/(min_n01_n10 + n11)


def positive_Jaccard_index_batch(n00, n10, n01, n11 ):
    pos_jaccard_dist = n11/(n11+n10+n01)
    return pos_jaccard_dist


def corrected_overlap_coefficient(n00, n10, n01, n11):
    min_n01_n10 = np.minimum(n10, n01)

    N = n00 + n11 + n10 + n01
    expected_overlap = (n11+n01)*(n11+n10)/N

    corrected_score = ((n11 - expected_overlap)/(np.minimum((n11+n01), (n11+n10)) - expected_overlap))
    corrected_score2 = ((n00*n11 - n10*n01)/((min_n01_n10+n11)*(min_n01_n10 + n00)))
    # corrected_score = {:0.2f}(corrected_score)
    # corrected_score2 = round(corrected_score2, 3)
    # assert corrected_score == corrected_score2, "Error in computing some of the index! "
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


def corrected_positive_Jaccard(n00, n10, n01, n11):
    N = n00 + n11 + n10 + n01
    expected_positive_overlap = (n11+n01)*(n11+n10)/N

    corrected_score = ((n11 - expected_positive_overlap)/(n10 + n11 +n01 - expected_positive_overlap))
    corrected_score2 = (n00*n11 - n10*n01)/((n00*n11) - (n01*n10) + ((n10+n01)*N))
    # corrected_score = round(corrected_score, 3)
    # corrected_score2 = round(corrected_score2, 3)

    # assert corrected_score == corrected_score2, "Error in computing some of the index! "
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


def corrected_IOU(n00, n10, n01, n11):
    N = n00 + n11 + n10 + n01
    expected_positive_overlap = (n11 + n01) * (n11 + n10) / N
    expected_negative_overlap = (n00 + n01) * (n00 + n10) / N

    corrected_score = ((n11 + n00 - expected_positive_overlap - expected_negative_overlap) /
                       (n10 + n11 + n01 + n00 - expected_positive_overlap - expected_negative_overlap))
    corrected_score2 = (2*n00 * n11 - 2*n10 * n01) / (2*(n00 * n11) - 2*(n01 * n10) + ((n10 + n01) * N))

    # assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
    #          np.ma.masked_array(corrected_score2,
    #                             np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


def corrected_Jaccard_pigeonhole(n00, n10, n01, n11):
    N = n00 + n11 + n10 + n01
    pigeonhole_positive_correction = (2*n11 + n01 + n10) - N
    max_overlap = np.maximum(pigeonhole_positive_correction, 0)

    corrected_score = ((n11 - max_overlap) /
                       (n10 + n11 + n01 - max_overlap))
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))
#

# N = 200
# n11=50
# n00 = 1
#
# for step in range(1, 17):
#     n10 = N - n11
#     n01 = N  - n11
#     if n01 < 0:
#         print("Negative instances")
#     overlap_coeff = overlap_coefficient(n00, n10, n01, n11)
#     pos_jacc = positive_Jaccard_index_batch(n00, n10, n01, n11)
#     corrected_overlap =  corrected_overlap_coefficient(n00, n10, n01, n11)
#     corrected_jaccard = corrected_positive_Jaccard(n00, n10, n01, n11)
#
#     overlap_coll.append(overlap_coeff)
#     jacc_coll.append(pos_jacc)
#     corr_overlap_col.append(corrected_overlap)
#     corr_jaccard_coll.append(corrected_jaccard)
#     if step >= 8:
#         # print(corrected_jaccard)
#         print(corrected_overlap)
#         print(n11)
#         print(n00)
#     set1_size_col.append(n11)
#
#     n11 += 10
#     # set2_size -= 10
#
# print(len(jacc_coll))
# print(len(overlap_coll))
# print(len(set1_size_col))
# # plot_line_graph(overlap_coll, 'Overlap coefficient' ,jacc_coll, 'Positive Jaccard distance', None, None, None, None,
# #                 set1_size_col)
# # plot_line_graph(corr_overlap_col, 'Corrected Overlap coefficient' ,corr_jaccard_coll, 'Corrected Positive Jaccard distance',
# #                 None, None, None, None, set1_size_col)
# plot_line_graph(overlap_coll, 'Overlap coefficient' ,jacc_coll, 'Positive Jaccard distance',
#                 corr_overlap_col, 'Corrected Overlap coefficient' ,corr_jaccard_coll,
#                 'Corrected Positive Jaccard distance', set1_size_col)


################################################### fixing agreement ration n00 and n11 ########################
import math

def simulate_distributions_n01_n10_extreme(res_path):
    overlap_coll = []
    jacc_coll = []
    corr_overlap_col = []
    corr_jaccard_coll = []
    corr_iou_coll = []
    corr_jaccard_pgn_coll = []
    set1_size_col = []

    N = 200
    n11=5
    n00 = 5
    n01 = math.ceil((N - n11 - n00)/2)
    n10 = N - n01 - n11 - n00
    init_n10 = n10
    init_n01 = n01
    for step in range(0, int(n10/5)+1):
        assert n00>= 0, "instance number should be bigger than 0"
        assert n01 >= 0, "instance number should be bigger than 0"
        assert n10 >= 0, "instance number should be bigger than 0"
        assert n11 >= 0, "instance number should be bigger than 0"
        overlap_coeff = overlap_coefficient(n00, n10, n01, n11)
        pos_jacc = positive_Jaccard_index_batch(n00, n10, n01, n11)
        corrected_overlap =  corrected_overlap_coefficient(n00, n10, n01, n11)
        corrected_jaccard = corrected_positive_Jaccard(n00, n10, n01, n11)
        corrected_iou = corrected_IOU(n00, n10, n01, n11)
        corrected_pigeonhole = corrected_Jaccard_pigeonhole(n00, n10, n01, n11)


        overlap_coll.append(overlap_coeff)
        jacc_coll.append(pos_jacc)
        corr_overlap_col.append(corrected_overlap)
        corr_jaccard_coll.append(corrected_jaccard)
        corr_iou_coll.append(corrected_iou)
        corr_jaccard_pgn_coll.append(corrected_pigeonhole)

        if corrected_overlap <= -2:
            # print(corrected_jaccard)
            print("printing extreme")
            print(corrected_overlap)
            print(n01)
            print(n10)

        set1_size_col.append(n01)
        n01 += 5
        n10 -=5
    fig_text = 'Total N: ' + str(N) +  '\n n00: ' + str(n00) + '\n n11: ' + str(n11) + '\n Initial n01: ' + \
                   str(init_n01) + '\n Initial n10: ' + str(init_n10) + '\n n01 is increasing w.r.t. n10'
    plot_line_graph(overlap_coll, 'Overlap coefficient' ,jacc_coll, 'Positive Jaccard distance',
                    corr_overlap_col, 'Corrected Overlap coefficient' ,corr_jaccard_coll,
                    'Corrected Positive Jaccard distance', corr_iou_coll, "Corrected IoU",
                    corr_jaccard_pgn_coll, "Corrected Positive Jaccard using Pigeonhole",
                    set1_size_col, 'n01', res_path, 'distribution_Ns_extreme', fig_text)


def simulate_distributions_n01_n10_equal_distribution(res_path):
    overlap_coll = []
    jacc_coll = []
    corr_overlap_col = []
    corr_jaccard_coll = []
    corr_iou_coll = []
    corr_jaccard_pgn_coll =[]
    set1_size_col = []

    N = 200
    n11=50
    n00 = 50
    n01 = math.ceil((N - n11 - n00)/2)
    n10 = N - n01 - n11 - n00
    init_n10 = n10
    init_n01 = n01
    for step in range(0, int(n10/10)+1):
        assert n00>= 0, "instance number should be bigger than 0"
        assert n01 >= 0, "instance number should be bigger than 0"
        assert n10 >= 0, "instance number should be bigger than 0"
        assert n11 >= 0, "instance number should be bigger than 0"
        overlap_coeff = overlap_coefficient(n00, n10, n01, n11)
        pos_jacc = positive_Jaccard_index_batch(n00, n10, n01, n11)
        corrected_overlap =  corrected_overlap_coefficient(n00, n10, n01, n11)
        corrected_jaccard = corrected_positive_Jaccard(n00, n10, n01, n11)
        corrected_iou =  corrected_IOU(n00, n10, n01, n11)
        corrected_pigeonhole = corrected_Jaccard_pigeonhole(n00, n10, n01, n11)


        overlap_coll.append(overlap_coeff)
        jacc_coll.append(pos_jacc)
        corr_overlap_col.append(corrected_overlap)
        corr_jaccard_coll.append(corrected_jaccard)
        corr_iou_coll.append(corrected_iou)
        corr_jaccard_pgn_coll.append(corrected_pigeonhole)
        if corrected_overlap >= 0.2:
            # print(corrected_jaccard)
            print("printing")
            print(corrected_overlap)
            print(n01)
            print(n10)

        set1_size_col.append(n01)
        n01 += 10
        n10 -=10
    fig_text = 'Total N: ' + str(N) +  '\n n00: ' + str(n00) + '\n n11: ' + str(n11) + '\n Initial n01: ' + \
                   str(init_n01) + '\n Initial n10: ' + str(init_n10) + '\n n01 is increasing w.r.t. n10'
    plot_line_graph(overlap_coll, 'Overlap coefficient' ,jacc_coll, 'Positive Jaccard distance',
                corr_overlap_col, 'Corrected Overlap coefficient' ,corr_jaccard_coll,
                'Corrected Positive Jaccard distance', corr_iou_coll, "Corrected IoU",
                    corr_jaccard_pgn_coll, "Corrected Positive Jaccard using Pigeonhole",
                    set1_size_col, 'n01', res_path, 'distribution_Ns_equal_distribution',
                fig_text)
    # plot_line_graph(overlap_coll, 'Overlap coefficient', jacc_coll, 'Positive Jaccard distance',
    #                 corr_overlap_col, 'Corrected Overlap coefficient',None, None, None, None, None,None,
    #                 set1_size_col, 'n01', results_path, 'distribution_Ns_equal_distribution',
    #                 fig_text)


def simulate_distributions_n01_n10(res_path):
    N = 200
    n11= n00 = 5
    # n00 = 5


    step_size = 5
    total_out_steps = (N/2 - n11)/step_size

    for step_agreement in range(0, int(total_out_steps+1)):
        print("n11 now is: "+str(n11))
        overlap_coll = []
        jacc_coll = []
        corr_overlap_col = []
        corr_jaccard_coll = []
        corr_iou_coll = []
        corr_jaccard_pgn_coll = []
        set1_size_col = []

        n01 = math.ceil((N - n11 - n00) / 2)
        n10 = N - n01 - n11 - n00
        init_n10 = n10
        init_n01 = n01
        for step in range(0, int(n10/step_size)+1):


            assert n00>= 0, "instance number should be bigger than 0"
            assert n01 >= 0, "instance number should be bigger than 0"
            assert n10 >= 0, "instance number should be bigger than 0"
            assert n11 >= 0, "instance number should be bigger than 0"
            # overlap_coeff = overlap_coefficient(n00, n10, n01, n11)
            # pos_jacc = positive_Jaccard_index_batch(n00, n10, n01, n11)
            # corrected_overlap =  corrected_overlap_coefficient(n00, n10, n01, n11)
            corrected_jaccard = corrected_positive_Jaccard(n00, n10, n01, n11)
            # corrected_iou = corrected_IOU(n00, n10, n01, n11)
            # corrected_pigeonhole = corrected_Jaccard_pigeonhole(n00, n10, n01, n11)


            # overlap_coll.append(overlap_coeff)
            # jacc_coll.append(pos_jacc)
            # corr_overlap_col.append(corrected_overlap)
            corr_jaccard_coll.append(corrected_jaccard)
            # corr_iou_coll.append(corrected_iou)
            # corr_jaccard_pgn_coll.append(corrected_pigeonhole)

            set1_size_col.append(n01)
            n01 += step_size
            n10 -=step_size
        fig_text = 'Total N: ' + str(N) +  '\n n00: ' + str(n00) + '\n n11: ' + str(n11) + '\n Initial n01: ' + \
                       str(init_n01) + '\n Initial n10: ' + str(init_n10) + '\n n01 is increasing w.r.t. n10'
        plot_line_graph(overlap_coll, 'Overlap coefficient' ,jacc_coll, 'Positive Jaccard distance',
                        corr_overlap_col, 'Corrected Overlap coefficient' ,corr_jaccard_coll,
                        'Corrected Positive Jaccard distance', corr_iou_coll, "Corrected IoU",
                        corr_jaccard_pgn_coll, "Corrected Positive Jaccard using Pigeonhole",
                        set1_size_col, 'n01', res_path, 'varying_n01_n10_'+str(step_agreement), fig_text)
        n00 +=step_size
        n11 += step_size


def simulate_distributions_n11_n00(res_path):

    N = 200
    new_n01= new_n10 = 5
    # n00 = 5

    step_size = 5
    total_out_steps = (N/2 - new_n01)/step_size

    for step_agreement in range(0, int(total_out_steps+1)):
        print("n11 now is: "+str(new_n01))
        overlap_coll = []
        jacc_coll = []
        corr_overlap_col = []
        corr_jaccard_coll = []
        corr_iou_coll = []
        corr_jaccard_pgn_coll = []
        set1_size_col = []

        new_n11 = math.ceil((N - new_n01 - new_n10) / 2)
        new_n00 = N - new_n11 - new_n01 - new_n10
        init_n00 = new_n00
        init_n11 = new_n11
        for step in range(0, int(new_n00/step_size)+1):


            assert new_n10>= 0, "instance number should be bigger than 0"
            assert new_n11 >= 0, "instance number should be bigger than 0"
            assert new_n00 >= 0, "instance number should be bigger than 0"
            assert new_n01 >= 0, "instance number should be bigger than 0"
            overlap_coeff = overlap_coefficient(new_n00, new_n10, new_n01, new_n11)
            pos_jacc = positive_Jaccard_index_batch(new_n00, new_n10, new_n01, new_n11)
            corrected_overlap =  corrected_overlap_coefficient(new_n00, new_n10, new_n01, new_n11)
            corrected_jaccard = corrected_positive_Jaccard(new_n00, new_n10, new_n01, new_n11)
            corrected_iou = corrected_IOU(new_n00, new_n10, new_n01, new_n11)
            corrected_pigeonhole = corrected_Jaccard_pigeonhole(new_n00, new_n10, new_n01, new_n11)


            overlap_coll.append(overlap_coeff)
            jacc_coll.append(pos_jacc)
            corr_overlap_col.append(corrected_overlap)
            corr_jaccard_coll.append(corrected_jaccard)
            corr_iou_coll.append(corrected_iou)
            corr_jaccard_pgn_coll.append(corrected_pigeonhole)

            if corrected_overlap <= -2:
                # print(corrected_jaccard)
                print("printing extreme")
                print(corrected_overlap)
                print(new_n11)
                print(new_n00)

            set1_size_col.append(new_n11)
            new_n11 += step_size
            new_n00 -=step_size
        fig_text = 'Total N: ' + str(N) +  '\n n01: ' + str(new_n01) + '\n n10: ' + str(new_n10) + '\n Initial n11: ' + \
                       str(init_n11) + '\n Initial n00: ' + str(init_n00) + '\n n11 is increasing w.r.t. n00'
        plot_line_graph(overlap_coll, 'Overlap coefficient' ,jacc_coll, 'Positive Jaccard distance',
                        corr_overlap_col, 'Corrected Overlap coefficient' ,corr_jaccard_coll,
                        'Corrected Positive Jaccard distance', corr_iou_coll, "Corrected IoU",
                        corr_jaccard_pgn_coll, "Corrected Positive Jaccard using Pigeonhole",
                        set1_size_col, 'n11', res_path, 'varying_n11_n00_'+str(step_agreement), fig_text)
        new_n10 +=step_size
        new_n01 += step_size

# simulate_distributions_n01_n10_extreme()
#
# simulate_distributions_n01_n10_equal_distribution()
#
# simulate_distributions_n01_n10()
# simulate_distributions_n11_n0()
def simulate_distributions_corollary5(res_path):

    n00 = 0
    n01 = 1
    n10 = 0
    n11 = 1
    N = n11+ n00+ n01 + n10
    new_n10 = 0

    step_size = 1
    total_steps = 30
    # total_out_steps = (N/2 - new_n01)/step_size
    corr_overlap_col = []
    corr_jaccard_coll = []
    overlap_coll = []
    jacc_coll = []

    corr_iou_coll = []
    corr_jaccard_pgn_coll = []
    set1_size_col = []
    for step_agreement in range(0, int(total_steps+1)):
        print("n10 now is: "+str(new_n10))



        corrected_jaccard = corrected_positive_Jaccard(n00, new_n10, n01, n11)
        corrected_iou = corrected_IOU(n00, new_n10, n01, n11)
            # corrected_pigeonhole = corrected_Jaccard_pigeonhole(new_n00, new_n10, new_n01, new_n11)


            # overlap_coll.append(overlap_coeff)
            # jacc_coll.append(pos_jacc)
            # corr_overlap_col.append(corrected_overlap)
        corr_jaccard_coll.append(corrected_jaccard)
        corr_iou_coll.append(corrected_iou)
            # corr_jaccard_pgn_coll.append(corrected_pigeonhole)
            #
            # if corrected_overlap <= -2:
            #     # print(corrected_jaccard)
            #     print("printing extreme")
            #     print(corrected_overlap)
            #     print(new_n11)
            #     print(new_n00)

        set1_size_col.append(new_n10)
        new_n10 += step_size

    fig_text = 'Total N: ' + str(N) +  '\n n01: ' + str(n01) + '\n n10: ' + str(n10) + '\n Initial n11: ' + \
                   str(n11) + '\n  n00: ' + str(n00) + '\n n10 is increasing'
    plot_line_graph(corr_jaccard_coll,
                    'Corrected Positive Jaccard distance', corr_iou_coll, "Corrected IoU", line3=None,
                    label3=None, line4=None, label4=None, line5=None, label5=None,  line6=None, label6=None,
                    x_axis_data= set1_size_col, x_label='n10_error', results_path=res_path,
                    fig_name='step'+str(step_agreement), text_string= fig_text)


def simulate_distributions_monotonic(res_path):
    n00 = 10
    n01 = 50
    n10 = 50
    n11 = 1
    N = n11 + n00 + n01 + n10
    new_n11 = 0
    new_n01 = n01

    step_size = 1
    total_steps = 50
    # total_out_steps = (N/2 - new_n01)/step_size
    corr_overlap_col = []
    corr_jaccard_coll = []
    overlap_coll = []
    jacc_coll = []

    corr_iou_coll = []
    corr_jaccard_pgn_coll = []
    set1_size_col = []
    for step_agreement in range(0, int(total_steps + 1)):
        print("n10 now is: " + str(new_n11))

        corrected_jaccard = corrected_positive_Jaccard(n00, n10, new_n01, new_n11)
        corrected_iou = corrected_IOU(n00, n10, new_n01, new_n11)
        # corrected_pigeonhole = corrected_Jaccard_pigeonhole(new_n00, new_n10, new_n01, new_n11)

        # overlap_coll.append(overlap_coeff)
        # jacc_coll.append(pos_jacc)
        # corr_overlap_col.append(corrected_overlap)
        corr_jaccard_coll.append(corrected_jaccard)
        corr_iou_coll.append(corrected_iou)
        # corr_jaccard_pgn_coll.append(corrected_pigeonhole)
        #
        # if corrected_overlap <= -2:
        #     # print(corrected_jaccard)
        #     print("printing extreme")
        #     print(corrected_overlap)
        #     print(new_n11)
        #     print(new_n00)

        set1_size_col.append(new_n11)
        new_n11 += step_size
        new_n01-= step_size

    fig_text = 'Total N: ' + str(N) + '\n n01: ' + str(n01) + '\n n10: ' + str(n10) + '\n Initial n11: ' + \
               str(n11) + '\n  n00: ' + str(n00) + '\n n11 is increasing'
    plot_line_graph(corr_jaccard_coll,
                    'Corrected Positive Jaccard distance', corr_iou_coll, "Corrected IoU", line3=None,
                    label3=None, line4=None, label4=None, line5=None, label5=None, line6=None, label6=None,
                    x_axis_data=set1_size_col, x_label='n10_error', results_path=res_path,
                    fig_name='step' + str(step_agreement), text_string=fig_text)


def corrected_positive_Jaccard_v2(n00, n10, n01, n11):
    N = n00 + n11 + n10 + n01
    expected_positive_overlap = (n11 + n01) * (n11 + n10) / N

    corrected_score = ((n11 - expected_positive_overlap) / (n10 + n11 + n01 - expected_positive_overlap))
    corrected_score2 = (n00 * n11 - n10 * n01) / ((n00 * n11) - (n01 * n10) + ((n10 + n01) * N))
    # assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
    #          np.ma.masked_array(corrected_score2, np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


def simulate_score_v1():
    a = 5
    b = 25
    c = 1
    d = 5
    for step in range(0, 50):
        print(corrected_positive_Jaccard_v2(d, c, b, a))
        a += 1


def simulate_score_v1():
    a = 5
    b = 25
    c = 1
    d = 5
    for step in range(0, 25):
        print(corrected_positive_Jaccard_v2(d, c, b, a))
        a += 1
        b -=1


def simulate_jaccard_increasing_subpopulation(increasing_group, decreasing_group, N, step_size, n00, n01, n10, n11 ):
    corr_jaccard_coll_n11 = []
    step_coll=[]
    if increasing_group.lower() == 'n00':
        init_increasing = n00
        updated_increasing = n00
    elif increasing_group.lower() == 'n01':
        init_increasing = n01
        updated_increasing = n01
    elif increasing_group.lower() == 'n10':
        init_increasing = n10
        updated_increasing = n10
    elif increasing_group.lower() == 'n11':
        init_increasing = n11
        updated_increasing = n11
    else:
        assert "Error, no group recognized"


    if decreasing_group.lower() == 'n00':
        init_decreasing = n00
        updated_decreasing = n00
    elif decreasing_group.lower() == 'n01':
        init_decreasing = n01
        updated_decreasing = n01
    elif decreasing_group.lower() == 'n10':
        init_decreasing = n10
        updated_decreasing = n10
    elif decreasing_group.lower() == 'n11':
        init_decreasing = n11
        updated_decreasing = n11
    else:
        assert "Error, no group recognized"

    fixed_group1 = math.ceil((N - init_increasing - init_decreasing) / 2)
    fixed_group2 =  N - fixed_group1 - init_increasing- init_decreasing
    # updated_increasing
    # updated_decreasing
    # # init_increasing_gr
    # # init_decreasing_gr =
    # new_n11 = math.ceil((N - new_n01 - new_n10) / 2)
    # new_n00 = N - new_n11 - new_n01 - new_n10
    # init_n00 = new_n00
    # init_n11 = new_n11



    ####### EXPERIMENT 2 ######################
    for step in range(0, int(init_decreasing / step_size) + 1):
        new_n00= new_n10= new_n01= new_n11 = -1


        assert updated_increasing >= 0, "instance number should be bigger than 0"
        assert updated_decreasing >= 0, "instance number should be bigger than 0"
        assert fixed_group1 >= 0, "instance number should be bigger than 0"
        assert fixed_group2 >= 0, "instance number should be bigger than 0"

        if increasing_group.lower() == 'n00':
            new_n00 = updated_increasing
        elif increasing_group.lower() == 'n01':
            new_n01 = updated_increasing
        elif increasing_group.lower() == 'n10':
            new_n10 = updated_increasing
        elif increasing_group.lower() == 'n11':
            new_n11 = updated_increasing
        else:
            assert "Error, no group recognized"

        if decreasing_group.lower() == 'n00':
            new_n00 = updated_decreasing
        elif decreasing_group.lower() == 'n01':
            new_n01 = updated_decreasing
        elif decreasing_group.lower() == 'n10':
            new_n10 = updated_decreasing
        elif decreasing_group.lower() == 'n11':
            new_n11 = updated_decreasing
        else:
            assert "Error, no group recognized"
        if new_n00 < 0:
            new_n00 = fixed_group1
        elif new_n01 < 0:
            new_n01 = fixed_group1
        elif new_n10 < 0:
            new_n10 = fixed_group1
        elif new_n11 < 0:
            new_n11 = fixed_group1

        if new_n00 < 0:
            new_n00 = fixed_group2
        elif new_n01 < 0:
            new_n01 = fixed_group2
        elif new_n10 < 0:
            new_n10 = fixed_group2
        elif new_n11 < 0:
            new_n11 = fixed_group2

        corrected_jaccard = corrected_positive_Jaccard(new_n00, new_n10, new_n01, new_n11)
        corr_jaccard_coll_n11.append(corrected_jaccard)

        step_coll.append(updated_increasing)
        updated_increasing += step_size
        updated_decreasing -= step_size
    return corr_jaccard_coll_n11, step_coll


def simulate_distributions_jacc_n11_n00(res_path):
    N = 200
    step_size = 5

    new_n01 = new_n10 = 50

    corr_jaccard_coll_n11 = []
    corr_jaccard_coll_n10 = []
    set1_size_col = []

    new_n11 = math.ceil((N - new_n01 - new_n10) / 2)
    new_n00 = N - new_n11 - new_n01 - new_n10
    init_n00 = new_n00
    init_n11 = new_n11


    ####### EXPERIMENT 2 ######################
    for step in range(0, int(new_n00 / step_size) + 1):

        assert new_n10 >= 0, "instance number should be bigger than 0"
        assert new_n11 >= 0, "instance number should be bigger than 0"
        assert new_n00 >= 0, "instance number should be bigger than 0"
        assert new_n01 >= 0, "instance number should be bigger than 0"

        corrected_jaccard = corrected_positive_Jaccard(new_n00, new_n10, new_n01, new_n11)
        corr_jaccard_coll_n11.append(corrected_jaccard)

        set1_size_col.append(new_n11)
        new_n11 += step_size
        new_n00 -= step_size

    ###### EXPERIMENT 2 ######################
    new_n01_2 = new_n10_2 = 50

    new_n11_2 = math.ceil((N - new_n01_2 - new_n10_2) / 2)
    new_n00_2 = N - new_n11_2 - new_n01_2 - new_n10_2
    init_n01_2 = new_n01_2
    init_n10_2 = new_n10_2

    for step in range(0, int(new_n01 / step_size) + 1):
        assert new_n10_2 >= 0, "instance number should be bigger than 0"
        assert new_n11_2 >= 0, "instance number should be bigger than 0"
        assert new_n00_2 >= 0, "instance number should be bigger than 0"
        assert new_n01_2 >= 0, "instance number should be bigger than 0"

        corrected_jaccard = corrected_positive_Jaccard(new_n00_2, new_n10_2, new_n01_2, new_n11_2)
        corr_jaccard_coll_n10.append(corrected_jaccard)

        assert set1_size_col[step] == new_n10_2, "the rate of growth is the same"
        new_n10_2 += step_size
        new_n01_2 -= step_size

    fig_text = 'Total N: ' + str(N) + '\n n01: ' + str(new_n01) + '\n n10: ' + str(new_n10) + '\n Initial n11: ' + \
               str(init_n11) + '\n Initial n00: ' + str(init_n00) + '\n n11 is increasing w.r.t. n00'
    plot_line_graph(corr_jaccard_coll_n11, 'Overlap coefficient', corr_jaccard_coll_n10, 'Positive Jaccard distance',
                    line3=None, label3=None, line4=None, label4=None, line5=None, label5=None, line6=None, label6=None,
                    x_axis_data=set1_size_col, x_label='n01/n11', results_path= res_path,
                    fig_name= 'varying_n11_n00_COMBO' + str('only_jacc'), text_string= fig_text)


def simulate_distributions_jacc_v2(res_path):
    N = 200
    step_size = 5

    jacc_scores1, steps_col1 =  simulate_jaccard_increasing_subpopulation(increasing_group='n11',
                                                                          decreasing_group='n00',
                                                                          N=200, step_size=5,
                                                                          n00=50, n01=50, n10=50, n11=50)
    label1 = 'n01 = n10; Increasing n11 w.r.t. n00'
    jacc_scores6, steps_col6 = simulate_jaccard_increasing_subpopulation(increasing_group='n10',
                                                                         decreasing_group='n01',
                                                                         N=200, step_size=5,
                                                                         n00=50, n01=50, n10=50, n11=50)
    label6 ='n11 = n00; Increasing n10 w.r.t. n01'
    jacc_scores3, steps_col3 = simulate_jaccard_increasing_subpopulation(increasing_group='n10',
                                                                         decreasing_group='n00',
                                                                         N=200, step_size=5,
                                                                         n00=50, n01=50, n10=50, n11=50)
    label3 = 'n11=n01; Increasing n10 w.r.t. n00'
    # jacc_scores4, steps_col4 = simulate_jaccard_increasing_subpopulation(increasing_group='n00',
    #                                                                      decreasing_group='n11',
    #                                                                      N=200, step_size=5,
    #                                                                      n00=50, n01=50, n10=50, n11=50)
    jacc_scores2, steps_col2 = simulate_jaccard_increasing_subpopulation(increasing_group='n11',
                                                                         decreasing_group='n10',
                                                                         N=200, step_size=5,
                                                                         n00=50, n01=50, n10=50, n11=50)
    label2 = 'n01=n00, increasing n11 w.r.t. n10'
    # jacc_scores3, steps_col2 = simulate_jaccard_increasing_subpopulation(increasing_group='n11',
    #                                                                      decreasing_group='n01',
    #                                                                      N=200, step_size=5,
    #                                                                      n00=50, n01=50, n10=50, n11=50)
    # label3 = 'n10=n00; Increasing n11 w.r.t. n01'
    assert (steps_col1==steps_col2), "steps are different in the  measurements "
    fig_text = 'Total N: ' + str(N) + '\n Initial values of subgroups: ' + str(50)
    plot_line_graph(jacc_scores1, label1, jacc_scores2, label2 ,
                    line3=jacc_scores6, label3=label6, line4=jacc_scores3, label4=label3,
                    line5=None, label5=None, line6=None, label6=None,
                    x_axis_data=steps_col1, x_label='Increasing group size', results_path= res_path,
                    fig_name= 'increasing n11' + str('only_jacc'), text_string=fig_text)
