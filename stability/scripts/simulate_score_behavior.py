from stability.behavior_stability_score_visualizations import simulate_distributions_n01_n10_extreme, \
    simulate_distributions_n01_n10_equal_distribution, simulate_distributions_n01_n10, simulate_distributions_n11_n00, \
    simulate_distributions_corollary5, simulate_distributions_monotonic, simulate_distributions_jacc_n11_n00, \
    simulate_distributions_jacc_v2, corrected_positive_Jaccard

results_path = 'C:/Users/s161590/Documents/Project_li/stability/'
# C:\Users\s161590\Documents\Project_li\stability

# simulate_distributions_n01_n10_extreme(results_path)
#
# simulate_distributions_n01_n10_equal_distribution(results_path)
#
# simulate_distributions_n01_n10(results_path)
# simulate_distributions_n11_n00(results_path)
#
# simulate_distributions_corollary5(results_path)
# simulate_distributions_monotonic(results_path)

# simulate_distributions_jacc_v2(results_path)

#### SOURCE: "High agreement but low kappa: II. Resolving the paradoxes" Cicchetti, Feinstein
def compute_positive_agreement_ratio(positive_overlap, observer1_positive, observer2_positive):
    return (2*positive_overlap)/(observer1_positive + observer2_positive)


def compute_negative_agreement_ratio(negative_overlap, observer1_negative, observer2_negative):
    return (2*negative_overlap)/(observer1_negative + observer2_negative)


def compute_total_agreement_ratio(neg_overlap, pos_overlap, N):
    return (pos_overlap+neg_overlap)/N


def compute_f1_f2(observer1_pos, observer1_neg):
    return observer1_pos - observer1_neg

#### experiment 1 #####
a = 40
b = 9
c = 6
d = 45
ck = corrected_positive_Jaccard(d, b, c, a)
print(ck)
print(compute_total_agreement_ratio(d,a, a+b+c+d))
print(compute_positive_agreement_ratio(a, a+c, a+b))
print(compute_negative_agreement_ratio(d, b+d, c+d))

print(compute_f1_f2(a+c,  b+d))


#### experiment 1 #####
a = 80
b = 10
c = 5
d = 5
ck = corrected_positive_Jaccard(d, b, c, a)
print(ck)

#### experiment 1 #####
a = 40
b = 20
c = 20
d = 20
ck = corrected_positive_Jaccard(d, b, c, a)
print("#### experiment 1 #####")
print(ck)

#### experiment 1 #####
a = 60
b = 20
c = 20
d = 0
ck = corrected_positive_Jaccard(d, b, c, a)
print("#### experiment 1.1 #####")
print(ck)


#### experiment 1 #####
a = 40
b = 15
c = 25
d = 20
ck = corrected_positive_Jaccard(d, b, c, a)
print("#### experiment 2 #####")
print(ck)


#### experiment 1 #####
a = 40
b = 35
c = 5
d = 20
ck = corrected_positive_Jaccard(d, b, c, a)
print("#### experiment 3 #####")
print(ck)

#### experiment 1 #####
a = 60
b = 35
c = 5
d = 0
ck = corrected_positive_Jaccard(d, b, c, a)
print("#### experiment 32 #####")
print(ck)


#### experiment 1 #####
a = 40
b = 10
c = 10
d = 40
ck = corrected_positive_Jaccard(d, b, c, a)
print("#### experiment prevalence 1 #####")
print(ck)

#### experiment 1 #####
a = 40
b = 30
c = 30
d = 0
ck = corrected_positive_Jaccard(d, b, c, a)
print("#### experiment prevalence 2 #####")
print(ck)

#### experiment 1 #####
a = 70
b = 10
c = 10
d = 10
ck = corrected_positive_Jaccard(d, b, c, a)
print("#### experiment prevalence 3 #####")
print(ck)

#### experiment 1 #####
a = 45
b = 35
c = 5
d = 15
ck = corrected_positive_Jaccard(d, b, c, a)
print(ck)


a = 45
b = 0
c = 0
d = 15
ck = corrected_positive_Jaccard(d, b, c, a)
print(ck)



a = 47
b = 0
c = 0
d = 10
ck = corrected_positive_Jaccard(d, b, c, a)
print(ck)

