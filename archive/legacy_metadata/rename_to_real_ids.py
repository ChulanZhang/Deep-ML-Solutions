import os
import re

# Load the real IDs from the grep output
raw_list = """
100_implement-the-softsign-activation-function
101_implement-the-grpo-objective-function
102_implement-the-swish-activation-function
103_implement-the-selu-activation-function
104_binary-classification-with-logistic-regression
105_train-softmax-regression-with-gradient-descent
106_train-logistic-regression-with-gradient-descent
107_implement-masked-self-attention
108_measure-disorder-in-apple-colors
109_implement-layer-normalization-for-sequence-data
10_calculate-covariance-matrix
110_evaluate-translation-quality-with-meteor-score
111_compute-pointwise-mutual-information
112_min-max-normalization-of-feature-values
113_implement-a-simple-residual-block-with-shortcut-co
114_implement-global-average-pooling
115_implement-batch-normalization-for-bchw-input
116_derivative-of-a-polynomial
117_compute-orthonormal-basis-for-2d-vectors
118_compute-the-cross-product-of-two-3d-vectors
119_solve-system-of-linear-equations-using-cramer-s-ru
11_solve-linear-equations-using-jacobi-method
120_bhattacharyya-distance-between-two-distributions
121_vector-element-wise-sum
122_policy-gradient-with-reinforce
123_calculate-computational-efficiency-of-moe
124_implement-the-noisy-top-k-gating-function
125_implement-a-sparse-mixture-of-experts-layer
126_implement-group-normalization
127_find-captain-redbeard-s-hidden-treasure
128_dynamic-tanh-normalization-free-transformer-activa
129_calculate-unigram-probability-from-corpus
12_singular-value-decomposition-svd
130_implement-a-simple-cnn-training-function-with-back
131_implement-efficient-sparse-window-attention
132_simulate-markov-chain-transitions
133_implement-q-learning-algorithm-for-mdps
134_compute-multi-class-cross-entropy-loss
135_implement-early-stopping-based-on-validation-loss
136_calculate-kl-divergence-between-two-multivariate-g
137_implement-a-dense-block-with-2d-convolutions
138_find-the-best-gini-based-split-for-a-binary-decisi
139_elastic-net-regression-via-gradient-descent
13_determinant-of-a-4x4-matrix-using-laplace-s-expans
140_bernoulli-naive-bayes-classifier
141_shift-and-scale-array-to-target-range
142_gridworld-policy-evaluation
143_instance-normalization-in-implementation
145_adagrad-optimizer
146_momentum-optimizer
148_adamax-optimizer
149_adadelta-optimizer
14_linear-regression-using-normal-equation
150_nesterov-accelerated-gradient-optimizer
151_dropout-layer
152_implementing-rouge-score
153_stepLR
154_exponentialLR
155_CossineAnnealingLR
156_implement_SwiGLU_activation_function
157_implement-the-bellman-equation-for-value-iteration
158_epsilon-greedy-action-selection-for-n-armed-bandit
158_incremental-mean-for-online-reward-estimation
15_linear-regression-using-gradient-descent
160_mixed_precision_training
161_exponential-weighted-average-of-rewards
16_feature-scaling-implementation
173_implement-k-nearest-neighbors
174_train-a-simple-gan-on-1d-gaussian-data
17_k-means-clustering
186_guassian_mixture_regression
187_mlops-etl-pipeline
188_gradient-checkpointing
18_implement-k-fold-cross-validation
19_principal-component-analysis-pca-implementation
1_matrix-vector-dot-product
20_decision-tree-learning
21_pegasos-kernel-svm-implementation
22_sigmoid-activation-function-understanding
23_softmax-activation-function-implementation
24_single-neuron
25_single-neuron-with-backpropagation
26_implementing-basic-autograd-operations
27_transformation-matrix-from-basis-b-to-c
28_svd-of-a-2x2-matrix-using-eigen-values-vectors
29_random-shuffle-of-dataset
2_transpose-of-a-matrix
30_batch-iterator-for-dataset
31_divide-dataset-based-on-feature-threshold
32_generate-sorted-polynomial-features
33_generate-random-subsets-of-a-dataset
34_one-hot-encoding-of-nominal-values
35_convert-vector-to-diagonal-matrix
36_calculate-accuracy-score
37_calculate-correlation-matrix
38_implement-adaboost-fit-method
39_implementation-of-log-softmax-function
3_reshape-matrix
40_implementing-a-custom-dense-layer-in-python
41_simple-convolutional-2d-layer
42_implement-relu-activation-function
43_implement-ridge-regression-loss-function
44_leaky-relu-activation-function
45_linear-kernel-function
46_implement-precision-metric
47_implement-gradient-descent-variants-with-mse-loss
48_implement-reduced-row-echelon-form-rref-function
49_implement-adam-optimization-algorithm
4_calculate-mean-by-row-or-column
50_implement-lasso-regression-using-gradient-descent
51_optimal-string-alignment-distance
52_implement-recall-metric-in-binary-classification
53_implement-self-attention-mechanism
54_implementing-a-simple-rnn
55_2d-translation-matrix-implementation
56_kl-divergence-between-two-normal-distributions
57_gauss-seidel-method-for-solving-linear-systems
58_gaussian-elimination-for-solving-linear-systems
59_implement-long-short-term-memory-lstm-network
5_scalar-multiplication-of-a-matrix
60_implement-tf-idf-term-frequency-inverse-document-f
61_implement-f-score-calculation-for-binary-classific
62_implement-a-simple-rnn-with-backpropagation-throug
63_implement-the-conjugate-gradient-method-for-solvin
64_implement-gini-impurity-calculation-for-a-set-of-c
65_implement-compressed-row-sparse-matrix-csr-format-
66_implement-orthogonal-projection-of-a-vector-onto-a
67_implement-compressed-column-sparse-matrix-format-c
68_find-the-image-of-a-matrix-using-row-echelon-form
69_calculate-r-squared-for-regression-analysis
6_calculate-eigenvalues-of-a-matrix
70_calculate-image-brightness
71_calculate-root-mean-square-error-rmse
72_calculate-jaccard-index-for-binary-classification
73_calculate-dice-score-for-classification
74_create-composite-hypervector-for-a-dataset-row
75_generate-a-confusion-matrix-for-binary-classificat
76_calculate-cosine-similarity-between-vectors
77_calculate-performance-metrics-for-a-classification
78_descriptive-statistics-calculator
79_binomial-distribution-probability
7_matrix-transformation
80_normal-distribution-pdf-calculator
81_poisson-distribution-probability-calculator
82_grayscale-image-contrast-calculator
83_dot-product-calculator
84_phi-transformation-for-polynomial-features
85_positional-encoding-calculator
86_detect-overfitting-or-underfitting
87_adam-optimizer
88_gpt-2-text-generation
89_the-pattern-weaver-s-code
8_calculate-2x2-matrix-inverse
90_bm25-ranking
91_calculate-f1-score-from-predicted-and-true-labels
92_linear-regression-power-grid-optimization
93_calculate-mean-absolute-error-mae
94_implement-multi-head-attention
95_calculate-the-phi-coefficient
96_implement-the-hard-sigmoid-activation-function
97_implement-the-elu-activation-function
98_implement-the-prelu-activation-function
99_implement-the-softplus-activation-function
9_matrix-times-matrix
"""

def normalize_name(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

name_to_id = {}
for line in raw_list.split('\n'):
    line = line.strip()
    if not line: continue
    parts = line.split('_', 1)
    if len(parts) == 2:
        qid, slug = parts
        name_to_id[normalize_name(slug)] = qid

def rename_in_dir(directory):
    if not os.path.exists(directory): return
    for f in os.listdir(directory):
        if not f.endswith(".py"): continue
        
        # Extract the original name without the fake ID
        match = re.match(r"^\d+-(.*)\.py$", f)
        if not match: continue
        title = match.group(1)
        
        norm_title = normalize_name(title)
        
        real_id = None
        # Try direct match
        if norm_title in name_to_id:
            real_id = name_to_id[norm_title]
        else:
            # Try partial matching (substring)
            for k, v in name_to_id.items():
                if norm_title.startswith(k) or k.startswith(norm_title):
                    real_id = v
                    break
                    
        if real_id:
            new_name = f"{real_id}-{title}.py"
            if new_name != f:
                os.rename(os.path.join(directory, f), os.path.join(directory, new_name))
        else:
            print(f"Could not find real ID for {title}")

rename_in_dir("numpy")
rename_in_dir("pytorch")
print("Finished renaming!")
