# Deep-ML 题号索引：PyTorch 复习版

核对日期：2026-10-15（America/Indiana/Indianapolis）。题号与题名来自官网公开 API，来源和抓取时间见 audit/public_examples.json。

**每个官网 ID 只出现一行。** 40 道核心题已重写；其他题仅保留归档映射，不能当作已修复答案。原 248 个文件涉及 218 个官网 ID；新增正确的 #125 Sparse MoE 后，共 219 个 ID。

原 124-Calculate Number of Parameters... 本次规范到 #291；官网另有同名有效题 #371，无法仅凭题名唯一确定历史来源，映射中保留说明。

| 官网题号 | 官网题名 | 本轮实现 | 旧稿数 |
|---|---|---|---|
| [#1](https://www.deep-ml.com/problems/1) | Matrix-Vector Dot Product | 未修复；见归档映射 | 1 |
| [#2](https://www.deep-ml.com/problems/2) | Transpose of a Matrix | 未修复；见归档映射 | 1 |
| [#3](https://www.deep-ml.com/problems/3) | Reshape Matrix | 未修复；见归档映射 | 1 |
| [#4](https://www.deep-ml.com/problems/4) | Calculate Mean by Row or Column | 未修复；见归档映射 | 1 |
| [#5](https://www.deep-ml.com/problems/5) | Scalar Multiplication of a Matrix | 未修复；见归档映射 | 1 |
| [#6](https://www.deep-ml.com/problems/6) | Calculate Eigenvalues of a Matrix | 未修复；见归档映射 | 1 |
| [#7](https://www.deep-ml.com/problems/7) | Matrix Transformation | 未修复；见归档映射 | 1 |
| [#8](https://www.deep-ml.com/problems/8) | Calculate 2x2 Matrix Inverse | 未修复；见归档映射 | 1 |
| [#9](https://www.deep-ml.com/problems/9) | Matrix times Matrix | 未修复；见归档映射 | 1 |
| [#10](https://www.deep-ml.com/problems/10) | Calculate Covariance Matrix | 未修复；见归档映射 | 1 |
| [#11](https://www.deep-ml.com/problems/11) | Solve Linear Equations using Jacobi Method | 未修复；见归档映射 | 1 |
| [#12](https://www.deep-ml.com/problems/12) | Singular Value Decomposition (SVD) of 2x2 Matrix | 未修复；见归档映射 | 1 |
| [#13](https://www.deep-ml.com/problems/13) | Determinant of a 4x4 Matrix using Laplace's Expansion (hard) | 未修复；见归档映射 | 1 |
| [#14](https://www.deep-ml.com/problems/14) | Linear Regression Using Normal Equation | 未修复；见归档映射 | 1 |
| [#15](https://www.deep-ml.com/problems/15) | Linear Regression Using Gradient Descent | 未修复；见归档映射 | 1 |
| [#17](https://www.deep-ml.com/problems/17) | K-Means Clustering | 未修复；见归档映射 | 1 |
| [#18](https://www.deep-ml.com/problems/18) | Implement K-Fold Cross-Validation | 未修复；见归档映射 | 1 |
| [#19](https://www.deep-ml.com/problems/19) | Principal Component Analysis (PCA) Implementation | 未修复；见归档映射 | 1 |
| [#21](https://www.deep-ml.com/problems/21) | Pegasos Kernel SVM Implementation | 未修复；见归档映射 | 1 |
| [#22](https://www.deep-ml.com/problems/22) | Sigmoid Activation Function Understanding | 未修复；见归档映射 | 1 |
| [#23](https://www.deep-ml.com/problems/23) | Softmax Activation Function Implementation | [代码](pytorch/0023_softmax.py) | 1 |
| [#24](https://www.deep-ml.com/problems/24) | Single Neuron | 未修复；见归档映射 | 1 |
| [#25](https://www.deep-ml.com/problems/25) | Single Neuron with Backpropagation | 未修复；见归档映射 | 1 |
| [#26](https://www.deep-ml.com/problems/26) | Implementing Basic Autograd Operations | 未修复；见归档映射 | 1 |
| [#27](https://www.deep-ml.com/problems/27) | Transformation Matrix from Basis B to C | 未修复；见归档映射 | 1 |
| [#35](https://www.deep-ml.com/problems/35) | Convert Vector to Diagonal Matrix | 未修复；见归档映射 | 1 |
| [#36](https://www.deep-ml.com/problems/36) | Calculate Accuracy Score | 未修复；见归档映射 | 1 |
| [#38](https://www.deep-ml.com/problems/38) | Implement AdaBoost Fit Method | 未修复；见归档映射 | 1 |
| [#39](https://www.deep-ml.com/problems/39) | Implementation of Log Softmax Function | [代码](pytorch/0039_log_softmax.py) | 1 |
| [#41](https://www.deep-ml.com/problems/41) | Simple Convolutional 2D Layer | 未修复；见归档映射 | 1 |
| [#42](https://www.deep-ml.com/problems/42) | Implement ReLU Activation Function | 未修复；见归档映射 | 1 |
| [#44](https://www.deep-ml.com/problems/44) | Leaky ReLU Activation Function | 未修复；见归档映射 | 1 |
| [#46](https://www.deep-ml.com/problems/46) | Implement Precision Metric | 未修复；见归档映射 | 1 |
| [#47](https://www.deep-ml.com/problems/47) | Implement Gradient Descent Variants with MSE Loss | 未修复；见归档映射 | 1 |
| [#48](https://www.deep-ml.com/problems/48) | Implement Reduced Row Echelon Form (RREF) Function | 未修复；见归档映射 | 1 |
| [#49](https://www.deep-ml.com/problems/49) | Implement Adam Optimization Algorithm | [代码](pytorch/0049_adam.py) | 1 |
| [#50](https://www.deep-ml.com/problems/50) | Implement Lasso Regression using ISTA | 未修复；见归档映射 | 1 |
| [#52](https://www.deep-ml.com/problems/52) | Implement Recall Metric in Binary Classification | 未修复；见归档映射 | 1 |
| [#53](https://www.deep-ml.com/problems/53) | Implement Self-Attention Mechanism | [代码](pytorch/0053_self_attention.py) | 1 |
| [#54](https://www.deep-ml.com/problems/54) | Implementing a Simple RNN | 未修复；见归档映射 | 1 |
| [#55](https://www.deep-ml.com/problems/55) | 2D Translation Matrix Implementation | 未修复；见归档映射 | 1 |
| [#56](https://www.deep-ml.com/problems/56) | KL Divergence Between Two Normal Distributions | 未修复；见归档映射 | 1 |
| [#57](https://www.deep-ml.com/problems/57) | Gauss-Seidel Method for Solving Linear Systems | 未修复；见归档映射 | 1 |
| [#59](https://www.deep-ml.com/problems/59) | Implement Long Short-Term Memory (LSTM) Network | 未修复；见归档映射 | 2 |
| [#61](https://www.deep-ml.com/problems/61) | Implement F-Score Calculation for Binary Classification | 未修复；见归档映射 | 1 |
| [#62](https://www.deep-ml.com/problems/62) | Implement a Simple RNN with Backpropagation Through Time (BPTT) | 未修复；见归档映射 | 1 |
| [#65](https://www.deep-ml.com/problems/65) | Implement Compressed Row Sparse Matrix (CSR) Format Conversion | 未修复；见归档映射 | 1 |
| [#66](https://www.deep-ml.com/problems/66) | Implement Orthogonal Projection of a Vector onto a Line | 未修复；见归档映射 | 1 |
| [#67](https://www.deep-ml.com/problems/67) | Implement Compressed Column Sparse Matrix Format (CSC) | 未修复；见归档映射 | 1 |
| [#68](https://www.deep-ml.com/problems/68) | Find the column space of a matrix | 未修复；见归档映射 | 1 |
| [#69](https://www.deep-ml.com/problems/69) | Calculate R-squared for Regression Analysis | 未修复；见归档映射 | 1 |
| [#70](https://www.deep-ml.com/problems/70) | Calculate Image Brightness | 未修复；见归档映射 | 1 |
| [#71](https://www.deep-ml.com/problems/71) | Calculate Root Mean Square Error (RMSE) | 未修复；见归档映射 | 1 |
| [#72](https://www.deep-ml.com/problems/72) | Calculate Jaccard Index for Binary Classification | 未修复；见归档映射 | 1 |
| [#75](https://www.deep-ml.com/problems/75) | Generate a Confusion Matrix for Binary Classification | 未修复；见归档映射 | 1 |
| [#76](https://www.deep-ml.com/problems/76) | Calculate Cosine Similarity Between Vectors | 未修复；见归档映射 | 1 |
| [#77](https://www.deep-ml.com/problems/77) | Calculate Performance Metrics for a Classification Model | 未修复；见归档映射 | 1 |
| [#78](https://www.deep-ml.com/problems/78) | Descriptive Statistics Calculator | 未修复；见归档映射 | 1 |
| [#79](https://www.deep-ml.com/problems/79) | Binomial Distribution Probability | 未修复；见归档映射 | 1 |
| [#80](https://www.deep-ml.com/problems/80) | Normal Distribution PDF Calculator | 未修复；见归档映射 | 1 |
| [#81](https://www.deep-ml.com/problems/81) | Poisson Distribution Probability Calculator | 未修复；见归档映射 | 1 |
| [#82](https://www.deep-ml.com/problems/82) | Grayscale Image Contrast Calculator | 未修复；见归档映射 | 1 |
| [#83](https://www.deep-ml.com/problems/83) | Dot Product Calculator | 未修复；见归档映射 | 1 |
| [#85](https://www.deep-ml.com/problems/85) | Positional Encoding Calculator | [代码](pytorch/0085_sinusoidal_position.py) | 1 |
| [#88](https://www.deep-ml.com/problems/88) | GPT-2 Text Generation | 未修复；见归档映射 | 1 |
| [#89](https://www.deep-ml.com/problems/89) | The Pattern Weaver's Code | 未修复；见归档映射 | 1 |
| [#93](https://www.deep-ml.com/problems/93) | Calculate Mean Absolute Error (MAE) | 未修复；见归档映射 | 1 |
| [#94](https://www.deep-ml.com/problems/94) | Implement Multi-Head Attention | [代码](pytorch/0094_multi_head_attention.py) | 1 |
| [#98](https://www.deep-ml.com/problems/98) | Implement PReLU Forward and Backward Pass | 未修复；见归档映射 | 1 |
| [#101](https://www.deep-ml.com/problems/101) | Implement the GRPO Objective Function | [代码](pytorch/0101_grpo_objective.py) | 1 |
| [#104](https://www.deep-ml.com/problems/104) | Binary Classification with Logistic Regression | 未修复；见归档映射 | 1 |
| [#107](https://www.deep-ml.com/problems/107) | Implement Masked Self-Attention | [代码](pytorch/0107_masked_attention.py) | 1 |
| [#109](https://www.deep-ml.com/problems/109) | Implement Layer Normalization for Sequence Data | [代码](pytorch/0109_layer_norm.py) | 1 |
| [#113](https://www.deep-ml.com/problems/113) | Implement a Simple Residual Block with Shortcut Connection | 未修复；见归档映射 | 1 |
| [#114](https://www.deep-ml.com/problems/114) | Implement Global Average Pooling | 未修复；见归档映射 | 1 |
| [#115](https://www.deep-ml.com/problems/115) | Implement Batch Normalization for BCHW Input | 未修复；见归档映射 | 1 |
| [#124](https://www.deep-ml.com/problems/124) | Implement the Noisy Top-K Gating Function | [代码](pytorch/0124_noisy_topk_gating.py) | 1 |
| [#125](https://www.deep-ml.com/problems/125) | Implement a Sparse Mixture of Experts Layer | [代码](pytorch/0125_sparse_moe.py) | 0 |
| [#126](https://www.deep-ml.com/problems/126) | Implement Group Normalization | 未修复；见归档映射 | 1 |
| [#137](https://www.deep-ml.com/problems/137) | Implement a Dense Block with 2D Convolutions | 未修复；见归档映射 | 1 |
| [#146](https://www.deep-ml.com/problems/146) | Momentum Optimizer | 未修复；见归档映射 | 1 |
| [#147](https://www.deep-ml.com/problems/147) | GeLU Activation Function | 未修复；见归档映射 | 1 |
| [#148](https://www.deep-ml.com/problems/148) | Adamax Optimizer | 未修复；见归档映射 | 1 |
| [#149](https://www.deep-ml.com/problems/149) | Adadelta Optimizer | 未修复；见归档映射 | 1 |
| [#151](https://www.deep-ml.com/problems/151) | Dropout Layer | [代码](pytorch/0151_dropout.py) | 1 |
| [#156](https://www.deep-ml.com/problems/156) | Implement SwiGLU activation function | [代码](pytorch/0156_swiglu.py) | 2 |
| [#157](https://www.deep-ml.com/problems/157) | Implement the Bellman Equation for Value Iteration | 未修复；见归档映射 | 1 |
| [#158](https://www.deep-ml.com/problems/158) | Epsilon-Greedy Action Selection for n-Armed Bandit | 未修复；见归档映射 | 1 |
| [#159](https://www.deep-ml.com/problems/159) | Incremental Mean for Online Reward Estimation | 未修复；见归档映射 | 1 |
| [#161](https://www.deep-ml.com/problems/161) | Exponential Weighted Average of Rewards | 未修复；见归档映射 | 1 |
| [#162](https://www.deep-ml.com/problems/162) | Upper Confidence Bound (UCB) Action Selection | 未修复；见归档映射 | 1 |
| [#163](https://www.deep-ml.com/problems/163) | Gradient Bandit Action Selection | 未修复；见归档映射 | 1 |
| [#164](https://www.deep-ml.com/problems/164) | Gambler's Problem: Value Iteration | 未修复；见归档映射 | 1 |
| [#165](https://www.deep-ml.com/problems/165) | Compute Discounted Return | 未修复；见归档映射 | 1 |
| [#166](https://www.deep-ml.com/problems/166) | Evaluate Expected Value in a Markov Decision Process | 未修复；见归档映射 | 1 |
| [#167](https://www.deep-ml.com/problems/167) | Calculate the Discounted Return for a Given Trajectory | 未修复；见归档映射 | 1 |
| [#172](https://www.deep-ml.com/problems/172) | Muon Optimizer Update with Newton-Schulz Iteration | 未修复；见归档映射 | 1 |
| [#174](https://www.deep-ml.com/problems/174) | Train a Simple GAN on 1D Gaussian Data | 未修复；见归档映射 | 1 |
| [#177](https://www.deep-ml.com/problems/177) | Implement MuonClip (qk-clip) for Stabilizing Attention | 未修复；见归档映射 | 1 |
| [#178](https://www.deep-ml.com/problems/178) | Implement Position-wise Feed-Forward Block with Residual and Dropout | [代码](pytorch/0178_ffn_block.py) | 2 |
| [#185](https://www.deep-ml.com/problems/185) | Optical Flow EPE with Masks (OmniWorld-style metric) | 未修复；见归档映射 | 1 |
| [#189](https://www.deep-ml.com/problems/189) | Implement Local Response Normalization (LRN) | 未修复；见归档映射 | 2 |
| [#190](https://www.deep-ml.com/problems/190) | Overlapping Max Pooling | 未修复；见归档映射 | 2 |
| [#191](https://www.deep-ml.com/problems/191) | PCA Color Augmentation | 未修复；见归档映射 | 2 |
| [#208](https://www.deep-ml.com/problems/208) | Flash Attention v1 - Forward Pass | [代码](pytorch/0208_flash_attention_reference.py) | 2 |
| [#216](https://www.deep-ml.com/problems/216) | Thanksgiving Feast Predictor: Softmax for Dish Selection | 未修复；见归档映射 | 1 |
| [#217](https://www.deep-ml.com/problems/217) | Derivatives of Activation Functions | 未修复；见归档映射 | 1 |
| [#218](https://www.deep-ml.com/problems/218) | Compute the Hessian Matrix | 未修复；见归档映射 | 1 |
| [#222](https://www.deep-ml.com/problems/222) | LoRA: Low-Rank Adaptation Forward Pass | [代码](pytorch/0222_lora_forward.py) | 2 |
| [#223](https://www.deep-ml.com/problems/223) | QLoRA: Quantized Low-Rank Adaptation Forward Pass | [代码](pytorch/0223_qlora_forward.py) | 2 |
| [#224](https://www.deep-ml.com/problems/224) | Group Relative Advantage for GRPO | [代码](pytorch/0224_group_relative_advantage.py) | 1 |
| [#225](https://www.deep-ml.com/problems/225) | KL Divergence Estimator for GRPO | [代码](pytorch/0225_kl_estimator.py) | 1 |
| [#229](https://www.deep-ml.com/problems/229) | Sparse MoE Top-K Routing | [代码](pytorch/0229_moe_topk_routing.py) | 1 |
| [#230](https://www.deep-ml.com/problems/230) | 3D CNN Forward Pass Implementation | 未修复；见归档映射 | 1 |
| [#231](https://www.deep-ml.com/problems/231) | Temperature Decay Scheduler | 未修复；见归档映射 | 1 |
| [#232](https://www.deep-ml.com/problems/232) | PTX Loss for Catastrophic Forgetting Prevention (RLHF) | 未修复；见归档映射 | 1 |
| [#233](https://www.deep-ml.com/problems/233) | Inference Head Pruning for Transformers | 未修复；见归档映射 | 1 |
| [#234](https://www.deep-ml.com/problems/234) | Block-wise FP8 Quantization | 未修复；见归档映射 | 2 |
| [#235](https://www.deep-ml.com/problems/235) | Implement the SGTM Parameter Update Step | 未修复；见归档映射 | 1 |
| [#236](https://www.deep-ml.com/problems/236) | Mean Ablation for Circuit Discovery | 未修复；见归档映射 | 1 |
| [#237](https://www.deep-ml.com/problems/237) | Convert RGB Image to Grayscale | 未修复；见归档映射 | 1 |
| [#238](https://www.deep-ml.com/problems/238) | Flip an Image Horizontally or Vertically | 未修复；见归档映射 | 1 |
| [#239](https://www.deep-ml.com/problems/239) | Apply Zero Padding to an Image | 未修复；见归档映射 | 1 |
| [#240](https://www.deep-ml.com/problems/240) | Bilinear Image Resizing | 未修复；见归档映射 | 1 |
| [#241](https://www.deep-ml.com/problems/241) | Sobel Edge Detection | 未修复；见归档映射 | 1 |
| [#242](https://www.deep-ml.com/problems/242) | Non-Maximum Suppression for Object Detection | 未修复；见归档映射 | 1 |
| [#262](https://www.deep-ml.com/problems/262) | Implement the Mish Activation Function | 未修复；见归档映射 | 1 |
| [#263](https://www.deep-ml.com/problems/263) | Implement Binary Cross-Entropy Loss | [代码](pytorch/0263_binary_cross_entropy.py) | 1 |
| [#264](https://www.deep-ml.com/problems/264) | Implement the Tanh Activation Function | 未修复；见归档映射 | 1 |
| [#265](https://www.deep-ml.com/problems/265) | Implement 2D Average Pooling | 未修复；见归档映射 | 1 |
| [#266](https://www.deep-ml.com/problems/266) | Implement the Hardtanh Activation Function | 未修复；见归档映射 | 1 |
| [#267](https://www.deep-ml.com/problems/267) | Implement Neural Memory Update with Surprise and Momentum | 未修复；见归档映射 | 1 |
| [#271](https://www.deep-ml.com/problems/271) | Implement Gated Attention | 未修复；见归档映射 | 1 |
| [#287](https://www.deep-ml.com/problems/287) | Implement GRU Cell | 未修复；见归档映射 | 1 |
| [#289](https://www.deep-ml.com/problems/289) | Implement Xavier/Glorot Weight Initialization | 未修复；见归档映射 | 1 |
| [#290](https://www.deep-ml.com/problems/290) | Implement He Weight Initialization | 未修复；见归档映射 | 1 |
| [#291](https://www.deep-ml.com/problems/291) | Calculate Number of Parameters in Neural Network | 未修复；见归档映射 | 2 |
| [#292](https://www.deep-ml.com/problems/292) | Implement Gradient Clipping by Value | 未修复；见归档映射 | 1 |
| [#294](https://www.deep-ml.com/problems/294) | Implement INT8 Quantization | [代码](pytorch/0294_int8_quantization.py) | 2 |
| [#298](https://www.deep-ml.com/problems/298) | Implement mHC Forward Pass | 未修复；见归档映射 | 1 |
| [#302](https://www.deep-ml.com/problems/302) | Diffusion Reconstruction Loss | 未修复；见归档映射 | 1 |
| [#303](https://www.deep-ml.com/problems/303) | Forward Diffusion Process | 未修复；见归档映射 | 1 |
| [#304](https://www.deep-ml.com/problems/304) | Forward & Backward Diffusion Process | 未修复；见归档映射 | 1 |
| [#316](https://www.deep-ml.com/problems/316) | MMLU Log-Probability Scoring | 未修复；见归档映射 | 1 |
| [#317](https://www.deep-ml.com/problems/317) | Rubric-Based LLM Judge Evaluation | 未修复；见归档映射 | 1 |
| [#318](https://www.deep-ml.com/problems/318) | Boxed Answer Extraction for Math Benchmarks | 未修复；见归档映射 | 1 |
| [#319](https://www.deep-ml.com/problems/319) | Math Answer Verification with Equivalence Checking | 未修复；见归档映射 | 1 |
| [#320](https://www.deep-ml.com/problems/320) | Calculate Perplexity for Language Models | [代码](pytorch/0320_perplexity.py) | 1 |
| [#323](https://www.deep-ml.com/problems/323) | Pairwise Preference Judge for LLM Comparison | 未修复；见归档映射 | 1 |
| [#324](https://www.deep-ml.com/problems/324) | Code Execution Verifier for Programming Benchmarks | 未修复；见归档映射 | 1 |
| [#327](https://www.deep-ml.com/problems/327) | Engram Context-Aware Gating | 未修复；见归档映射 | 1 |
| [#335](https://www.deep-ml.com/problems/335) | Train a Paris-Style Decentralized Expert Model | 未修复；见归档映射 | 1 |
| [#358](https://www.deep-ml.com/problems/358) | Implement Core MDN Residualization | 未修复；见归档映射 | 2 |
| [#359](https://www.deep-ml.com/problems/359) | Distance Correlation for Measuring Metadata Dependence | 未修复；见归档映射 | 1 |
| [#360](https://www.deep-ml.com/problems/360) | MDN with Label Collinearity Control | 未修复；见归档映射 | 2 |
| [#369](https://www.deep-ml.com/problems/369) | Implement Xavier/Glorot Weight Initialization | 未修复；见归档映射 | 1 |
| [#370](https://www.deep-ml.com/problems/370) | Implement He Weight Initialization for Neural Networks | 未修复；见归档映射 | 1 |
| [#371](https://www.deep-ml.com/problems/371) | Calculate Number of Parameters in Neural Network | 未修复；见归档映射 | 1 |
| [#372](https://www.deep-ml.com/problems/372) | Implement RMSNorm (Root Mean Square Layer Normalization) | [代码](pytorch/0372_rmsnorm.py) | 2 |
| [#373](https://www.deep-ml.com/problems/373) | Implement the Square ReLU Activation Function | 未修复；见归档映射 | 2 |
| [#374](https://www.deep-ml.com/problems/374) | Character-Level Tokenizer (stoi/itos/BOS) | [代码](pytorch/0374_char_tokenizer.py) | 2 |
| [#375](https://www.deep-ml.com/problems/375) | Learned Positional Embeddings | [代码](pytorch/0375_learned_position.py) | 2 |
| [#376](https://www.deep-ml.com/problems/376) | KV Cache for Efficient Autoregressive Attention | [代码](pytorch/0376_kv_cache_step.py) | 2 |
| [#377](https://www.deep-ml.com/problems/377) | Linear Learning Rate Decay | 未修复；见归档映射 | 1 |
| [#378](https://www.deep-ml.com/problems/378) | Temperature Sampling | 未修复；见归档映射 | 1 |
| [#381](https://www.deep-ml.com/problems/381) | Rotary Positional Embeddings (RoPE) | [代码](pytorch/0381_rope.py) | 2 |
| [#382](https://www.deep-ml.com/problems/382) | Direct Preference Optimization (DPO) Loss | [代码](pytorch/0382_dpo_loss.py) | 1 |
| [#383](https://www.deep-ml.com/problems/383) | Top-p (Nucleus) Sampling | [代码](pytorch/0383_top_p.py) | 1 |
| [#384](https://www.deep-ml.com/problems/384) | Contrastive Loss (InfoNCE / SimCLR-style) | [代码](pytorch/0384_infonce.py) | 1 |
| [#386](https://www.deep-ml.com/problems/386) | Spectral Normalization | 未修复；见归档映射 | 1 |
| [#387](https://www.deep-ml.com/problems/387) | Triplet Margin Loss | 未修复；见归档映射 | 1 |
| [#388](https://www.deep-ml.com/problems/388) | Sliding Window Attention | [代码](pytorch/0388_sliding_window_attention.py) | 2 |
| [#390](https://www.deep-ml.com/problems/390) | Implement Multiquery Attention (MQA) | [代码](pytorch/0390_multiquery_attention.py) | 1 |
| [#391](https://www.deep-ml.com/problems/391) | Implement Grouped Query Attention (GQA) | [代码](pytorch/0391_grouped_query_attention.py) | 2 |
| [#393](https://www.deep-ml.com/problems/393) | Implement Variational Autoencoder (VAE) Loss (ELBO) | 未修复；见归档映射 | 1 |
| [#394](https://www.deep-ml.com/problems/394) | Implement Speculative Decoding Verification | 未修复；见归档映射 | 1 |
| [#395](https://www.deep-ml.com/problems/395) | DDPM Noise Schedule (Linear Beta Schedule) | 未修复；见归档映射 | 1 |
| [#396](https://www.deep-ml.com/problems/396) | Implement DDPM Reverse Sampling Step | 未修复；见归档映射 | 1 |
| [#397](https://www.deep-ml.com/problems/397) | Classifier-Free Guidance for Conditional Diffusion | 未修复；见归档映射 | 1 |
| [#398](https://www.deep-ml.com/problems/398) | DDIM Deterministic Sampling Step | 未修复；见归档映射 | 1 |
| [#399](https://www.deep-ml.com/problems/399) | Diffusion Model U-Net Time Embedding | 未修复；见归档映射 | 1 |
| [#400](https://www.deep-ml.com/problems/400) | Noise Prediction Loss for Diffusion Training | 未修复；见归档映射 | 1 |
| [#401](https://www.deep-ml.com/problems/401) | Exponential Moving Average (EMA) for Diffusion Model Weights | 未修复；见归档映射 | 1 |
| [#402](https://www.deep-ml.com/problems/402) | Latent Diffusion Encoding and Decoding | 未修复；见归档映射 | 1 |
| [#403](https://www.deep-ml.com/problems/403) | Diffusion Cosine Noise Schedule | 未修复；见归档映射 | 1 |
| [#404](https://www.deep-ml.com/problems/404) | Implement Score Matching for Score-Based Diffusion | 未修复；见归档映射 | 1 |
| [#406](https://www.deep-ml.com/problems/406) | NoPE (No Positional Embedding) with iRoPE Attention | 未修复；见归档映射 | 1 |
| [#407](https://www.deep-ml.com/problems/407) | QK-Norm (Query-Key Normalization) | [代码](pytorch/0407_qk_norm_attention.py) | 1 |
| [#408](https://www.deep-ml.com/problems/408) | Pre-Norm vs Post-Norm Transformer Block | [代码](pytorch/0408_pre_post_norm.py) | 2 |
| [#409](https://www.deep-ml.com/problems/409) | MoE with Shared Expert Forward Pass | [代码](pytorch/0409_moe_shared_expert.py) | 1 |
| [#410](https://www.deep-ml.com/problems/410) | Speculative Decoding End-to-End Simulation | 未修复；见归档映射 | 1 |
| [#414](https://www.deep-ml.com/problems/414) | Compute Arithmetic Intensity and Classify Bottleneck | 未修复；见归档映射 | 1 |
| [#415](https://www.deep-ml.com/problems/415) | Roofline Model Analysis for GPU Operations | 未修复；见归档映射 | 2 |
| [#416](https://www.deep-ml.com/problems/416) | Compute Attention Memory Traffic and FLOPs | 未修复；见归档映射 | 1 |
| [#417](https://www.deep-ml.com/problems/417) | Classify LLM Prefill vs Decode as Compute-Bound or Memory-Bound | 未修复；见归档映射 | 1 |
| [#418](https://www.deep-ml.com/problems/418) | Estimate KV Cache Size from Model Config | [代码](pytorch/0418_kv_cache_size.py) | 1 |
| [#419](https://www.deep-ml.com/problems/419) | Combined Token Sampling Pipeline (Temperature + Top-k + Top-p) | [代码](pytorch/0419_combined_sampling.py) | 1 |
| [#420](https://www.deep-ml.com/problems/420) | Latent Space Patchification for Diffusion Transformers | 未修复；见归档映射 | 1 |
| [#421](https://www.deep-ml.com/problems/421) | GPU Ops:Byte Ratio Calculation from Spec Sheet | 未修复；见归档映射 | 1 |
| [#422](https://www.deep-ml.com/problems/422) | Multi-GPU Communication Overhead (NVLink vs InfiniBand) | 未修复；见归档映射 | 1 |
| [#423](https://www.deep-ml.com/problems/423) | Multi-Instance GPU (MIG) Resource Allocation | 未修复；见归档映射 | 1 |
| [#424](https://www.deep-ml.com/problems/424) | Kernel Fusion Memory Savings Calculator | 未修复；见归档映射 | 1 |
| [#425](https://www.deep-ml.com/problems/425) | Continuous Batching (In-Flight Batching) Simulator | 未修复；见归档映射 | 2 |
| [#426](https://www.deep-ml.com/problems/426) | Post-Training Quantization with Per-Channel Scale Factors | 未修复；见归档映射 | 1 |
| [#427](https://www.deep-ml.com/problems/427) | FP4 Quantization with Microscaling (MXFP4) | 未修复；见归档映射 | 2 |
| [#428](https://www.deep-ml.com/problems/428) | Number Format Precision Comparison (FP16 vs BF16 vs FP8 vs FP4) | 未修复；见归档映射 | 1 |
| [#429](https://www.deep-ml.com/problems/429) | Quantization Quality Check via Perplexity Delta | [代码](pytorch/0429_quantization_quality.py) | 1 |
| [#431](https://www.deep-ml.com/problems/431) | EAGLE-Style Draft Model from Hidden States | 未修复；见归档映射 | 1 |
| [#435](https://www.deep-ml.com/problems/435) | KV Cache Memory Budget and Eviction Policy | 未修复；见归档映射 | 1 |
| [#438](https://www.deep-ml.com/problems/438) | Tensor Parallelism All-Reduce Communication Cost | 未修复；见归档映射 | 1 |
| [#442](https://www.deep-ml.com/problems/442) | VLM Visual Token Count from Image Resolution and Patch Size | 未修复；见归档映射 | 1 |
| [#446](https://www.deep-ml.com/problems/446) | Video Generation Latent Space Memory Estimation | 未修复；见归档映射 | 1 |
| [#447](https://www.deep-ml.com/problems/447) | Classifier-Free Guidance Skip Speedup Calculator | 未修复；见归档映射 | 1 |
| [#448](https://www.deep-ml.com/problems/448) | Context Parallelism with Ring Attention for Video Models | 未修复；见归档映射 | 1 |
| [#453](https://www.deep-ml.com/problems/453) | Implement Graph Convolution Network (GCN) Layer | 未修复；见归档映射 | 1 |
| [#454](https://www.deep-ml.com/problems/454) | Autoregressive Video Chunk FPS Calculator | 未修复；见归档映射 | 1 |
| [#455](https://www.deep-ml.com/problems/455) | Historical Context Compression Ratio | 未修复；见归档映射 | 1 |
| [#456](https://www.deep-ml.com/problems/456) | Multi-term Memory Patchification for Video | 未修复；见归档映射 | 1 |
| [#457](https://www.deep-ml.com/problems/457) | Implement Attention Sink Detection | 未修复；见归档映射 | 2 |

## 自定义练习：不强行编造官网题号

| 原文件 | 处理 |
|---|---|
| 109-Feature Deconfounder Lab.py | 未找到可确认的一一对应题；原稿保留在 archive/pytorch_legacy/ |
| 117-Language Modeling Tokenizer Lab.py | 未找到可确认的一一对应题；原稿保留在 archive/pytorch_legacy/ |
| 125-Build a Digit Classifier from Scratch.py | 未找到可确认的一一对应题；原稿保留在 archive/pytorch_legacy/ |

逐文件原号、新号和归档路径：[id_mapping.json](audit/id_mapping.json)。
复习顺序：[REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md)。测试边界：[AUDIT_REPORT.md](AUDIT_REPORT.md)。
