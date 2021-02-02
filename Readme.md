# Face Recognition

This project serves as a research and evaluation for different popular techniques such as:

• ML estimation with Gaussian assumption followed by Bayes’ classification.
• k-Nearest Neighbors (k-NN) rule.
• Kernel SVM and Boosted SVM.
• PCA followed by Bayes classifier and K-NN rule.
• MDA followed by Bayes classifier and K-NN rule.

Here different data sets were used to perform the different tasks such as binary classification and multiclass subject classification.

# Run the code:

OPEN THE FOLLOWING MATLAB (.M) FILES IN MATLAB RUN it.

Requires the dataset in the same folder as .m files.

BINARY CLASSIFICATION (Neutral vs facial expression classification):

Bayesian + PCA + MDA classification task: BAYESIAN_BINARY.m (data.mat)

KNN + PCA + MDA classification task: KNN_BINARY.M (data.mat)

SVM + KERNEL SVM classification task: SVM_BINARY.m and SVM_KERNEL_BINARY.m (data.mat)

MULTICLASS CLASSIFICATION:
BAYESIAN : BAYESIAN_MULTICLASS.m (data.mat)
KNN : KNN_POSEVSILLUMINATION.m (pose.mat and illumination.mat)
SVM : SVM_MULTICLASS.m (illumination.mat)
