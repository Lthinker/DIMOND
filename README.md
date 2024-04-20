# DIMOND Tutorial
![DIMOND framework](https://github.com/Lthinker/DIMOND/blob/main/Images/DIMOND_Framewor1.png)

**DIMOND framework**. a) DIMOND consists of three steps, namely mapping, modeling and optimization. For the mapping, a neural network (NN) is utilized to transform input diffusion MRI data to unknown parameters of a diffusion model (e.g., one b = 0 image and six component maps for the diffusion tensor model). NN-estimated parameters are then used to synthesize the input diffusion data via the forward model (e.g., tensor model) during the modeling process. During the optimization, the training of the NN aims to minimize the difference (e.g., mean squared error, MSE) between the acquired and synthesized diffusion data using gradient descent. Only the loss within the mask specified by the diffusion model assumption is considered. b) In this work, a plain NN is adopted, which consists of _M_ 3 × 3 × 3 convolutional layers to utilize spatial redundancy and _N_ fully connected layers. Each convolutional layer is paired with a ReLU activation layer. Each fully connected layer, except for the output layer, is paired with a ReLU activation layer and a dropout layer with a dropout rate of _p_.


## **Refereces**
Z. Li, Z. Li, B. Bilgic, H.-H. Lee, K. Ying, S. Y. Huang, H. Liao, Q. Tian, DIMOND: DIffusion Model OptimizatioN with Deep Learning. _Adv. Sci._  2024, 2307965. [https://doi.org/10.1002/advs.202307965](https://doi.org/10.1002/advs.202307965)

