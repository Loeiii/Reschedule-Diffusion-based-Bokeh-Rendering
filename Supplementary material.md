## A  Derivation of the Equation (8)

Typically, we let $\beta$ to vary linearly where $\beta_{end} - \beta_{start} = \gamma$. Consequently, when the number of inference steps is $N$, $\Delta\beta$ is determined as $\gamma/(N - 1)$ and $\beta$ can be represented by Equation~(\ref{eq:beta}),
$$
\begin{align}
    \beta_{i;n} =& \beta_{i;start} + \frac{\gamma}{N - 1}n\\
        = & \beta_{i;end} - \frac{\gamma}{N - 1}(N - 1 - n). \label{eq:beta}
\end{align}
$$
Correspondingly, $\bar{\alpha}$ can be expressed as follows:
$$
\begin{align}
    \bar{\alpha}_{i;T} &= \prod_{n = 0}^{N - 1}(1 - \beta_{n})\\ 
        & = \prod_{n = 0}^{N - 1}\left(1 - \left(\beta_{i;end} - \frac{\gamma}{N - 1}(N - 1 - n)\right)\right)\\
        & = \prod_{n = 0}^{N - 1}\left(1 - \left(\beta_{i;end} - \frac{\gamma}{N - 1}n\right)\right),
\end{align}
$$
which is identical to Equation~(8).

## B  Additional Experimental Results

The additional experimental results under the different combinations of $N$ and $\gamma$ in Section 5.3, Table 4 of the main text, are shown in Figure 8. For comparison, we also conduct an experiment using standard Gaussian noise sampling when the sample step is $N = 100$. The results, as shown in Figure 8, indicate that rescheduling noise to reduce the number of inference steps without utilizing priors leads to a considerable decline in model performance. Additionally, as observed in Figure 8, particularly in the sixth column, with the use of priors at the same scale, a decrease in $N$ leads to an insufficient amount of bokeh effect for the sampling results.

![Figure8](https://image.oct.org.cn/2024/04/Figure8.png)

Table 7 and Figure 9 demonstrate the importance of the refined noise schedule. The decrease of $N$ amplifies the effect of the condition, leading to a convergence of the sampling results towards the condition. Hence, when $N$ is small, it is crucial to moderately reduce the proportion of prior information in the initial state. Through the refinement, we can maintain the model's performance while also reducing the required inference steps.

<img src="https://image.oct.org.cn/2024/04/Table7.png" alt="Table 7" style="zoom: 67%;" />

![Figure9](https://image.oct.org.cn/2024/04/Figure9.png)

Additionally, we present the experimental results for additional noise rescheduling parameter combinations $(N, \gamma)$, as detailed in Table 8 and Figure 10. Table 8 indicates that the impact of $\gamma$ values on sampling results is relatively minor. This is partly because, 
the multiplicative terms in Equation (9) reduce $\gamma$'s impact to some extent, especially for large $N$. However, when $N$ is very small, such as $N = 2$, it is advisable to avoid choosing a too small $\gamma$, as this leads to an overall increase in the rescheduled noise level and consequently, a decline in model performance in some extent, as indicated by Table 8. To provide visual evidence for the minor influence of $\gamma$, the sampling results for $N=2$ with different $\gamma$ are depicted in Figure 10. It can be observed that the differences are visually imperceptible. Therefore, considering that very large values of $N$ are not used in the model application, we set $\gamma = 0.1$ in our experiments.

<img src="https://image.oct.org.cn/2024/04/Table8.png" alt="Table8" style="zoom:80%;" />

![Figure10](https://image.oct.org.cn/2024/04/Figure10.png)

## C  Comparison with DDIM

In our sampling process, although we employ the DDPM[^1] sampling method, it is implemented through the non-Markovian reverse process as proposed by DDIM[^2] by setting $\eta  = 1$. Given that the non-Markovian sampling process inherently reduces the number of sampling steps, we deem it essential to additionally compare the results with those of DDIM in this context.

To ensure the fairness of the comparison, we employ a consistent noise rescheduling approach during the sampling process, with $N = 10, \gamma = 0.1$, and also use priors within the DDIM sampling process. As depicted in Table 9[^table 9], there is a certain degree of performance degradation when using DDIM, consistent with the observations made by [^2].

[^table 9]:<img src="https://image.oct.org.cn/2024/04/Table9.png" alt="Table9" style="zoom:67%;" />

Furthermore, we compared the variability of our method with that of DDIM when using priors.
As shown in Table 10, the introduction of prior information in the initial state greatly reduces the variability of DDIM in terms of SSIM and LPIPS (as compared with Table 1), further validating the efficacy of our proposed method.

![Table10](https://image.oct.org.cn/2024/04/Table10.png)

## D  The significance of our method in the context beyond the immediate community (bokeh rendering) and its generalizability.

For image restoration tasks, though degraded, the input image still contains considerable information. Therefore, utilizing the prior information contained in the input image would benefit the restoration.
In the proposed method, we try to find a critical point where the noised input and target images have the same distribution.  Since the common information shared by the input and target images has not yet been discarded, the noised input image at the critical point is more suitable to serve as the starting point in the sampling process than random Gaussian noise.
Consequently, our proposed prior-aware method could effectively retain essential information from the input, ensuring information continuity and structural integrity in the resulting output. This is crucial for many tasks in image restoration.

We have conducted preliminary experiments on several other image restoration tasks, such as draining, super-resolution, and inpainting, to verify the proposed method's generalization ability beyond the immediate community.We find that, given the variance among different tasks, adjusting the prior proportions allows the method to appropriately adapt to other tasks. 

## Reference

[^1]: *Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 70* *2020.*
[^2]: *Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.*