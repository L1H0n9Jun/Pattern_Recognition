# PRHW\_03\_EM_Readme

### 李洪军，2017310864

---

| Points | <img src="https://latex.codecogs.com/svg.latex?$x_1$" />  | <img src="https://latex.codecogs.com/svg.latex?$x_2$" /> | <img src="https://latex.codecogs.com/svg.latex?$x_3$" /> | <img src="https://latex.codecogs.com/svg.latex?$x_1$" /> | <img src="https://latex.codecogs.com/svg.latex?$x_2$" /> | <img src="https://latex.codecogs.com/svg.latex?$x_3$" />|
| :----: | :----: | :----: | :---: | :----: | :---: | :----: |
|   1    |  0.42  | -0.087 | 0.58  |  -0.4  | 0.58  | 0.089  |
|   2    |  -0.2  |  -3.3  | -3.4  | -0.31  | 0.27  | -0.04  |
|   3    |  1.3   | -0.32  |  1.7  |  0.38  | 0.055 | -0.035 |
|   4    |  0.39  |  0.71  | 0.23  | -0.15  | 0.53  | 0.011  |
|   5    |  -1.6  |  -5.3  | -0.15 | -0.35  | 0.47  | 0.034  |
|   6    | -0.029 |  0.89  | -4.7  |  0.17  | 0.69  |  0.1   |
|   7    | -0.23  |  1.9   |  2.2  | -0.011 | 0.55  | -0.18  |
|   8    |  0.27  |  -0.3  | -0.87 | -0.27  | 0.61  |  0.12  |
|   9    |  -1.9  |  0.76  | -2.1  | -0.065 | 0.49  | 0.0012 |
|   10   |  0.87  |  -1.0  | -2.6  | -0.12  | 0.054 | -0.063 |


**1.** Suppose we know that the ten data points in category <img src="https://latex.codecogs.com/svg.latex?$\omega_1$" /> in the table above come from a three-dimensional Gaussian. Suppose, however, that we do not have access to the <img src="https://latex.codecogs.com/svg.latex?$x_3$" /> components for the even-numbered data points.
+ Write an EM program to estimate the mean and covariance of the distribution. Start your estimate with <img src="https://latex.codecogs.com/svg.latex?$\mu_0&space;=&space;0$"/> and <img src="https://latex.codecogs.com/svg.latex?$\Sigma_0&space;=&space;I$" />, the three-dimensional identity matrix.
+ Compare your final estimate with that for the case when there is no missing data.

### 说明

**E-step:**
基于初始或M步的参数值，求不完全样本的<img src="https://latex.codecogs.com/svg.latex?$x_3$" />的期望，获得完整数据。
<img src="https://latex.codecogs.com/svg.latex?$$x_{3}=arg\mathop{\max}_{x_{3}}L(\mu,\Sigma|x)=\displaystyle\frac{1}{(2\pi)^{3/2}|\Sigma|^{1/2}}\exp\left\{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right\}$$" style="display:block;margin:0 auto"/>

**M-step:**
基于E步获得的完整数据和原始完整数据求参数$\mu$和协方差阵$\Sigma$的最大似然估计
<img src="https://latex.codecogs.com/svg.latex?$$\mu=E\{x\}\\\Sigma=E\{(x-\mu)(x-\mu)^T\}$$" />

具体地说，
<img src="https://latex.codecogs.com/svg.latex?$$\mu_i=E\{x_i\}=\displaystyle\int_{E^d}x_ip(x)dx=\displaystyle\int_{-\infty}^{&plus;\infty}x_ip(x_i)dx_i\\\sigma_{ij}^2=E[(x_i-\mu_i)(x_j-\mu_j)]=\displaystyle\int_{-\infty}^{&plus;\infty}(x_i-\mu_i)(x_j-\mu_j)p(x_i,x_j)dx_ix_j$$" />

其中<img src="https://latex.codecogs.com/svg.latex?$p(x_i)$" />为边缘分布，<img src="https://latex.codecogs.com/svg.latex?$\sigma_{ij}^2$" title="$\sigma_{ij}^2$" />为协方差阵<img src="https://latex.codecogs.com/svg.latex?$\Sigma$" title="$\Sigma$" />对应位置元素。

如此迭代至收敛。


**代码执行:**
``` bash
python3 3d_gaussian_em.py
```

**结果:**

<img src="https://latex.codecogs.com/svg.latex?$\mu$" title="$\mu$" /> EM估计结果:
<img src="https://latex.codecogs.com/svg.latex?$$\left[-0.0709,&space;-0.6047,&space;0.773&space;\right]$$" title="$$\left[-0.0709, -0.6047, 0.773 \right]$$" />

<img src="https://latex.codecogs.com/svg.latex?$\Sigma$" title="$\Sigma$" /> EM估计结果:
<img src="https://latex.codecogs.com/svg.latex?$$&space;\left[&space;\begin{matrix}&space;0.90617729&&space;0.56778177&&space;0.8813737&space;\\&space;0.56778177&&space;4.20071481&&space;0.4622071&space;\\&space;0.8813737&space;&&space;0.4622071&space;&&space;1.321021&space;\\&space;\end{matrix}&space;\right]&space;$$" title="$$ \left[ \begin{matrix} 0.90617729& 0.56778177& 0.8813737 \\ 0.56778177& 4.20071481& 0.4622071 \\ 0.8813737 & 0.4622071 & 1.321021 \\ \end{matrix} \right] $$" />

<img src="https://latex.codecogs.com/svg.latex?$\mu$" title="$\mu$" /> MLE结果:
<img src="https://latex.codecogs.com/svg.latex?$$\left[-0.0709,&space;-0.6047,&space;-0.911&space;\right]$$" title="$$\left[-0.0709, -0.6047, -0.911 \right]$$" />
<img src="https://latex.codecogs.com/svg.latex?$\Sigma$" title="$\Sigma$" /> MLE结果:
<img src="https://latex.codecogs.com/svg.latex?$$\left[&space;\begin{matrix}&space;0.90617729&&space;0.56778177&&space;0.3940801&space;\\&space;0.56778177&&space;4.20071481&&space;0.7337023&space;\\&space;0.3940801&space;&&space;0.7337023&space;&&space;4.541949&space;\\&space;\end{matrix}&space;\right]$$" title="$$\left[ \begin{matrix} 0.90617729& 0.56778177& 0.3940801 \\ 0.56778177& 4.20071481& 0.7337023 \\ 0.3940801 & 0.7337023 & 4.541949 \\ \end{matrix} \right]$$" />
**分析**

从估计结果来看第三维结果很不理想，但是参数值的确已经收敛。由于收敛时迭代次数较少，我尝试手动指定500次迭代次数，但发现最终结果差别并不大，我猜测可能参数收敛到了局部极值，或者是由于参数较少的原因。



**2.** Suppose we know that the ten data points in category <img src="https://latex.codecogs.com/svg.latex?$\omega_2$" title="$\omega_2$" /> in the table above come from a three-dimensional uniform distribution <img src="https://latex.codecogs.com/svg.latex?$p(x|\omega_2)&space;\sim&space;U(x_l,&space;x_u)$" title="$p(x|\omega_2) \sim U(x_l, x_u)$" />. Suppose, however, that we do not have access to the $x_3$ components for the even-numbered data points.

+ Write an EM program to estimate the six scalars comprising <img src="https://latex.codecogs.com/svg.latex?$x_l$" title="$x_l$" /> and <img src="https://latex.codecogs.com/svg.latex?$x_l$" title="$x_u$" /> of the distribution. Start your estimate with <img src="https://latex.codecogs.com/svg.latex?$x_l&space;=&space;(-2,&space;-2,&space;-2)^t$" title="$x_l = (-2, -2, -2)^t$" /> and <img src="https://latex.codecogs.com/svg.latex?$x_u&space;=&space;(&plus;2,&space;&plus;2,&space;&plus;2)^t$" title="$x_u = (+2, +2, +2)^t$" />.
+ Compare your final estimate with that for the case when there is no missing data.

### 说明

**E-step:**
基于初始或M步的参数值，求不完全样本的$x_3$的期望，获得完整数据。
<img src="https://latex.codecogs.com/svg.latex?$$E\{x_3\}&space;=&space;\displaystyle\frac{x_{l,3}&plus;x_{u,3}}{2}$$" title="$$E\{x_3\} = \displaystyle\frac{x_{l,3}+x_{u,3}}{2}$$" />

**M-step:**

基于E步获得的完整数据和原始完整数据求参数$x_l$和$x_u$的最大似然估计
<img src="https://latex.codecogs.com/svg.latex?$$x_{l,i}=x_{i(1)}\\x_{u,i}=x_{i(10)}$$" title="$$x_{l,i}=x_{i(1)}\\x_{u,i}=x_{i(10)}$$" />
其中<img src="https://latex.codecogs.com/svg.latex?$x_{(1)}$" title="$x_{(1)}$" />为顺序统计量。
如此迭代至收敛。

**代码执行:**
``` bash
python3 3d_gaussian_em.py
```

**结果:**

<img src="https://latex.codecogs.com/svg.latex?$x_l$" title="$x_l$" /> EM估计结果:
<img src="https://latex.codecogs.com/svg.latex?$$\left[-0.4,0.054,-0.18&space;\right]$$" title="$$\left[-0.4,0.054,-0.18 \right]$$" />
<img src="https://latex.codecogs.com/svg.latex?$x_u$" title="$x_u$" /> EM估计结果:
<img src="https://latex.codecogs.com/svg.latex?$$\left[0.38,0.69,0.089&space;\right]$$" title="$$\left[0.38,0.69,0.089 \right]$$" />
<img src="https://latex.codecogs.com/svg.latex?$x_l$" title="$x_l$" /> MLE结果:
<img src="https://latex.codecogs.com/svg.latex?$$\left[-0.4,0.054,-0.18&space;\right]$$" title="$$\left[-0.4,0.054,-0.18 \right]$$" />
<img src="https://latex.codecogs.com/svg.latex?$x_u$" title="$x_u$" /> MLE结果:
<img src="https://latex.codecogs.com/svg.latex?$$\left[0.38,0.69,0.12&space;\right]$$" title="$$\left[0.38,0.69,0.12 \right]$$" />
**分析:**

下界的估计比较准确，而上界有一定偏差。

### 总结

由于未探究EM算法对样本数的依赖性，但是由于其基于最大似然估计，所以样本量过少可能导致过拟合而不能较好的反映总体的特征。仿真中除均匀分布的下界估计较好，其他结果都不甚理想。