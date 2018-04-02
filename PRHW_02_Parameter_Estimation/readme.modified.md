# PRHW\_02\_Parameter_Estimation

## Programming  

---
Assume ![equation](https://latex.codecogs.com/svg.latex?$p(x)&nbsp;\backsim&nbsp;0.2N(-1,1)&nbsp;+&nbsp;0.8N(1,1)$). Draw ![equation](https://latex.codecogs.com/svg.latex?$n$) samples from ![equation](https://latex.codecogs.com/svg.latex?$p(x)$), for example, ![equation](https://latex.codecogs.com/svg.latex?$n=5,10,50,100,\cdots,1000,\cdots,10000$). Use Parzen-window method to estimate ![equation](https://latex.codecogs.com/svg.latex?$p_n(x)\approx&nbsp;p(x)$).

**(a)** Try window-function ![equation](https://latex.codecogs.com/svg.latex?$P(x)=\left\{\begin{aligned}&\frac{1}{a},-\frac{1}{2}a\leq&nbsp;x\leq&nbsp;\frac{1}{2}a&nbsp;\\&0,otherwise.\end{aligned}\right.$). Estimate ![equation](https://latex.codecogs.com/svg.latex?$p(x)$) with different window width ![equation](https://latex.codecogs.com/svg.latex?$a$).

**(b)** Derive how to compute ![equation](https://latex.codecogs.com/svg.latex?$\epsilon(p_n)=&nbsp;\displaystyle\int[p_n(x)-p(x)]^2dx$) numerically.

**(c)** Demonstrate the expectation and variance of ![equation](https://latex.codecogs.com/svg.latex?$\epsilon(p_n)$) w.r.t different ![equation](https://latex.codecogs.com/svg.latex?$n$) and ![equation](https://latex.codecogs.com/svg.latex?$a$).

**(d)** With ![equation](https://latex.codecogs.com/svg.latex?$n$) given, how to choose optimal ![equation](https://latex.codecogs.com/svg.latex?$a$) from above the empirical experiences?

**(e)** Substitute ![equation](https://latex.codecogs.com/svg.latex?$h(x)$) in (a) with Gaussian window. Repeat (a)-(e).

**(f)** Try different window functions and parameters as many as you can. Which window function/parameter is the best one? Demonstrate it numerically.

### 解答

**(a)**
对于(a)中的window-function结果如下：
```bash
python3 parzen_window_pe.py -n 5,10,50,100,1000,10000 -w 0.25,1,2 -t uniform
```
![uniform_window_function](question_a_fig/uniform_window_function.png)

结果位于`./question_a_fig/uniform_window_function.png`



**分析：**

1. 窗宽![equation](https://latex.codecogs.com/svg.latex?$h_N$)影响某点的概率密度和的幅度，在样本数有限时![equation](https://latex.codecogs.com/svg.latex?$h_N$)很大则幅度很小，这时估计量![equation](https://latex.codecogs.com/svg.latex?$\hat{p}(x)$)为N个宽度较大且函数值变化缓慢的函数叠加，估计分辨率降低。反之![equation](https://latex.codecogs.com/svg.latex?$h_N$)很小则某点的概率密度和的幅度很大，这时估计量为N个尖峰，使估计量变化很大。

2. 样本数增加，单个样本对估计起的作用越来越模糊，随着N增加，估计量越来越好，而样本较少时估计量中会出现不规则扰动。N趋于无穷时估计量会趋于平滑的正态曲线。

   ​

**(b)**
为了估计积分，化为离散值求和，取**x** 为-4 : 0.1 : 4, 计算每一点的均方误差最后求和。窗宽![equation](https://latex.codecogs.com/svg.latex?$h_N$)与样本数N的选取与**(a)**中相同，以此进行评估。
```bash
python3 .\error_estimation.py -n 5,10,50,100,1000,10000 -w 0.25,1,2 -t uniform
```
结果位于`./question_b_result/uniform_window_mse.txt`

可见当样本数达到一定水平，均方误差![equation](https://latex.codecogs.com/svg.latex?$\epsilon(p_n)$)趋于稳定，结果较为一致。而实验中窗宽在2之前随窗宽增加表现出均方误差的减小。



**(c)**
在特定![equation](https://latex.codecogs.com/svg.latex?$h_N$)和N取值的情况下，重复平行采样100次，计算各自的![equation](https://latex.codecogs.com/svg.latex?$\epsilon(p_n)$)，并由此估计其均值和方差。窗宽![equation](https://latex.codecogs.com/svg.latex?$h_N$)与样本数N的选取与**(a)**中相同，以此进行评估。
```bash
python3 exp_var_estimation.py -n 5,10,50,100,1000,10000 -w 0.25,1,2 -t uniform
```
结果位于`./question_c_result/uniform_window_mean_variance.txt`



**(d)**
取定样本数为10,100,1000,10000，估计a在不同取值时(设置为0.5到6，步长为0.1)的![equation](https://latex.codecogs.com/svg.latex?$\epsilon(p_n)$)，来确定窗宽最优取值。
```bash
python3 .\window_width_estimation.py -
n 10,100,1000,10000 -t uniform
```
![uniform_window_width](question_d_fig/uniform_window_width.png)

结果位于`./question_d_fig/uniform_window_width.png`

由结果可见，对于均匀分布窗函数，在不同样本量时，窗宽为1.8左右为最优窗宽。




**(e)**
对于高斯窗函数，结果如下
```bash
python3 parzen_window_pe.py -n 5,10,50,100,1000,10000 -w 0.25,1,2 -t gaussian
```
![gaussian_window_function](question_a_fig/gaussian_window_function.png)
结果位于`./question_a_fig/gaussian_window_function.png`

其余结果除指定窗函数类别这个参数`-t gaussian`不同外，均与均匀分布窗函数相同，其余结果也放在了相同文件夹。需要指出一点，对于高斯窗函数，其最优窗宽在1.0左右，与均匀分布窗函数有所不同。

![gaussian_window_width](question_d_fig/gaussian_window_width.png)




**(e)**
本项目同样实现了指数窗函数，不同参数下拟合结果如下
```bash
python3 parzen_window_pe.py -n 5,10,50,100,1000,10000 -w 0.25,1,2 -t exponential
```
![exponential_window_function](question_a_fig/exponential_window_function.png)
结果位于`./question_a_fig/exponential_window_function.png`
其余结果同以上位于相同文件夹。其在不同样本量下最优窗宽位于2.5左右。

![exponential_window_width](question_d_fig/exponential_window_width.png)



**综合以上结果**，样本量和窗宽对拟合效果的影响具有相同的趋势，样本量越大，拟合越平滑精度越高，越接近真实。窗宽则存在一个最优值，在相同样本量下两侧值均比最优值效果差。

从结果比较来看，三个窗函数中，高斯窗能够在样本量较低(100)时就获得对真实分布较好的拟合，而均匀窗和指数窗相对需要更多的样本达到相同的效果。

综合考虑到计算复杂度，高斯分布选择样本量为1000窗宽为1.0即可达到比较好的效果，错误率相对很低而且其方差接近0。
