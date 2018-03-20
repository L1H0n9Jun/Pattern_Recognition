# 总体分布的非参数估计-Parzen  Window方法
---

## 概述
以`python3`实现了`parzen window`的总体分布的非参数估计的方法。
项目基于以下三种窗函数：
1. 方窗函数

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(u)&space;=&space;\begin{cases}&space;\displaystyle\frac{1}{a},&space;&&space;-\displaystyle\frac{1}{2}a&space;\le&space;u&space;\le&space;\frac{1}{2}a&space;\\&space;0,&space;&&space;otherwise.&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi(u)&space;=&space;\begin{cases}&space;\displaystyle\frac{1}{a},&space;&&space;-\displaystyle\frac{1}{2}a&space;\le&space;u&space;\le&space;\frac{1}{2}a&space;\\&space;0,&space;&&space;otherwise.&space;\end{cases}" title="\phi(u) = \begin{cases} \displaystyle\frac{1}{a}, & -\displaystyle\frac{1}{2}a \le u \le \frac{1}{2}a \\ 0, & otherwise. \end{cases}" /></a>

2. 正态窗函数

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(u)=&space;\frac{1}{\sqrt{2\pi}}&space;\text{exp}&space;\left\{-\frac{1}{2}u^2&space;\right\}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi(u)=&space;\frac{1}{\sqrt{2\pi}}&space;\text{exp}&space;\left\{-\frac{1}{2}u^2&space;\right\}" title="\phi(u)= \frac{1}{\sqrt{2\pi}} \text{exp} \left\{-\frac{1}{2}u^2 \right\}" /></a>

3. 指数窗函数

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(u)=&space;\text{exp}\{-&space;|u|&space;\}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi(u)=&space;\text{exp}\{-&space;|u|&space;\}" title="\phi(u)= \text{exp}\{- |u| \}" /></a>

对总体
<a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;\backsim&space;0.2N(-1,1)&plus;0.8N(1,1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p(x)&space;\backsim&space;0.2N(-1,1)&plus;0.8N(1,1)" title="p(x) \backsim 0.2N(-1,1)+0.8N(1,1)" /></a>
进行非参数估计。

## 结果
代码使用了以下非标准类库：
```python
numpy           # 从正态总体采样(随机数生成器)
matplotlib      # 计算结果可视化
```
在python3环境通过项目目录下`requirements.txt`可快速部署依赖环境。
```bash
pip install -r requirements.txt
```

通过标准类库`argparse`传递参数, 参数说明：
```
python3 parzen_window_pe.py -h
usage: parzen_window_pe.py [-h] -n SAMPLE_NUMBER -w WINDOW_WIDTH -t
                           {uniform,gaussian,exponential}

optional arguments:
  -h, --help            show this help message and exit
  -n SAMPLE_NUMBER, --sample_number SAMPLE_NUMBER
                        Number of samples used to estimate the population.
                        Could be a single int number or multiple numbers
                        separated by comma, on which the final figure depends.
  -w WINDOW_WIDTH, --window_width WINDOW_WIDTH
                        Parzen window width used. Could be a single float
                        number or multiple numbers separated by comma, on
                        which the final figure depends.
  -t {uniform,gaussian,exponential}, --window_type {uniform,gaussian,exponential}
                        Parzen window type used.
```

其中`-n`传递样本数参数，`-w`传递窗宽参数，二者均可同时为多值进行比较，多值时以`,`分隔，脚本可自动解析，并在同一Figure中绘制所有结果。
`-t`传递窗类型参数，脚本可根据参数自动进行选择，`uniform`为方窗，`gaussian`为高斯窗,`exponential`为指数窗

代码执行实例：
```bash
python3 parzen_window_pe.py -n 1,16,256 -w 0.25,1,2 -t uniform
```

---
Author: L1H0n9Jun

Data:   2018/03/20


