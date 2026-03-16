# MS for AI 第二章节实践
本章的内容是Python入门.

基本要求:
- 允许使用AI, 但是这里任务太简单了, AI可以秒, 所以希望自己动手.
- 多查资料.
## 任务
### 1 简单数据处理与可视化
我们将使用Python进行一个简单的数据处理与可视化. 你需要读入一个`csv`格式文件, 其格式如下:
```csv
index, X, Y
1, 1.4, 6.1
2, 5.12, 8.13
3, -1.4, -7.4
...
```
含有index, X, Y三列. 你需要计算每一行对应的X-Y的值, 然后平方并取对数. 即:

$$ Z = \ln{(X - Y)^2} $$

最后使用`matplotlib`库绘制折线图, 横轴为index, 纵轴为Z.

### 2 CSV 数据清洗与分析工程化练习
你需要完成一个可运行、可测试、可复现的小型数据处理程序。输入仍然是 `csv` 文件，且包含 `index, X, Y` 三列，但现在数据可能包含脏数据。你需要完成以下任务：

1. 读取并清洗数据
   - 非法行（缺失值、非数值）需要被丢弃
   - 重复 `index` 保留最后一次出现的数据
   - 清洗后按 `index` 升序排列

2. 计算特征
   - 对每一行计算：
   $$
   Z = \ln\left((X - Y)^2 + \epsilon\right), \quad \epsilon = 10^{-12}
   $$
   - 结果必须保证数值稳定，不出现 NaN/Inf

3. 输出结果  
   - 输出结果 CSV（列为 `index, Z`）  
   - 绘制折线图（横轴 `index`，纵轴 `Z`）  
   - 输出统计摘要：`count/min/max/mean/std`

4. 工程要求  
   - 提供 `startup/` 与 `solution/` 双目录结构  
   - 使用统一测试入口，支持 `--mode startup|solution`  
   - 提供日志输出与清晰错误信息  
   - 提供 CI 脚本，自动执行本题测试

## 目录结构（Exercise 2）
```text
chapter02/
├── startup/
│   └── model.py
├── solution/
│   └── model.py
├── data.py
├── main.py
├── conftest.py
├── test_bench.py
└── test_solution_model.py
```

## 如何运行
在仓库根目录执行：

```bash
python chapter02/main.py --mode solution
```

程序提供菜单，可执行测试与练习流程。首次运行会在 `chapter02/data/` 下自动生成示例输入与输出文件。

## 如何测试
```bash
pytest chapter02/test_bench.py --mode solution
pytest chapter02/test_bench.py --mode startup
pytest chapter02/test_solution_model.py
```

## 验收标准
- solution 模式下流程完整可运行，生成 CSV 与图像输出
- 边界条件（非法行、重复索引、极值输入）均可稳定处理
- 自动化测试通过，模式切换正确，不发生导入串味

