# Volume Surprise and Liquidity Stress Analysis

## 项目简介

本项目是 **manip-ofi-joint-analysis** 的因子3（Volume Surprise & Liquidity Stress）分析模块，用于分析市场微观结构中的成交量异常和流动性压力。

## 核心功能

### 因子3定义

**VolLiqScore** 是一个综合因子，结合了成交量异常和流动性压力两个维度：

```
VolLiqScore = 0.5 × z_vol + 0.5 × z_liq_stress
```

其中：
- **z_vol**: Volume Surprise (成交量惊喜) - `z-score(log(volume))`
- **z_liq_stress**: Liquidity Stress (流动性压力) - `z-score(range / ATR)`

## 数据覆盖

- **品种**: 6个（BTCUSD, ETHUSD, EURUSD, USDJPY, XAGUSD, XAUUSD）
- **时间周期**: 7个（5min, 15min, 30min, 1H, 2H, 4H, 8H）
- **数据规模**: 18,885,016行（约1890万行）
- **因子有效率**: 99.92%

## 分析阶段

### 阶段0: 数据复用与样本对齐
- 复用旧项目OFI数据
- 创建符号链接
- 生成数据可用性摘要

### 阶段1: 因子构造
- 计算True Range (TR)
- 计算Average True Range (ATR)
- 计算Volume Surprise (z_vol)
- 计算Liquidity Stress (z_liq_stress)
- 生成综合因子VolLiqScore

### 阶段2: 时间结构与频率分析
- 自相关性分析（ACF）
- 持续性指标（半衰期、均值回归速度）
- 高值事件频率统计

### 阶段3: 预测能力分析
- 与未来波动率的相关性
- 尾部事件预测能力
- 多窗口前瞻分析

### 阶段4: 跨品种/跨周期稳健性
- 品种×周期交叉统计
- 变异系数分析
- 稳健性评估

## 核心发现

✅ **因子质量优秀** - 有效率99.92%  
✅ **时间特征清晰** - 中等持续性（ACF(1)≈0.49），快速均值回归（半衰期≈1.2周期）  
✅ **预测能力验证** - 对短期波动有正向预测能力（相关性≈0.12）  
✅ **稳健性良好** - 跨品种、跨周期表现一致  
✅ **应用价值高** - 可用于风险管理和交易策略

## 项目结构

```
.
├── 因子3完整分析报告.md          # 完整的中文分析报告
├── ip.txt                        # 服务器IP地址
├── mishi/                        # SSH密钥（不提交到git）
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件配置
```

## 服务器端项目结构

```
manip-ofi-joint-analysis/
├── data/
│   ├── bars_with_ofi/           # 原始OFI数据（362个文件）
│   └── intermediate/            # 因子3增强数据（317个文件）
├── results/
│   ├── stats/                   # 数据处理摘要
│   ├── stage2_time_structure/   # 时间结构分析结果
│   ├── stage3_predictive_power/ # 预测能力分析结果
│   ├── stage4_robustness/       # 稳健性分析结果
│   └── final_report/            # 最终报告
├── src/
│   └── joint_factors/           # 因子计算模块
└── scripts/                     # 执行脚本
```

## 使用方法

### 环境要求

- Python 3.x
- pandas
- numpy

### 运行分析

所有分析在远程服务器上运行：

```bash
# 连接到服务器
ssh -i mishi/lianxi.pem ubuntu@<server-ip>

# 进入项目目录
cd manip-ofi-joint-analysis

# 运行所有阶段分析
python3 scripts/run_all_stages.py
```

## 应用场景

### 1. 风险管理
- 作为市场压力指标，监控异常波动和流动性风险
- 高因子值时提高风险警戒，降低仓位或扩大止损
- 结合因子1(ManipScore)和因子2(OFI)构建综合风险雷达

### 2. 交易策略
- 高因子值时避免开仓或减少交易频率
- 低因子值时可适度增加仓位
- 因子快速上升时警惕趋势反转

### 3. 市场微观结构研究
- 分析成交量异常与价格波动的关系
- 研究流动性压力对市场质量的影响
- 识别市场操纵或异常交易行为

## 后续研究方向

1. **三因子联合分析** - 结合ManipScore、OFI和VolLiqScore构建综合画像
2. **参数敏感性测试** - 测试不同lookback窗口和权重的影响
3. **Event Study** - 分析重大市场事件前后的因子行为
4. **策略回测验证** - 在实际交易中测试因子的应用价值

## 技术规格

- **服务器**: Ubuntu 22.04 LTS
- **计算参数**:
  - Volume z-score lookback: 50
  - ATR lookback: 50
  - Liquidity stress z-score lookback: 50
  - 因子权重: 0.5 (volume) + 0.5 (liquidity)
- **处理时间**: 217秒（3.6分钟）
- **成功率**: 100%

## 许可证

本项目仅供研究和学习使用。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。