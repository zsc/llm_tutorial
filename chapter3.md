# 第3章：微调技术与对齐方法

预训练模型具备了强大的语言理解和生成能力，但要将其转化为实用的AI助手，还需要通过微调来适应特定任务和人类偏好。本章深入探讨从预训练到实用系统的关键技术桥梁。

## 章节目录

1. [监督微调（SFT）基础](#section1)
2. [参数高效微调技术](#section2)  
3. [指令遵循能力培养](#section3)
4. [对齐方法概览](#section4)
5. [数据质量与多样性](#section5)
6. [评估与迭代改进](#section6)

---

## <a name="section1"></a>3.1 监督微调（SFT）基础

监督微调是将预训练模型适配到特定任务的第一步。虽然概念简单，但细节决定成败。

### 3.1.1 从预训练到微调的范式转变

**预训练 vs 微调的本质区别：**

| 维度 | 预训练 | 微调 |
|------|--------|------|
| 目标 | 学习通用语言模式 | 学习特定任务模式 |
| 数据规模 | TB级别 | GB级别 |
| 数据质量 | 容忍噪声 | 要求高质量 |
| 学习率 | 较大（1e-4） | 较小（1e-5） |
| 训练时长 | 数月 | 数天 |

**微调的数学视角：**
$$\theta_{fine} = \arg\min_{\theta} \mathcal{L}_{task}(\theta) + \lambda ||\theta - \theta_{pre}||^2$$

第二项是隐式的正则化，防止灾难性遗忘。

### 3.1.2 全参数微调流程

**标准流程：**
```python
def supervised_finetuning(pretrained_model, dataset, config):
    model = load_pretrained(pretrained_model)
    
    # 关键配置
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,  # 1e-5 to 1e-6
        weight_decay=0.01
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_ratio * total_steps,
        num_training_steps=total_steps
    )
    
    for epoch in range(config.num_epochs):  # 通常3-5个epoch
        for batch in dataloader:
            loss = compute_loss(model, batch)
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.max_grad_norm
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

**关键超参数选择：**
1. **学习率**：
   - 太大→灾难性遗忘
   - 太小→收敛慢/欠拟合
   - 经验值：1e-5到5e-6

2. **批大小**：
   - 影响梯度噪声
   - 受显存限制
   - gradient accumulation补救

3. **训练轮数**：
   - 过少→欠拟合
   - 过多→过拟合
   - early stopping监控

### 3.1.3 灾难性遗忘与缓解策略

**问题表现：**
- 在新任务上表现提升
- 但通用能力急剧下降
- 特别是少见任务/语言

**缓解策略：**

**1. 混合训练数据**
```python
def create_mixed_dataset(task_data, general_data, mix_ratio=0.1):
    # 保留部分预训练数据
    mixed_data = []
    for task_sample in task_data:
        mixed_data.append(task_sample)
        if random.random() < mix_ratio:
            mixed_data.append(random.choice(general_data))
    return mixed_data
```

**2. 正则化技术**
- L2正则化
- Dropout保持开启
- 知识蒸馏从原模型

**3. 渐进式解冻**
```python
def progressive_unfreezing(model, stage):
    # 逐步解冻更多层
    if stage == 1:
        unfreeze_layers(model, ['lm_head'])
    elif stage == 2:
        unfreeze_layers(model, ['lm_head', 'transformer.h[-2:]'])
    elif stage == 3:
        unfreeze_layers(model, 'all')
```

### 3.1.4 任务特定的适配技巧

**分类任务：**
- 添加分类头
- 使用pooled representation
- 标签平滑缓解过拟合

**生成任务：**
- 保持自回归目标
- 调整生成长度限制
- 控制重复惩罚

**问答任务：**
- 设计合适的prompt格式
- 区分问题和上下文
- 答案抽取vs生成

#### 练习 3.1：设计微调实验
给定一个情感分析任务，设计完整的微调方案，包括数据处理、训练配置和评估指标。

<details>
<summary>查看答案</summary>

**完整微调方案：**

1. **数据预处理：**
   ```python
   def preprocess_sentiment_data(examples):
       # 标准化标签
       label_map = {'positive': 1, 'negative': 0, 'neutral': 2}
       
       # 构造输入格式
       prompts = []
       for text, label in examples:
           prompt = f"Classify the sentiment: {text}\nSentiment:"
           prompts.append({
               'input': prompt,
               'target': label_map[label],
               'text_only': text  # 用于数据增强
           })
       
       return prompts
   ```

2. **数据增强策略：**
   - 同义词替换（保持情感极性）
   - 回译增强
   - 对抗样本生成

3. **训练配置：**
   ```python
   config = {
       'learning_rate': 2e-5,
       'warmup_ratio': 0.1,
       'batch_size': 32,
       'gradient_accumulation': 4,
       'num_epochs': 3,
       'weight_decay': 0.01,
       'max_grad_norm': 1.0,
       'eval_steps': 100,
       'save_steps': 500,
       'early_stopping_patience': 3
   }
   ```

4. **评估指标：**
   - 准确率（主要）
   - F1分数（类别平衡）
   - 混淆矩阵（错误分析）
   - 推理时间（实用性）

5. **防止过拟合：**
   - 验证集监控
   - Dropout = 0.1
   - 标签平滑 = 0.1
   - 数据增强

6. **错误分析：**
   ```python
   def error_analysis(model, test_data):
       errors = []
       for example in test_data:
           pred = model.predict(example['input'])
           if pred != example['target']:
               errors.append({
                   'text': example['text_only'],
                   'true': example['target'],
                   'pred': pred,
                   'confidence': model.get_confidence()
               })
       
       # 按错误类型分组
       analyze_error_patterns(errors)
   ```

</details>

### 3.1.5 微调的计算优化

**混合精度训练：**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = model(input_ids, labels=labels).loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**梯度累积：**
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(model, batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**DeepSpeed集成：**
- ZeRO-2：适合单节点多卡
- ZeRO-3：跨节点大模型
- CPU卸载：显存不足时

### 3.1.6 微调效果诊断

**训练曲线分析：**
1. **Loss曲线**：
   - 平滑下降→正常
   - 震荡→学习率过大
   - 平台→学习率过小

2. **验证指标**：
   - 持续上升→正常
   - 早期下降→过拟合
   - 不变→欠拟合

**梯度健康检查：**
```python
def check_gradient_health(model):
    total_norm = 0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** 0.5
    avg_norm = total_norm / param_count
    
    # 健康范围：0.01 - 10
    return avg_norm
```

**🔬 研究线索：**
- 如何自动选择最优超参数？
- 能否预测需要的训练数据量？
- 如何量化灾难性遗忘程度？

---

## <a name="section2"></a>3.2 参数高效微调技术

当模型规模达到数十亿参数时，全参数微调变得不切实际。参数高效微调（PEFT）技术应运而生。

### 3.2.1 LoRA：低秩适应

**核心思想：**
微调等价于学习一个低秩的参数更新。

**数学原理：**
$$W_{fine} = W_{pre} + \Delta W = W_{pre} + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**实现细节：**
```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 冻结预训练权重
        self.W.weight.requires_grad = False
        
        # 低秩分解
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # 缩放因子
        self.scaling = alpha / rank
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # 原始前向 + 低秩更新
        return self.W(x) + self.lora_B(self.lora_A(x)) * self.scaling
```

**超参数选择：**
- rank：4-64，越大表达力越强
- alpha：通常等于rank
- 目标模块：Q、V最重要，K、O次之

**优势分析：**
1. 参数量：减少10000倍
2. 显存：只需存储r个参数
3. 切换任务：更换LoRA权重即可
4. 训练速度：3-10倍加速

### 3.2.2 QLoRA：量化LoRA

**创新点：**
基础模型量化到4-bit，只有LoRA部分使用全精度。

**关键技术：**
1. **NF4量化**：
   - 专为正态分布设计
   - 信息损失最小

2. **双重量化**：
   - 量化权重
   - 量化量化常数

3. **分页优化器**：
   - 自动CPU卸载
   - 避免OOM

**实现要点：**
```python
def prepare_model_for_qlora(model):
    # 1. 量化基础模型
    model = load_in_4bit(
        model,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 2. 准备训练
    model = prepare_model_for_kbit_training(model)
    
    # 3. 添加LoRA
    config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    return model
```

**内存计算：**
```python
def estimate_qlora_memory(model_size_B, lora_rank):
    # 基础模型（4-bit）
    base_memory = model_size_B * 0.5  # GB
    
    # LoRA参数（16-bit）
    lora_params = model_size_B * 0.001 * (lora_rank / 16)
    lora_memory = lora_params * 2  # GB
    
    # 优化器状态
    optimizer_memory = lora_memory * 4
    
    # 激活值（批大小相关）
    activation_memory = 2  # GB (估计)
    
    total = base_memory + lora_memory + optimizer_memory + activation_memory
    return total
```

### 3.2.3 Prefix Tuning与Prompt Tuning

**Prefix Tuning：**
在每层注入可学习的前缀向量。

```python
class PrefixTuning(nn.Module):
    def __init__(self, num_layers, num_heads, head_dim, prefix_len=10):
        super().__init__()
        self.prefix_len = prefix_len
        
        # 每层的前缀embeddings
        self.prefix_embeddings = nn.ModuleList([
            nn.Embedding(prefix_len, num_heads * head_dim * 2)  # K和V
            for _ in range(num_layers)
        ])
        
    def forward(self, layer_idx, key_values):
        prefix = self.prefix_embeddings[layer_idx](
            torch.arange(self.prefix_len, device=key_values.device)
        )
        
        # 分离K和V
        prefix_k, prefix_v = prefix.chunk(2, dim=-1)
        
        # 拼接到原始K、V
        key = torch.cat([prefix_k, key_values[0]], dim=1)
        value = torch.cat([prefix_v, key_values[1]], dim=1)
        
        return key, value
```

**Prompt Tuning：**
只在输入层添加可学习向量。

**对比分析：**
| 方法 | 参数量 | 表达力 | 训练难度 |
|------|--------|--------|----------|
| Prefix | 中等 | 强 | 较难 |
| Prompt | 最少 | 弱 | 容易 |
| LoRA | 较少 | 最强 | 中等 |

### 3.2.4 Adapter：瓶颈层方法

**架构设计：**
```python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # 残差连接
```

**插入位置：**
- FFN之后
- 自注意力之后
- 或两者都加

**与LoRA对比：**
- Adapter：串行计算，有延迟
- LoRA：并行计算，无额外延迟
- 但Adapter更稳定

#### 练习 3.2：设计混合PEFT策略
结合多种PEFT技术，设计一个既高效又强大的微调方案。

<details>
<summary>查看答案</summary>

**混合PEFT方案设计：**

1. **技术组合：**
   - LoRA：用于注意力层（表达力）
   - Adapter：用于FFN层（稳定性）
   - Prompt Tuning：任务特定前缀

2. **架构实现：**
   ```python
   class HybridPEFT(nn.Module):
       def __init__(self, base_model, config):
           super().__init__()
           self.base_model = base_model
           
           # LoRA for attention
           self.lora_modules = nn.ModuleDict()
           for name, module in base_model.named_modules():
               if 'attention' in name and isinstance(module, nn.Linear):
                   self.lora_modules[name] = LoRALinear(
                       module.in_features,
                       module.out_features,
                       rank=config.lora_rank
                   )
           
           # Adapters for FFN
           self.adapters = nn.ModuleList([
               Adapter(config.d_model, config.adapter_dim)
               for _ in range(config.num_layers)
           ])
           
           # Task-specific prompts
           self.task_prompts = nn.ModuleDict({
               'classification': nn.Embedding(10, config.d_model),
               'generation': nn.Embedding(10, config.d_model),
               'qa': nn.Embedding(10, config.d_model)
           })
   ```

3. **训练策略：**
   - 第一阶段：只训练prompts（快速适应）
   - 第二阶段：解冻LoRA（精细调整）
   - 第三阶段：全部PEFT参数（最终优化）

4. **参数分配：**
   ```python
   def allocate_parameters(total_budget, model_size):
       # 经验分配
       lora_ratio = 0.6
       adapter_ratio = 0.3
       prompt_ratio = 0.1
       
       lora_rank = int(total_budget * lora_ratio / (model_size * 0.01))
       adapter_dim = int(total_budget * adapter_ratio / (model_size * 0.02))
       prompt_len = int(total_budget * prompt_ratio / model_size)
       
       return {
           'lora_rank': min(64, lora_rank),
           'adapter_dim': min(128, adapter_dim),
           'prompt_len': min(20, prompt_len)
       }
   ```

5. **动态选择：**
   ```python
   def select_peft_method(task_type, data_size, model_size):
       if data_size < 1000:
           return 'prompt_tuning'  # 数据少
       elif model_size > 10e9 and data_size < 10000:
           return 'qlora'  # 大模型小数据
       elif task_type in ['classification', 'ner']:
           return 'adapter'  # 稳定性重要
       else:
           return 'hybrid'  # 综合最优
   ```

</details>

### 3.2.5 PEFT技术的统一视角

**数学统一：**
所有PEFT方法都可以看作在参数空间施加约束：
$$\theta_{fine} = \theta_{pre} + P\delta$$

其中P是投影矩阵：
- LoRA：低秩投影
- Adapter：瓶颈投影
- Prefix：位置投影

**选择指南：**
```python
def recommend_peft_method(requirements):
    if requirements['memory_critical']:
        return 'qlora'
    elif requirements['inference_speed_critical']:
        return 'lora'  # 无额外延迟
    elif requirements['stability_critical']:
        return 'adapter'
    elif requirements['few_shot']:
        return 'prompt_tuning'
    else:
        return 'lora'  # 默认选择
```

### 3.2.6 PEFT的未来方向

**1. 自动化PEFT：**
- 自动选择秩
- 自动选择模块
- 神经架构搜索

**2. 任务间迁移：**
- LoRA组合
- 知识蒸馏
- 持续学习

**3. 理论理解：**
- 为什么低秩有效？
- 最优秩如何确定？
- 不同层的重要性？

**⚡ 设计选择：**
选择PEFT方法时考虑：
- 参数预算
- 推理延迟要求
- 任务类型
- 数据规模

**🔬 研究线索：**
- PEFT方法的理论最优性？
- 如何组合多个LoRA实现新能力？
- 是否存在通用的PEFT架构？

---

[← 返回目录](index.md) | [上一节：监督微调基础 →](#section1) | [下一节：指令遵循能力培养 →](#section3)