（交流可以用英文，本文档中文，保留这句）

# 语言模型项目说明

## 项目目标
编写一份语言设计的全面教程markdown，要包含大量的习题和参考答案（答案默认折叠）。合适时提及相关 pytorch 函数名但不写代码。

## 工具说明
当需要时，可以通过 `gemini -p "深入回答：<要问的问题> -m gemini-2.5-pro"` 来获取 gemini-2.5-pro 的参考意见(gemini 系只问 gemini-2.5-pro 不问别人)
当需要时，可以通过 `echo "<要问的问题>"|llm -m 4.1 来获取 gpt-4.1 的参考意见

## 教程大纲

### 最终章节结构

### 内容设计原则

3. **交互元素**：
   - 保持简单，先用静态图像
   - 逐步增加交互性

5. **章节依赖性**：
   - 每章尽量自包含

6. **练习设计**：
   - 理论和实现混合
   - 包含挑战题
   - 难度递进

8. **前置知识**：
   - 假设学生已有概率论、神经网络基础、PyTorch经验
   - 在首页明确说明这些前置要求

## 章节格式要求

每个章节应包含：

1. **开篇段落** - 引入本章主题，说明学习目标
2. **丰富的文字描述** - 不仅是公式，要有充分的文字解释和直观说明
3. **本章小结** - 总结要点，预告下一章内容

## 输出大小控制

**重要原则**：
- 输入可以是章节级别的请求（如"创建第2章"）
- 但输出必须限制在一个子小节（subsection）的大小，不超过
- 这样确保每次生成的内容精炼且高质量

### 统一样式要求

1. **使用共享CSS/JS文件** - 将通用样式抽取到 `common.css` 和 `common.js`
2. **练习答案默认折叠** - 使用统一的折叠/展开机制
3. **响应式设计** - 确保移动端友好
4. **数学公式** - 使用KaTeX渲染
5. **代码高亮** - 使用Prism.js或类似库

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
