# Banking77 复现与运行

实验协议见 [PROTOCOL_ROUND3.md](PROTOCOL_ROUND3.md)。
`results_round3/selection.json` 保存验证集选定配置；
`results_round3/results.json` 只有完整测试结束后才生成。

## 依赖与源

本轮使用 Python 3.11、torch 2.9.1、transformers 5.2.0、numpy 2.4.6、
scikit-learn 1.7.2，CPU fp32、8线程。建议在独立环境安装相同版本；
原两轮实验的环境记录保留在各自结果文件中。

```bash
git clone https://github.com/NandhaKishorM/laya.git laya-upstream
git -C laya-upstream checkout 7a77439b0923d6efe00052d50a7f787840bfce98
hf download convaiinnovations/laya \
  --revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --local-dir laya-model
```

Banking77 源文件为官方
[固定提交](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)
的 `train.csv`、`test.csv`、`categories.json`。
下载到同一目录并传给 `--data`；脚本会校验 Git blob，
拒绝版本不一致、列缺失或标签不完整的输入。

数据引用：Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*,
[ACL论文](https://aclanthology.org/2020.nlp4convai-1.5/)；
[作者仓库](https://github.com/PolyAI-LDN/task-specific-datasets)。
数据版权与许可见发布包随附的 Banking77 LICENSE。

## 重新训练与测试

在本目录运行，路径替换为本机实际目录：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
USE_TF=0 PYTHONDONTWRITEBYTECODE=1 \
python banking77.py --upstream ./laya-upstream --model ./laya-model \
  --data ./data/banking77 --output ./new-round3
```

输出目录必须不存在，以防覆盖原结果。脚本依次完成：
数据和代码指纹冻结 → 词法监督头 → 冻结Encoder特征与线性头 →
验证集预算/候选重排对照 → 写入选择锁 → 测试全部预先注册候选。
不会用测试标签选择模型。

## 加载领域路由器

输入为 JSONL，每行包含唯一 `id` 和非空 `text`。不需要 gold label。

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 USE_TF=0 \
python banking77_runtime.py --artifact ./results_round3 \
  --upstream ./laya-upstream --model ./laya-model \
  --input ./requests.jsonl --output ./answers.jsonl
```

输出保持 Request ID，返回 77 类 schema 中的选择与完整概率向量。
只加载 `.npz/.npy/JSON` 参数，不使用 pickle。词表和 IDF 只由训练集拟合；
新请求不会更新参数。Encoder checkpoint 与上游 revision 必须匹配。
此入口是串行离线 CLI，不提供并发安全的服务实例或 HTTP 延迟保证。

## 资产说明

- `*_head.npz`：新领域线性头，不是完整 Laya 权重。
- `*_vocabulary.json`、`*_idf.npy`：词法向量器参数。
- `rows.jsonl`、`protocol.json`：实际去重划分和原始行索引。
- `*_laya_*.jsonl`：每条真实输入的候选、未经舍入 logits 与 token 使用。
- `test_predictions.jsonl`：最终各候选完整 77 类概率，便于独立复算。
- `*_timing.json`：离线阶段或 batch 计时，包含其路径的编码和执行开销。
- `*_embeddings.npy`：可由固定 Encoder 重算的缓存，不是新增训练样本。

固定分类头需要相同77类 schema；若增加工具、语言或领域，应重新划分数据并验证。
top-10 外的标签保留零概率，因此候选漏召回仍按完整系统错误计数。
