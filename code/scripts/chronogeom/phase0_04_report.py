"""Evidence-only report assembly. Does not select layers or rerun inference."""
from pathlib import Path
import sys,json,argparse,shutil,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
from scripts.chronogeom.phase0_03_report import table

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--docs',type=Path,required=True);args=ap.parse_args();root=args.root;docs=args.docs;dest=docs/'phase0_04/results';dest.mkdir(exist_ok=False);a=root/'analysis_final';s=pd.read_csv(a/'condition_summary.csv');ci=pd.read_csv(a/'paired_intervals.csv');geo=json.loads((root/'audit/geometry.json').read_text());m=json.loads((root/'execution_corrected/metadata.json').read_text());q=json.loads((a/'quality.json').read_text());assert q['gate'];raw=pd.read_json(root/'execution_corrected/interventions.jsonl',lines=True)
 for p in a.iterdir():
  if p.is_file():shutil.copy2(p,dest/p.name)
 for p in (root/'audit').glob('*.json'):shutil.copy2(p,dest/('audit_'+p.name))
 for folder in ['execution','repair_rescue_noop','execution_corrected']:
  for n in ['metadata.json','preflight.json','config.json']:
   p=root/folder/n
   if p.exists():shutil.copy2(p,dest/(folder+'_'+n))
 norm=raw[raw.requested_l2>0];nr=norm.groupby('layer').agg(n=('relative_norm_error','size'),median_relative_error=('relative_norm_error','median'),p95_relative_error=('relative_norm_error',lambda x:x.quantile(.95)),max_relative_error=('relative_norm_error','max'),over1percent=('relative_norm_error',lambda x:int((x>.01).sum()))).reset_index();nr.to_csv(dest/'norm_matching_by_layer.csv',index=False);norm[norm.relative_norm_error>.01].to_csv(dest/'norm_matching_failures.csv',index=False)
 diagnostics=[];angles=[]
 for g in geo:
  for pair in ['h_f','h_pca','f_pca','h8_f']:
   r=g[pair];diagnostics.append(dict(layer=g['layer'],comparison=pair,rank_a=r['rank_a'],rank_b=r['rank_b'],overlap_trace=r['overlap_trace'],overlap_fraction_a=r['overlap_fraction_a'],overlap_fraction_b=r['overlap_fraction_b'],projector_frobenius=r['projector_frobenius'],chordal=r['chordal']));angles.append(dict(layer=g['layer'],comparison=pair,angles_deg=', '.join(f'{v:.6g}' for v in r['angles_deg'])))
 diagnostics=pd.DataFrame(diagnostics);diagnostics.to_csv(dest/'subspace_distances.csv',index=False);angles=pd.DataFrame(angles);angles.to_csv(dest/'principal_angles.csv',index=False)
 donor=json.loads((root/'audit/donor_manifest.json').read_text());missing=pd.DataFrame([r for r in donor if not r['available']]);missing.to_csv(dest/'unmatched_references.csv',index=False)
 small=s[(s.control_seed==42)&(s['mode']=='raw')];selci=ci[(ci.bootstrap=='shared_nodes')&(ci.seed==1729)]
 def conditions(op,methods=None):
  part=small[small.operation==op]
  if methods:part=part[part.method.isin(methods)]
  return table(part[['layer','method','rank','n','effect_mean','accuracy','baseline_accuracy','l2_mean']])
 def intervals(comparisons,operation=None,mode=None):
  part=selci[selci.comparison.isin(comparisons)]
  if operation:part=part[part.operation==operation]
  if mode:part=part[part['mode']==mode]
  return table(part[['layer','operation','mode','comparison','rank_a','rank_b','n','point','low','high']])
 analysis='''# Numerical Subspace Analysis — Phase 0-04

2026-10-24。**Verified / 已有真实GPT-J激活的独立计算分析**。输入是Phase0-03训练316行（79个a身份）、4096维BF16来源激活；只分析a-token、全部六个固定block-input索引，不按因果分数选层。基于当前训练拟合定义的子空间不是网络已学会某种模块划分的证明。

## 定义与数值稳定性

原Phase03方向存为FP32，本轮FP64 SVD重新正交化，零/近零方向按相对1e-6阈值处理（绝对最低1e-12），不填充有效秩。两个正交行基U,V的principal cosines为svd(U Vᵀ)，裁剪到[0,1]后arccos；接近零的角度还用残差SVD的sinθ判定，避免1−cos²灾难性抵消。所有方向正交误差、残差谱及四容差敏感性保留在audit_geometry.json。

重叠trace(PU PV)=Σcos²θ；距离||PU−PV||F=√(rU+rV−2Σcos²θ)，chordal为前者/√2。无需显式分配4096²矩阵。秩不同的重叠两方向分母分别报告；H8/F7即使F完全包含在H中，Frobenius仍为1，不误称两个空间完全相同。

## 所有Principal Angles

'''+table(angles)+'''
H7/F7前六角仅约10^-6度的计算舍入，最大角依层为84.71°–89.74°，各自剩余一维近乎正交。不能因平均重叠较高就说所有方向相同，也不能把比较均值当两套独立机制。

## 投影距离与重叠

'''+table(diagnostics)+'''
H7/F7约6/7=85.7%投影重叠；H7/PCA7约66.0%–75.4%，是连续重叠而非断言存在相同精确离散交。PCA基由同样训练激活得到，不能用“PCA不同”推断所有保留信息都不是数值。

## 为什么六维共享部分在数学上几乎不可避免

训练Fourier特征矩阵XF的列空间包含于Helix特征XH=[XF,linear]。最小二乘拟合的hidden行空间满足row(PXF Y)⊆row(PXH Y)。本次H8秩8、F7秩7，故F7⊂H8；H7又是H8内按训练拟合方差截断的一维降秩空间。一般情况下两个7维子空间位于同一8维空间，交维至少7+7−8=6。**因此共享六维部分首先是嵌套回归定义的数学后果，不能单凭其存在宣称LLM发现了共享周期计算模块。** 新干预用于检验这部分信息的实际局部作用。

定义S为数值近交，UH为H7在S正交补的一维，UF为F7在S正交补的一维；UH与UF本身不强制互相正交。不能把(I−PF)H7直接叫作“仍属于H7”的唯一空间。本轮额外定义QL=span((I−PF)H8)，它与F7严格正交，是在当前特征拟合/输出度量下相对Fourier的线性增量方向，不是完整纯线性计算机制。

原整数网格sin(πa)退化，Helix设计10列含bias、有效设计秩9、中心hidden秩8；Fourier中心hidden秩7。维数、有效秩和参数数必须分开。S可能混合多个预设频率；在有限0..99域内，周期特征也可携带大小/顺序信息。此分解没有隔离“只含mod10”的语义方向。

四个sinθ阈值1e-7/1e-6/1e-5/1e-4在所有六层均得S6、UH1、UF1、QL1。QL与UH高度对齐（约sin最大角），与UF正交；近共线或零秩情况下程序返回真实秩，单元测试覆盖已知交、旋转不变性、近共线、常量及恢复恒等式。没有根据干预结果调阈值。

## 增量作用的实测边界

原始swap（相同85对）的QL平均LD增益，index1/9为0.2842/0.3186，而S6为7.4480/7.5767；QL单独恢复完整正确答案均为0/85。相同rank1、近似等扰动对随机方向的配对差约0.2857/0.2950，对应探索区间[0.0521,0.7369]/[0.0688,0.6831]。不能说QL完全无用，但它没有表现出Helix独有、稳定、足以解释整个计算的优势。等扰动有18条未满足1%误差，详见主报告，不能冒称所有记录严格等强度。

全部层、符号与不同操作分开见PHASE0_04_RESULTS.md及逐pair原始JSONL。共享方向和unique方向贡献非线性，不把均值比率解释成可加的机制百分比。
'''
 (docs/'NUMERICAL_SUBSPACE_ANALYSIS.md').write_text(analysis)
 report='''# ChronoGeom Phase 0-04：数值几何与计算性质最终集中审计

2026-10-24。**Verified：任务A子空间分析、任务B真实GPT-J局部干预已执行；任务C/E性质协议与下一阶段设计已完成，未训练。** MIT总体仍为Partial Replication。本轮结束后停止自动追加MIT模型复现，提交用户判断。

## 核心判断

项目层面的最终决策选 **C：尚不能完整区分几何形状、表示容量和训练机制的作用**。B获得清晰但有限的局部支持：在当前拟合定义、GPT-J、提示与正确题子集，共享S6携带大部分早中层的干预效果，Helix7与Fourier7不应解释为独立机制。没有A所要求的稳定独占优势；线性增量并非严格零效应，因此也不宣称B是排他的完整机制。

详见NUMERICAL_SUBSPACE_ANALYSIS.md、REPRESENTATION_PROPERTIES_PROTOCOL.md与CHRONOGEOM_RESEARCH_DECISION.md。全部固定层均报告；不因为晚层弱、方向小或置信区间跨零而删去。

## 输入、代码与运行

唯一代码根 `/home/home_new/qsmx/pycodes/ChronoGeom`；分支chronogeom/phase0-04。任务A先用旧缓存分析结构，再将B固定提交为 **59ead0c564360cbcc38e1bb76214f3dd75951f36**；控制修复为 **50fdea40e88f4dd3571e7217c8a348ca34b8047f**，其它干预未重复或选择性改写。配置和协议见phase0_04/config.json、PHASE0_04_PROTOCOL.md；原执行/修复时源码快照和SHA分别保留。

运行根 `/home/home_new/qsmx/runs/ChronoGeom/phase0_04/phase0_04_20261024_01`。audit保存训练几何/参照索引，execution为首轮原始记录，repair_rescue_noop为修正控制，execution_corrected仅替换492条无效控制用于最终统计，analysis_initial保留首次失败质量检查，analysis_final为最终统计。原目录/原结果不覆盖。

复用GPT-J revision b71ae8bc86cac13154e03e92b5855203086b722e、已验SHA256的12.11GB官方FP16存储权重；本轮BF16/eager推理，local_files_only和offline，不下载。Python3.10.4 / torch2.6.0+cu124 / transformers5.12.1，GPU5 RTX4090现场空闲后选用。主推理131.69秒，控制补跑21.52秒，合计153.21秒，峰值约11.31GiB；主任务600秒、补跑120秒外部上限，已耗主时长+补跑上限<600秒。进程均退出，无训练、Encoder、TSLib预测或大型新资产。

## A. 子空间关系

H7/F7共享6维，各剩1维最大角84.71°–89.74°；完整H8包含F7，QL为相对Fourier的1维正交增量。四个容差共享秩均为6。六维共享部分是嵌套回归/降秩的数学结果，不是单凭角度证明内部算法；实际因果作用由下面新实验检验。

'''+table(diagnostics[diagnostics.comparison=='h_f'])+'''
## B. 固定样本和条件替换

固定128候选中的85个旧联合正确pair，拟合79个a身份、独立测试20个，模板/六层/整数单token要求与Phase03相同，不根据新效果重筛。swap沿用全部85；条件实验仅82对有同b的训练参照。原3个未匹配pair及b逐层记录在unmatched_references.csv，不以跨b补齐或偷偷替换测试身份。

训练参照：同b且在H8正交补与clean隐藏状态最近的训练行，平局原行序，不读模型答案/logits；同层所有方法共享同一个参照。自回归a在b之前，b本身不会改变该位置a隐藏状态（浮点batch差异另计）；“同b”主要匹配题目与下游上下文，**不是估计了完整p(h_a|其余网络状态)**。训练行每个b仅少量参照，未对所有潜在混杂条件充分匹配。

操作定义：swap=corrupt+P(clean−corrupt)；conditional=clean+P(reference−clean)；rescue先对H8做conditional损伤，再用某个子空间的clean坐标恢复。Full只改同一个a位置，不改整层所有位置。S为共享6维，unique_H/F与QL各1维，PCA/Random按表中实际秩。Full/H8完全恢复主要是代数/实现控制，不能单凭该控制当机制证据。

指标LD=logit(clean答案)−logit(corrupt答案)。swap/rescue effect为相对各自baseline的LD增加；conditional effect为LD下降。accuracy始终是当前clean答案正确率，rescue baseline是损伤网络而非原正确网络。

### 全部六层 swap 原始对照

'''+conditions('swap')+'''
相同rank7时，index1/9 Helix7−Fourier7为−0.6570/−0.5824，探索区间跨零；没有稳定Helix优势。S6均值7.4480/7.5767接近H7的7.9454/8.0590，但S6与H7不同秩，且LD非线性；这些比值不能叫“解释了94%的计算机制”。QL增量小于共享部分，但非全零，亦不支持把Helix与Fourier说成完全相同空间。

### 全部六层 conditional coordinate resampling

'''+conditions('conditional')+'''
index1/9 S6替换损失4.7307/4.4043 LD，clean正确率由100%降至14.63%/15.85%；Random6同秩损失0.0015/−0.0010，但扰动范数明显更小，不能仅由此对照宣称形状特异性。PCA6损失3.7278/2.7267，部分信息同样有用；共享方向和PCA差异仍可能含监督拟合的相关信息/能量差异。

QL条件替换损失0.3757/0.3660，准确率89.02%/95.12%，不能声称独有方向完全没有作用。这些是局部信息参与/条件必要性的证据，重采样操作同时可能改变词元语义等，不能当普遍必要性。

### 全部六层 subspace rescue

'''+conditions('rescue')+'''
index1/9 H8损伤后正确率均7/82=8.54%；只恢复共享S6达到67/82=81.71%、71/82=86.59%。H7恢复75/82=91.46%、74/82=90.24%；F7恢复73/82=89.02%、78/82=95.12%，优势方向不稳定。仅QL恢复9/82=10.98%、8/82=9.76%。这些比较支持共享信息作用大，保留小的增量/相互作用；不能把各方向贡献简单相加。H8投影恢复index9仍有1例错误，BF16操作不严格可逆；Full clean位置精确恢复全部82。

## 等扰动与分布诊断

增量方向同rank1匹配QL的逐pair BF16扰动，而非只匹配平均。2890条正目标记录P95相对误差0.000311，**18条误差>1%，最大19.61%**。极小目标及BF16量化使四次乘法校准无法处处精确匹配；全部记录保留，未丢弃以美化效果。它是近似强度对照，执行质量门通过不等于严格等范数质量通过。所有失败行及逐层范数误差表已保存。

'''+table(nr)+'''
全部同rank1近似等扰动配对区间：

'''+intervals(['linear_increment_minus_unique_h','linear_increment_minus_unique_f','linear_increment_minus_pca1','linear_increment_minus_random1'],'swap','equal_linear_increment')+'''
诊断包括实际L2、BF16舍入、H7/F7/PCA7投影重建MSE、train-PCA16 regularized score-Mahalanobis及补空间MSE、最近同b训练激活L2、与各自baseline最近距离比。PCA16仅用训练316，ridge=最大方差×1e-4，未估计高维完整概率密度。所有层所有条件在condition_summary.csv，不按距离挑结果；无同b的swap3行相关诊断标缺失。

关键条件的分布诊断（包括没有强因果效应的晚层）：

'''+table(s[(s.control_seed==42)&(s.operation=='conditional')&s.method.isin(['noop','shared','linear_increment','helix8','full'])][['layer','method','n','l2_mean','h7_recon_mse','f7_recon_mse','p7_recon_mse','score_m2','pca16_off_mse','nn_same_b_l2','nn_ratio_median']])+'''
index1/9共享条件替换的最近距离比中位数约0.869，QL约0.997；低维score距离也可能上升。Full参照激活最近距离0只是因为它本身在训练缓存中，**不保证其他token/整段网络状态自然**。所有子空间混合即使来源真实激活，仍可能不在数据流形；没有声明已实现严格条件分布保持。

## 依赖感知统计

逐pair、逐层配对，不pool六层，不把22458行当独立样本。共享数值身份节点bootstrap对clean_a/corrupt_a/b统一权重，同一pair重复身份只乘一次；每次2000draws，seeds1729/1730/1731；另有clean/corrupt/b单角色敏感性。固定fit、一个模型、固定测试身份/选题，不能给出预训练或重新拟合的不确定性。节点图扩展没有有限样本95%覆盖保证，未校正多重比较；low/high仅探索2.5/97.5百分位，不宣称统计显著。

全部主要区别的seed1729区间（其它种子与单角色在paired_intervals.csv，逐方法区间在method_intervals.csv）：

'''+intervals(['shared_minus_random6','unique_h_minus_unique_f','linear_increment_minus_unique_f','helix7_minus_fourier7'])+'''
index1/9 uniqueH−uniqueF的conditional差0.1151/−0.0447、rescue差−0.0137/−0.1966，区间均跨零。QL相对随机方向部分LD区间不跨零也不能升级为稳健独占优势；完整答案恢复仍0/85，且与uniqueF/PCA1比较、随机控制seed、有限范数误差及固定拟合限制均保留。全部随机seed/层/分组n在CSV，carry分组只作描述，不作独立模型重复。

## 质量、失败与精度限制

最终33项单元测试通过，含数值交/角度、秩不足、旋转不变性、已知恢复以及随机小GPT-J上的damaged-Noop回归测试；小随机模型只是实现检查，不冒充科学结果。全部22458条最终分析行有限且完整；No-op最大logit误差0；Full index0 swap恢复100%，Full rescue六层均100%；原85当前激活与旧缓存最大差0，基线漂移0。

**控制修复**：首次rescue No-op没有重新hook损伤激活，因此成了运行原clean网络，初始quality gate=false（最大logit误差14.125）。没有隐瞒/覆盖：原结果execution、analysis_initial保留；只补跑6×82=492条No-op，损伤baseline LD与初次逐行完全相同（max difference0），其余21966条记录的原字段和值保持不变，另加来源标签，来源标签/两个原始SHA记录在corrected metadata。未改变样本、子空间或评价协议。补充代码把控制路由抽为函数并加入真实小transformer回归测试。

**batch数值差异**：85/82组成不同，5条样本的隐藏状态在后续层不同；最大绝对差按index1/9/14/20/27分别0.0625/0.1875/0.15625/0.25/1.0，index0零。已在主推理前为82子集单独捕获同batch clean及损伤baseline，并用其隐藏状态构造条件/恢复，避免把batch差异当干预。85的swap与旧缓存逐位一致；三组操作不可把baseline混用。Phase03和本轮FP64重新正交化后的H/F重复均值有约0.001–0.004差别，保留本轮实测值，不能冒称跨脚本逐位相同。

所有成功、失败、负效应、未匹配参照、范数校准不足、初次错误控制都保留。没有新增身份/模板/模型复制、浮点任务、数值精度或真实时序实验。全部“未证明”不补写成负定理；不把条件证据升级为通用数学计算机制。

## 复现与阶段终止

执行/分析源码、配置、输入/原始哈希与manifest保留。运行示例（务必新目录、先检查闲置GPU）：

```bash
source scripts/chronogeom/activate_remote.sh
python scripts/chronogeom/phase0_04_audit.py --source /ABS/PHASE03/execution --output /ABS/NEW_RUN/audit
# config中的audit/source需指向明确目录；当前脚本包含修正后的rescue No-op
 timeout 600s python scripts/chronogeom/phase0_04_run.py --config /ABS/NEW_CONFIG.json --output /ABS/NEW_RUN/execution
python scripts/chronogeom/phase0_04_analysis.py --run /ABS/NEW_RUN/execution --output /ABS/NEW_RUN/analysis
python -m unittest discover -s tests/chronogeom -p 'test_*.py' -v
```

本轮已完成当前集中复现审计，**不自动追加MIT模型实验**。可以准备ChronoGeom自有Track M与最小TSLib Track T，但正式启动新阶段由用户审阅本报告后决定；本轮只交付性质协议与最小验证设计，不训练。
'''
 (docs/'PHASE0_04_RESULTS.md').write_text(report)
 decision='''# ChronoGeom Research Decision — MIT集中验证结束

2026-10-24。**Decision：项目级选择C；B作为当前GPT-J局部工作解释；A不获支持。** 不将C解释为完全没有发现：已隔离可干预的共享方向及较小独有增量，但不能把这种拟合定义/单模型干预上升为通用几何最优性。

## A / B / C 与证据

| 选项 | 本轮与历史证据 | 判断 |
|---|---|---|
| A：Helix独立稳定优势 | H7/F7共享6维；H7−F7在替换、条件损伤和rescue中方向不稳定，区间跨零；QL弱正LD作用但单独0/85恢复 | 不支持；不能把“QL小作用”写成“QL严格无用” |
| B：主要来自共享周期子空间 | S6在index1/9的swap LD7.448/7.577；H8损伤后S6恢复准确率81.71%/86.59%，仅QL10.98%/9.76% | 支持当前操作性定义的局部解释；共享来自嵌套回归构造，周期方向也可表达大小，不能排他确证 |
| C：形状/容量/训练仍未完全分离 | 单模型/单模板/固定79/20身份；高秩/强扰动和probe监督方式仍混杂；无独立训练机制操作、真实OOD或精度测试 | 作为项目与下一阶段的主判断最稳妥 |

## MIT独立复现的支持与未证明

**支持（受限）**：Phase01/02官方真实缓存中数值结构的留出可预测性；Phase03独立GPT-J输入0..99的PC1相关平方0.901977接近论文0.902，完整0..360频谱有约10整数周期分量；Phase02–04真实新patch/消融/rescue支持早中层数值相关方向参与当前正确题的输出。PCA相关平方与full-hidden留出R²0.1879/独立输入0.12520是不同统计对象，不能互换。

**未证明**：官方全部历史缓存生成revision/采样精确复原；§5 Clock注意力/神经元完整路径；Helix普适必要性、独占充分性/优势；特定十进制频率是唯一计算基；图距离诊断等于在流形干预；浮点精度保持、真正数学/预训练分布外泛化；Circle真实SAE发现及Mistral/Llama因果复现；任何时序迁移。Circle合成校准和官方聚合缓存不能补上这些缺口。

## ChronoGeom原创问题

主线更新为 **Numerical Representation Principles → Mathematical Generalization → Time-Series Forecasting**；保留“numerical representation geometry”作为核心研究对象，但从特定形状转为可检验性质。两条验证轨道独立，不能由算术结果推出预测收益。

问题：在固定信息接口、表示/下游容量、数据和优化预算下，哪些可测性质（无碰撞分辨率、幅值与周期可区分、平移/相位可组合、数值条件稳定性）能预测或改善模型的数学/动力学泛化？竞争解释包括数值监督的信息、额外维度、线性读出可达性、优化便利与真正的变换结构；全部应有失败条件。

| 候选原则（Hypothesis） | 竞争解释/证伪 | Track M最小验证 | Track T最小验证 |
|---|---|---|---|
| 碰撞/精度受控 | 更多位宽/幅值归一化即可解释 | 精确有理网格、dtype、噪声下近邻与加减 | 幅值微扰/单位缩放；纯趋势负对照 |
| 大小与周期信息可区分 | 增维或parser改进而非几何 | 相同周期余数/不同数值，模运算和普通加法分别评估 | value/time分离；同相位不同幅值 |
| 平移/相位可组合 | 普通仿射/低阶多项式probe足够 | 未见δ、操作数组合与多步变换 | 已知周期合成动力学，未见相位/频率 |
| 条件数/范围稳定 | 训练覆盖/标准化充分即可 | 范围外/精度/符号留出 | 未见幅值/采样率、非周期与漂移 |

## 最小实施门槛（Planned，未启动）

先按REPRESENTATION_PROPERTIES_PROTOCOL.md完成解析性质/精度审计；随后可在统一标准probe上启动小规模Track M整数加减/模任务，保留Linear/Fourier/Helix/FoNE真实接口及强非周期基线，固定身份/范围/组合拆分。没有理由现在设计新Encoder。

Track T可以同步准备配置和已知方程合成数据规范，正式预测试验在THUML TSLib内、现有DLinear/小标准backbone、严格trajectory split、预算匹配；在M性质校准及用户确认转入新阶段后启动小pilot。大型预测/真实数据扩展后置。M失败也是有价值的停止/修正信号，不因论文目标而堆模块。

**本轮终止条件已满足：四份交付、完整负结果/失败记录、源代码/配置/Git/原始数据保留。此后不自动追加MIT模型/权重复现，不安排后台训练或自动科研任务。等待用户判断是否启动ChronoGeom原创M/T阶段。**
'''
 (docs/'CHRONOGEOM_RESEARCH_DECISION.md').write_text(decision)
 print('reports written; corrected rows',len(raw),'norm failures',len(norm[norm.relative_norm_error>.01]))
if __name__=='__main__':main()
