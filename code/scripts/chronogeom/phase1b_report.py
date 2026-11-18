"""Report all controlled factors and locked choices, with no test-based redesign."""
from pathlib import Path
import sys,argparse,json,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
from scripts.chronogeom.phase0_03_report import table
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--docs',type=Path,required=True);a=ap.parse_args();r=a.root;docs=a.docs;out=docs/'phase1b/results';out.mkdir(exist_ok=False);q=json.loads((r/'analysis/quality.json').read_text());assert q['gate'];s=pd.read_csv(r/'analysis/summary.csv');ci=pd.read_csv(r/'analysis/paired_trajectory_intervals.csv');rules=pd.read_csv(r/'analysis/frozen_rule_evaluation.csv');ranks=pd.read_csv(r/'analysis/stable_input_design_ranks.csv')
for folder in ['analysis','audit','lock']:
 for p in (r/folder).iterdir():
  if p.suffix in ['.json','.csv']:shutil.copy2(p,out/(folder+'_'+p.name))
for folder in ['train','test']:
 for n in ['metrics.csv','learning_curves.csv','metadata.json']:shutil.copy2(r/folder/n,out/(folder+'_'+n))
for n in ['preflight.json','checkpoint_replay.json','tests_final.txt']:shutil.copy2(r/n,out/n)

precision_path=r/'analysis/input_precision_note.json'
if precision_path.exists():
 documented=q.copy();documented['original_rms_field_scope']='FP64 interface before final FP32 cast';documented['actual_float32_input_rms_error']=json.loads(precision_path.read_text())['actual_float32_rms_error'];(out/'analysis_quality.json').write_text(json.dumps(documented,indent=2)+'\n')

def pivot(frame,index,value='mse'):
 p=frame.pivot_table(index=index,columns='encoding',values=value).reset_index();p.columns.name=None;return table(p)
primary=s[(s.energy=='rms')&(s.cohort=='iid')];b=s[(s.energy=='rms')&(s.task=='B')&(s.n_train==256)];shift=s[(s.task=='D')&(s.cohort=='phase')];types=s[(s.energy=='rms')&(s.cohort=='new_system')&(s.n_train==256)];energy=s[(s.cohort=='iid')&(s.n_train==256)].pivot_table(index=['task','decoder','encoding'],columns='energy',values='mse').reset_index();energy['rms_minus_native']=energy.rms-energy.native
keyci=ci[(ci.n_train==256)&(ci.energy=='rms')&(((ci.task=='B')&(ci.cohort=='iid')&(ci.contrast=='sincos_minus_raw'))|((ci.task=='C')&(ci.cohort=='iid')&(ci.encoding=='sincos')&(ci.contrast=='encoding_minus_none'))|((ci.task=='D')&(ci.cohort=='phase')&(ci.encoding=='sincos')&(ci.contrast=='encoding_minus_none')))]
text='''# Phase1b Controlled Results — Conditional Identifiability and Representation Utility

2026-10-24。**Verified：384次fit（96闭式ridge、288小MLP），768行train/val指标及3456行测试指标全部完成。** CPU2线程训练38.63秒、测试21.26秒，总约59.89秒，无GPU、下载或大型训练。协议5012221，训练实现fa8fadb，验证选择953c4d9在test前Git冻结。Phase1原冻结规则SHA保持不变。

运行根 `/home/home_new/qsmx/runs/ChronoGeom/phase1b/phase1b_20261024_01`，独立TSLib数据/实验扩展 `data_provider/chronogeom_identifiability.py`、`exp/exp_chronogeom_identifiability.py`。读出为标准ridge和Tanh32，不冒称DLinear/TSMixer或新ChronoGeom Encoder；公共TSLib预测逻辑与原模型不改。完整配置/方程见PHASE1B_PROTOCOL.md与run_config.json。

## 三因素的精确分解

记历史信息最优风险R*_H，增加编码后最优风险R*_{H,E}，以及有限训练Decoder的风险R(f)。学得的收益 = **信息收益Δ_info + (历史Decoder excess − 编码Decoder excess)**。编码是否丢信息、与目标函数是否匹配、有限Decoder是否可读，属于不同环节。加入可忽略信息不会使Bayes风险上升；D中的损害来自学得的错误依赖/分布变化，不是信息论反例。

## 任务、Oracle与合法元数据

L16/H32，过去精确无噪声，未来只有测量噪声σ=.05，c~N(0,.5²)、b~U[-.03,.03]、clock在4个周期均匀且与c/b独立。P16已知，H32为2周期而非真实长期预测。A线性趋势；B只在预测段激活已知排程正弦forcing；C过去/未来皆趋势+周期；D训练同C但测试clock相位可错位。C和D训练属于同结构，不能算两个无关动力系统。

B刻意构造scheduled-forcing歧义，不是声称自然序列普遍如此。clock/phase在future noise之前已知，不从未来观测计算；单元测试修改future value不改变features。所有条件同一raw history、基本P和合法future query index；Raw/SinCos/Fourier/Helix/phase-index都保留B所需相位，Raw还保留无关cycle，不同表示的信息Oracle相同。none不额外获得clock。

A/C/D理论R_hist=R_full=.0025；B为.5025→.0025。完整真实测试Oracle风险（包括不同σ/P/new law）如下；数值是理论总体期望与一次finite-test实现，后者不要求严格相等：

'''+table(pd.read_csv(r/'analysis/oracle_risks.csv'))+'''
C/D历史Oracle只从past和已知P解[1,h,sin,cos]，不读取latent phase或future target。新logistic系统Oracle从last state递推已知r=3.2；它获知新law，旧模型不保证zero-shot学会该law，不能把Oracle可达性误当训练模型应知晓的能力。

## 容量、RMS和归一化

原生编码中心化后固定isometry至8维，不逐坐标白化。native保持相对几何，RMS除训练全局向量RMS；所有非零编码的转换前训练RMS误差≤2.23e-16；实际送入模型的FP32编码列RMS最大误差6.61e-9（独立精度审计actual_input_energy.csv）。原metrics的RMS为FP32转换前数值，范围已注明，none0不能凭空匹配。测试仍固定训练factor，不为clock范围变化重新归一化。value只用train past全局mean/std、无RevIN；P中心化，训练恒定的P列零初始化，未见P仍是学习支持变化，不保证Decoder能适配。

per-horizon ridge26×32=832参数；共享pointwise MLP(26→32→1)=897参数，120Adam更新、batch32条轨迹、3seed。各Decoder内部编码/energy参数完全一致，两Decoder约8%差且函数类**非嵌套**，不能把线性优于MLP写成普遍非线性无用。none不活跃权重、intrinsic rank不同，严格有效容量无法完全匹配。N64是256固定前缀，固定更新数不等于相同epochs。

稳定输入谱rank另按相对1e-6重算，ridge逐h给范围、MLP跨全部h；原始first-horizon绝对1e-6 rank在原metrics保留。此补充仅数值审计，不改变任何预测。

'''+table(ranks[(ranks.n_train==256)&(ranks.energy=='rms')][['task','decoder','encoding','input_dimension','rank_min','rank_max','positive_spectrum_condition_max']])+'''
## IID主结果：两个N、两个Decoder、全部六编码

canonical MSE，线性只有一个确定fit，MLP为同一数据的3个初始化均值。每个数使用64独立测试实例，32未来点先均，不当独立样本。完整seed SD、MAE、clean mean误差和raw-unit MSE在summary/metrics CSV。

'''+pivot(primary,['task','n_train','decoder'])+'''
B N256线性：none.52402、Raw.47510、SinCos.00258、Fourier.00263、Helix.00264、phase-index.20781。前者信息充分但线性读不出4cycle sine，后者适当基底可逼近Oracle。Raw与phase-index都native1维/秩1、同RMS和参数，仍有差异，因此不能仅以多维或更多参数解释；但这属于**已知sin函数下的相位折叠/基底匹配**，不证明圆/Helix普适最优。

C N256线性none.00253、SinCos.00253，已有可读历史时额外相位近无收益；C的MLP none.52653、SinCos.03913，理论信息增益为零仍有有限Decoder收益。这否定“只有新信息才能产生编码收益”，并说明收益不等于信息可用性；MLP仍远高于.0025，存在有限函数类/优化excess，不宣称性能上限。

## 原生与RMS能量：全部IID条件

'''+table(energy)+'''
RMS不会创造/消除信息，主要改变conditioning/优化和重复坐标权重分配。闭式ridge对非零缩放大体不敏感（固定λ仍可有微小影响），MLP会变化；不能将所有变化归为几何信息效应。none两energy完全重复，是负控制而非两个独立复现。

## 独立泛化：B全部预设cohort、全部编码

没有重训；P24来自事先声明的合法P字段，不按test误差选周期。noise只改变future测量噪声，past仍无噪；trend×2；B/C phase真值与clock一起移π/3，D只clock移π/2；clock_range+8P；combo同时变化P/σ/trend/phase/cycle；units是iid的配对单位转换，不算新的独立任务。

'''+pivot(b,['decoder','cohort'])+'''
SinCos/Fourier在B的同law参数/phase/cycle变换下可稳定胜Raw，线性MSE接近对应noise floor。Helix无独占优势；其多余clock进展可能带来nuisance，测试RMS不保证仍为1。Raw信息没有消失，只是目标不依赖绝对cycle而有限读出不适配。MLP依然有较大excess，不能将其差距当Bayes信息缺失。

## Misleading D：时钟相关性失效

'''+pivot(shift,['n_train','decoder','energy'])+'''
D N256/RMS/phase：ridge none.00240、SinCos.37757；MLP none.52414、SinCos1.03161。时间即使与训练标签高度相关，真实测试law中given history的Oracle增益仍0；错误学得依赖可造成伤害。C一致phase变化与D clock错位必须分开解释。phase-index线性负效应小，不意味着它普适稳健；有限拟合分配的不同也会影响。

## 未见动力系统：一个logistic新类型，不是四个新law

'''+pivot(types,['task','decoder'])+'''
旧B模型面对clock无关logistic，RMS线性SinCos.53269、none.04184；MLP SinCos.51308、none.02982。固定周期优势没有跨新law成立。该结果是预注册stress，不说明旧学习器应当猜中从未见过的方程，也不能据它否定同law相位编码的有用性。

## 冻结规则：信息gate不是通用收益判定器

953c4d9前只用validation种子均MSE锁定16组choice，primary energy固定RMS。validation-only从六条件选；Oracle-gated仅训练B按理论gap选time条件，否则none，显式依赖已知law。两规则没有测试后更新，即使B在new_system真实gap0也不重新gate。

'''+table(rules[(rules.n_train==256)&rules.cohort.isin(['iid','phase','new_system'])][['task','decoder','cohort','rule','chosen','mse','none_mse','difference_vs_none','theory_information_gain']])+'''
信息gate可说明Bayes可改善空间，但会错过C小MLP的兼容性收益；validation-gain能描述IID有限Decoder收益，却不能保证D错位/新law的稳定效用。没有建立仅凭rank/RMS/几何或单一信息gap即可普适选编码的规则。完整288组frozen-rule评价保留，ex-post best encoding只作参照，不作selection。

## 配对不确定性、质量和负结果

按同test轨迹、相同seed配对；先平均固定3MLP再bootstrap64条轨迹1000次（seed1729）。以下是预设主要contrast（负值为左条件MSE更低），非多重比较显著性声明：

'''+table(keyci[['task','decoder','cohort','contrast','point','low95','high95']])+'''
全部2592组区间保留。32horizon、相同轨迹的N/energy模型、3init和配对units不是额外独立样本；该CI只条件于当前训练实例/模型，不覆盖所有训练集/机制不确定性。3seed SD另存，不选择最有利seed。

43项测试通过，5个checkpoint/数据逐位重放一致；所有3456测试指标有限，RMS匹配有效，单位cohort全部预测与iid最大差0（canonical单位）。无技术失败/超时，全部负效应、错误选择和每实例预测保留。Phase1冻结SHA核验未变。过去无噪、仅正弦/线性校准与一个新law、低预算Decoder、P由任务给定是重要限制。

本轮不启动Phase2。结果可支持条件性的信息/读出分解，不能成为Helix新网络、LLM或真实时序普遍优势的证据。完整复现命令/配置/输入hash见phase1b/results与run目录；必须使用新输出目录，禁止覆盖旧实验。
'''
(docs/'PHASE1B_CONTROLLED_RESULTS.md').write_text(text)
decision='''# Phase1b Research Decision

2026-10-24。**不具备进入原创模型/Encoder设计阶段的证据。停止“单一编码几何分数可决定通用预测效用”的强假设，不继续以堆模块或扩展编码排行榜延长该假设。** 保留本轮明确、受限的正结果，亦不将其当新理论。

## 1. Phase1失败主要因为额外时间信息不足吗？

**不能这样归因。** 原Phase1有噪历史/短窗口、信息不等价、坐标逐维缩放、有限优化与多个机制参数同时变化；law-informed OLS显示clock在mixed仍可降低风险。旧结果无法识别主要原因。本轮A的零信息收益、B的正信息收益、C的冗余和D的误导是在新构造中区分，不能倒推原实验全部是冗余。

## 2. 时间确实必要时，结构编码稳定优于简单编码吗？

**对本轮已知正弦forcing，支持有限肯定。** B理论风险.5025→.0025；同相关相位信息、相同Decoder参数与训练RMS下，SinCos/Fourier逼近线性Oracle，而Raw较差，period/phase/noise/trend/clock范围变化仍可保持优势。phase-index这个简单1维商空间也比1维Raw更好，说明目标相关不变性/解码函数形式比单纯维数或全局单射更重要。

但目标方程本来就含sin，给定sin基底把计算前移、让线性读出可达是标准feature engineering；不是Helix独有，也不是“结构编码总优”。新logistic type中优势翻转，不能外推到未知动力学。

## 3. Decoder是否改变优势？

**明确改变。** C的Bayes信息增益0，ridge none已接近noise floor，而有限pointwise MLP仍强烈受益于相位编码；D同一编码可被训练相关性误导。两读出类832/897参数且不嵌套，因此结论是本研究的Decoder兼容/学习偏置差异，不是所有线性/非线性网络的能力定理。固定120步的MLP高excess可能来自优化和有限函数类，不能归罪编码信息丢失。

## 4. 有可提前计算的效用指标吗？

**有已知law下的信息上界，没有本轮验证出的通用编码选择指标。** Δ_info区分Bayes改善空间，但不会预测C中有限Decoder的额外收益；validation-gain只对训练支持内的学习结果有效，不能保证错位clock或新law。信息+编码保持+Decoder excess须分开。增加可忽略的metadata不会使真正Bayes风险上升，D负效应属于训练/测试关系失效。

Oracle-gated选择使用生成机制知识，是校准而非现实可用的新预测算法；测试看到new law后未改gate。不能把.5 Oracle信息增益冠名为SinCos/Helix几何优势，也不能把一次bootstrap区间不跨0升级为通用规律。

## 5. 超越TTW/FoNE/普通Fourier了吗？

**没有建立这种新颖性。** 本轮是已知相位、目标充分性和标准基底线性化/不变性校准。TTW只测试标量phase-index代理，FoNE完整实现未纳入；无法声称超过它们。[Fourier Features已有频率/学习联系](https://papers.nips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)，[FoNE已有数值Fourier接口](https://proceedings.iclr.cc/paper_files/paper/2026/hash/30253ebdaa75eed08c4b10c75aa8f781-Abstract-Conference.html)。本轮没有独立证明新的数学必要充分条件或未知机制下的可估计指标。

## 支持与否定的结构关系

| 命题 | 证据/结论 |
|---|---|
| 额外信息收益与有限Decoder收益是同一件事 | 否定：C Bayes gap0但小MLP可明显受益 |
| 无条件全局单射/更多维总更有用 | 不支持：Raw保留cycle却难读，phase-index/SinCos可忽略无关cycle；Helix无独占优势 |
| 对已知相位forcing，兼容的周期基有局部效用 | 支持：B在同law独立轨迹/参数变化及两类Decoder下胜Raw；这是给定构造的结果 |
| 训练时间相关性/验证收益足够预测OOD效用 | 否定强版本：D clock错位和logistic新law失败，冻结规则全部保留 |
| RMS/参数匹配后就能完全隔离纯几何 | 不支持：intrinsic rank、函数类/优化、训练支持仍不同；none不能等能量而不加信息 |
| 只要B局部正结果就应设计新Encoder | 不支持：已知Fourier足够完成可达基准，没有新方法科学缺口 |

本轮384fit、9预设cohort、3MLP种子、43测试和5逐位重放是小型校准，不是现实长期forecast证明。units为paired metamorphic而非独立replication；新law只有logistic一种；过去无噪且P已知。交付后停止实验，不自动Phase2或新增MIT复现。

如用户决定继续，应先提出不依赖已知law Oracle、且可独立验证的新问题（如未知机制下的条件歧义估计或达到特定风险所需的Decoder复杂度/样本量），先证明其与标准统计/feature工程的区别。当前不默认再增加编码模块，也不宣称已找到这样的原创问题。
'''
(docs/'PHASE1B_RESEARCH_DECISION.md').write_text(decision);print('controlled and decision reports written')
