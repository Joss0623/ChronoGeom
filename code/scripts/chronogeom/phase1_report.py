"""Assemble Phase1 evidence without changing locked models, data or selectors."""
from pathlib import Path
import sys,json,argparse,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scripts.chronogeom.phase0_03_report import table

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--docs',type=Path,required=True);args=ap.parse_args();r=args.root;docs=args.docs;out=docs/'phase1/results';out.mkdir(exist_ok=False);quality=json.loads((r/'analysis/quality.json').read_text());assert quality['gate'];m=pd.read_csv(r/'track_m/metrics.csv');t=pd.read_csv(r/'track_t_heldout/metrics.csv');dev=pd.read_csv(r/'track_t_development/metrics.csv');dg=pd.read_csv(r/'diagnostics/diagnostics.csv');precision=pd.read_csv(r/'diagnostics/precision.csv');mc=pd.read_csv(r/'analysis/m_identity_intervals.csv');tc=pd.read_csv(r/'analysis/t_trajectory_intervals.csv');rule=pd.read_csv(r/'analysis/heldout_rule_evaluation.csv')
 for folder in ['analysis','diagnostics','analytical_supplement','rule_lock']:
  for p in (r/folder).iterdir():
   if p.suffix in ['.json','.csv']:shutil.copy2(p,out/p.name)
 for folder in ['track_m','track_t_development','track_t_heldout']:
  for n in ['metrics.csv','learning_curves.csv','metadata.json']:shutil.copy2(r/folder/n,out/(folder+'_'+n))
 def mt(readout):
  q=m[(m.readout==readout)&m.split.isin(['interpolation','extrapolation'])];a=q.groupby(['task','encoding','split'],as_index=False).agg(n=('n','first'),mse=('mse','mean'),seed_sd=('mse','std'),accuracy=('accuracy','mean'),mae=('mae_original_units','mean'),phase_mae=('phase_mae_radians','mean'));return table(a)
 p=precision[precision.encoding=='scalar'].copy();p=p[~p.input_collided].groupby(['dtype','magnitude'],as_index=False).epsilon.min().rename(columns={'epsilon':'minimum_tested_input_step_separated'})
 diag='''# Phase1 Representation Diagnostics

2026-10-24。**Verified：解析与数值校准已运行；此任务没有模型训练。** FoNE未纳入正式条件：Phase0固定官方代码的parser/符号/科学记数法等语义尚未通过核验，不用自写Fourier primitive冒充FoNE；这不是FoNE性能负结论。TTW-style在本轮仅是固定标量相位索引，不是完整TTW学习周期pattern/residual架构。

M编码的周期[2,5,10,100]是预设整数/十进制尺度；T编码P以小时为单位且由生成任务事先给定。ε为连续数值精度差，三者不混同。每表示训练feature mean/std、固定正交等距lift、训练相邻整数正距离中位数标准化；只比较碰撞/自身相对距离，不把不同坐标范数直接当优劣。

## 原生维度、秩与0..199网格

'''+table(dg)+'''
near collision阈值为该表示训练邻居距离单位下1e-6；并非声称原始浮点位模式完全相等。Scalar与Helix无近碰撞；SinCos/TTW_phase各1900对（周期10），Fourier100对（公共周期100），wrong_sincos周期12有1568对。输出维度8的Fourier有效秩7，整数Nyquist正弦退化；Helix9维秩8。固定lift到9维只统一Probe接口和参数，不提高信息或有效秩。

## 平移、单位与错误周期

全部编码的intrinsic shift误差约10^-14或更小，包括错误周期12；因此自身等变并不能识别任务真正周期10。TTW_index在mod1相位环上等变，但作为欧氏标量在wrap处不连续，不称其全局仿射旋转。Full-vector period10 return对含T100或linear的表示不必为零，不能据此说其没有T10分量。

单位×1000时，周期与magnitude scale同步×1000可复现原表示；盲目只改输入会造成周期/尺度错设。后者首先是接口单位错误压力测试，不是某种表示对合法单位变换天然无效。所有raw误差保留。

## 有限精度

输入/输出分别cast至float64、float32或BF16，再比较ε网格；sin/cos在float64中计算，因此这是**接口量化诊断，不是原生BF16三角核基准**。下面是预设ε列表内能区分输入的最小step，不是全局极限分辨率：

'''+table(p)+'''
例如大数处输入已经cast成同一值，增加Fourier维度或线性坐标不能恢复丢失信息。完整432条encoding/dtype/magnitude/ε记录见precision.csv，输出碰撞与输入碰撞单独列。

## 维度与时间坐标补充校准

下列解析补充在主实验结束后计算，**未改变任何训练条件或锁定规则**；不将其当额外预注册模型比较。更多谐波不必延长公共周期；线性坐标可消除精确算术周期碰撞，但不保证有限读出器泛化。

'''+table(pd.read_csv(r/'analytical_supplement/dimension_sweep.csv'))+'''
物理时间网格0..215小时、训练校准0..95小时的独立诊断：

'''+table(pd.read_csv(r/'analytical_supplement/time_diagnostics.csv'))+'''
none是1个全零占位列、有效秩0，不包含时间信息。shuffle是随机次序负对照，不定义单值E(t)的确定等变性，未伪造其解析性质。时间编码碰撞不意味着整个预测输入碰撞：所有模型仍观察相同的真实历史值。
'''
 (docs/'PHASE1_REPRESENTATION_DIAGNOSTICS.md').write_text(diag)
 mr='''# Phase1 Track M Results

2026-10-24。**Verified：36个固定条件×(1个闭式ridge+3个小MLP种子)=144次fit，四split共576行指标；没有LLM或新Encoder训练。** 全部完成，约18.12秒CPU2线程。代码执行提交b6e9f12；配置见PHASE1_RESEARCH_PROTOCOL.md和run_config.json；原始预测、state、loss曲线全部外置保留。

split seed20261024，0..79按身份64train/8val/8interpolation，extrapolation80..159；binary完整积4096/64/64/6400，unary64/8/8/80。相同数据和lift9维（binary拼接18）；同readout/task所有encoding参数量完全相同，intrinsic rank仍不同。Linear固定λ1e-6闭式；Tanh32 MLP Adam0.01、200步×128、seeds0/1/2，初始化/批次匹配，无early stop、无测试选周期/尺度。原生feature逐维标准化但总体能量未进一步完全匹配，不能声称所有优化条件完全消除。

## 线性读出：全部插值/外推

表中mse为归一化目标向量MSE、mae是原数值单位；classification accuracy为argmax、regression accuracy是round后整数exact。小数格式的0.0000不代表数学上误差严格零，CSV完整精度保留。解析readout无随机seed，std空值不能当三seed零方差复现。

'''+mt('linear')+'''
## 相同参数小型非线性读出：三seed均值与SD

'''+mt('mlp')+'''
## 已验证与负结果

1. Scalar/Helix线性读出可近乎精确外推大小、加减；周期SinCos及phase-index失去全局大小。Fourier公共周期100导致跨训练/外推别名，不能通过任意decoder恢复被碰撞的不同数值。这是已知信息限制的实测，不是原创定理。
2. 正确周期SinCos/Fourier的phase shift线性读出误差近零，wrong_sincos12约0.54 MSE；错误周期自身仍等变，因此intrinsic equivariance不足以预言正确任务行为。
3. 外推mod10小MLP：SinCos/Fourier/Helix均100%，Scalar约10%；TTW-style约76.67%，wrong_sincos约7.5%。同一周期碰撞对模任务可无害，对绝对大小任务则有害，性质应相对目标定义。
4. 更丰富不必更好：线性mod10外推Fourier100%、Helix73.75%。加入大小坐标会改变有限训练集上的最小二乘外推，不能仅由表示包含关系推出已拟合读出器的性能单调性。
5. 表示有信息也不保证神经外推：Scalar线性addition误差近零，而相同Scalar的Tanh Probe外推MAE34.06，Helix46.05。Scalar magnitude的MLP MAE20.78。非线性读出容量/优化/饱和偏置不能忽略；这些失败不是不可解码性证明。

## 不确定性

MSE相对Scalar的差，按共享a/b数值身份节点bootstrap1000次，seed1729；先同seed配对再平均3个MLP。8个插值身份尤其有限；pair不是IID、node扩展没有有限样本覆盖保证，未多重校正，不宣称显著性。完整120组区间在m_identity_intervals.csv，以下为addition和phase_shift所有表示/读出/范围：

'''+table(mc[mc.task.isin(['addition','phase_shift'])][['task','readout','split','comparison','n_cases','n_identities','point','low','high']])+'''
学习曲线、train/val/test原始指标与所有预测保留，不按表现删除seed。这里只有整数区间、模10、δ3；未做真实浮点算术精度/符号/多步组合的训练实验，TaskA量化诊断不能替代它们。
'''
 (docs/'PHASE1_TRACK_M_RESULTS.md').write_text(mr)
 a=dev.groupby(['family','backbone','encoding'],as_index=False).agg(validation_mse=('mse','mean'),validation_seed_sd=('mse','std'));b=t.groupby(['family','backbone','encoding'],as_index=False).agg(test_mse=('mse','mean'),test_seed_sd=('mse','std'),parameters=('parameters','first'));combined=a.merge(b,on=['family','backbone','encoding']);combined.to_csv(out/'validation_vs_heldout.csv',index=False)
 tr='''# Phase1 Track T Results — THUML TSLib controlled pilot

2026-10-24。**Verified：四种独立机制、9编码、2个现有Backbone、3初始化种子，development216 fits、heldout216 fits全部完成。** CPU2线程约28.01+32.62秒，无GPU/新权重/大型真实benchmark。所有时序代码在TSLib目录data_provider/chronogeom_synthetic.py与exp/exp_chronogeom_synthetic.py，脚本只调度；models/DLinear.py、models/TSMixer.py与固定upstream4e938a1767106324dd753b2a44832bf870a0252e逐字无差异，公共run.py/data_factory未改。

## 机制、数据和权限

每task每split独立生成128train/32val/64test实例，每条仅一个L16/H8窗口，随机幅度a、斜率s、截距c、独立σ0.02噪声；共享全局物理时钟并不共享随机轨迹。trend=c+s*t/32；periodic=c+a*sin(2πt/P+φ)；mixed二者相加；regime在τ后周期P/2、幅度1.5a、level+.5、额外trend .3s*max(t−τ,0)/32，相位按累计频率保持连续。所有参数范围/seed/t0/实例ID与公式在协议、metadata和instances.json。

development(P16,τ64,φ0)只评价validation，不读取其test。heldout是独立任务配置(P24,τ80,φ.4、幅度/斜率范围变化)，用自己的train数据训练，test t0=96..191，训练t0=0..95。它是新参数/时钟范围任务，**不是完全未见family或同一模型zero-shot**；窗口起点范围不重叠，但train窗口延至t=118，test窗口可从96开始，部分绝对时间坐标重叠；这是独立实例的起点范围外推，并非全坐标支持集严格互斥。多因素同时变化，不能把误差归因单个参数。P为任务训练前声明的oracle周期，regime之后P/2另有对应分量，不从测试搜索period。

raw values仅按训练历史全局mean/std；数值编码固定不变、无RevIN；本轮只控制时间编码，不搜索value+time组合。future时间合法可用，future观测绝不进forward。数据和目标跨encoding逐位相同的测试通过。

DLinear/TSMixer原模型忽略marks，因此统一输入历史value+8维固定时间covariates，Backbone输出统一channel线性头+合法future时间线性头；不是新ChronoGeom Encoder。DLinear适配后仍线性，TSMixer一层宽16/dropout0。参数按backbone精确相同，none零covariates，shuffle打乱时间关联且保持边际维度；有效输入自由度与优化条件不全相同。所有80步Adam0.003、batch32、最后checkpoint，seeds0/1/2；不增加步数改善测试排名。

## 所有编码、机制与Backbone

validation/test为各任务训练尺度归一化MSE；两阶段任务尺度不同，不能直接将两列差叫单因素退化幅度。原单位MAE/每轨迹预测另存；SD是3个训练种子，不是64×8个独立试验。

'''+table(combined)+'''
## 结构与失效

DLinear在heldout周期任务SinCos0.0509优于none0.0808，wrong_sincos0.0955；mixed中SinCos0.0314、wrong0.1314，存在条件性的周期匹配现象。但TSMixer周期任务none0.0122优于正确SinCos0.0293，wrong_sincos0.0189也优于正确周期；不能将此解释为错误周期普遍更好。趋势任务无真实周期，某错误周期条件仍可能优化更好，周期匹配不能独立决定排名。

Conventional在窗口起点范围外表现很差（如DLinear mixed0.4162、regime1.0788）。其24/168小时日历在168小时出现week wrap；训练窗口最长到118小时，部分测试跨过该边界。这是源码可见的结构支持变化候选解释，未被本轮独立消融隔离，不能记作预注册规则预测成功；它包含t/96因而无时间碰撞，却仍可因有限训练拟合/非线性或线性参数化读出外推失败。Helix也非稳定最优：TSMixer regime0.7172，none0.2967。Shuffle对照部分接近/差于coherent条件，不支持把所有优势归因额外参数，也没有消除优化/输入能量混杂。

## 冻结规则的真正独立检验（Task D）

在development按family做LOFO，仅基于train-signal线性重建NMSE、有效rank、native dimension，比较constant与ridgeλ0.1/1/10，最终DLinear ridge1、TSMixer ridge10。四个heldout任务的编码选择/排序在**da8ee34提交**中已保存，随后才训练或查看其test；heldout_prerun、locked_rule哈希与评估metadata对应。训练信号重建诊断不是完全无监督性质。

'''+table(pd.read_csv(r/'rule_lock/validation_rule_cv.csv'))+'''
'''+table(rule)+'''
锁定选择仅1/8组合胜过none（DLinear periodic）；其余均更差。DLinear预测排序平均Spearman约0.308，TSMixer约−0.246，整体约0.031。验证LOFO比constant略好的选择并未稳定迁移。这个数是8个相关描述项，不作8个独立科学复现；4个family配置太少，不能可靠估计通用meta规律。仍保留所有失败推荐，不用test重选feature、period或规则。

时间编码自身碰撞不能直接推出预测失败：整个输入含raw历史，none仍能从历史推断动力学。已预设的静态train-alignment/rank/dimension诊断不足以预测history-conditioned forecasting。本轮没有证实稳定的结构→预测效用链。

## 轨迹层面的区间

每轨迹先平均8步loss，同一3seed模型的配对差平均后，对64条独立测试轨迹bootstrap1000次（seed1729），仅条件于这些模型；不是把同轨迹窗口/horizon当独立。完整64组区间保留，以下列主要条件相对none：

'''+table(tc[tc.encoding.isin(['sincos','wrong_sincos','fourier','helix'])])+'''
部分区间不含零不能升级为普遍显著优势：3seed不足覆盖训练不确定性，多条件未校正，规则本身跨任务失败。没有大型真实数据、更多Backbone、未知周期估计或长期训练；不把这个低预算结果当所有编码的性能上限。
'''
 (docs/'PHASE1_TRACK_T_RESULTS.md').write_text(tr)
 decision='''# Phase1 Research Decision — Numerical Representation Principles

2026-10-24。**决定：不进入Phase2新方法/Encoder设计。完成Phase1首个预设Pilot，保留负结果，下一步应收窄性质与可证伪问题，而非堆叠模块。**

## A. 跨任务规律是否成立？

Track M支持受限且可预期的目标相关边界：周期碰撞阻碍绝对大小，却可适合mod/phase；线性坐标让大小/加减线性可读。相同表征在不同读出器下外推不同，信息保持不是神经优化/泛化保证。完整Representation Property → Computational Behavior → Forecasting Utility链**未建立**。

Track T未出现跨四机制/两Backbone稳定优胜，错误周期并非总差。预先锁定的诊断规则仅1/8任务×Backbone优于none，未通过独立任务泛化检验。它是本轮最重要负结果，不能事后改规则再声称规律成立。

## B. 适用条件和失效边界

可提前证明/本轮实测：目标需要区分而编码碰撞的数值，任何只读该编码的decoder都存在信息限制；正确周期相位有精确旋转结构，但错误周期也对自身等变；单位/scale必须同步；输入dtype已丢失精度无法靠加维恢复。

尚不能预判：history-conditioned预测的编码排名、不同网络优化/外推、regime变化。因为时间编码不是全部观察接口，raw history可以解除仅由时间坐标看见的歧义。Conventional无碰撞但范围外失败、Scalar-Tanh有信息但加法外推失败，均否定“单个编码几何分数足够”的强假设。

## C. 原创性审查

没有足够证据声称超出已有工作的一般新规律。周期映射/优化的联系已有[Fourier Features, NeurIPS2020](https://papers.nips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)；[FoNE, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/30253ebdaa75eed08c4b10c75aa8f781-Abstract-Conference.html)已研究数值Fourier接口；[Helix原论文](https://arxiv.org/abs/2502.00873)研究LLM中的结构和加法。TTW供稿的时间phase/pattern思想在已有资料中审计，本轮仅标量phase-index代理，不能声称完整TTW对比获胜。FoNE未纳入正式训练，不能声称超过FoNE。

本轮贡献是可复现的公平接口、身份/轨迹分离、预先锁定规则的失败检验与具体反例，尚非顶会方法创新或正式新理论；文献检索为有界核验，不称完整优先权证明。

## D. 下一步建议（仅计划，未执行）

Phase1b候选问题：**给定完整可观察接口和固定decoder，什么条件性目标歧义/变换可读性可以预测泛化？** 先构造“完整输入接口相同但目标不同”的反例/信息下界；数学任务接口是E(a),E(b)，预测接口包含raw history和未来合法covariates，不能只评E(time)。这是看到本轮失败后提出的**新假设**，没有在本轮独立验证，不替换已失败规则。

再预注册：原生/固定RMS能量对照，decoder兼容的平移结构，优化预算/学习曲线校准，逐因素变化（频率、clock range、amplitude分别变），增加独立任务配置和训练seed。在小型既有TSLib模型上验证条件歧义诊断是否优于none与constant规则。未通过前不开发Encoder；不扩大真实Benchmark以寻找正排名。

## 质量与可复现性

所有数据/脚本/配置/state/逐实例预测/学习曲线/失败推荐保存，38项测试通过，144个M fit与432个小TSLib fit全部完成且指标有限；没有LLM下载/GPU/大型训练。CPU两线程训练总墙钟约79秒。参数量在同任务读出器或同Backbone内精确一致；不同native/effective rank、输入能量与少量seed仍限制因果解释。

run根`/home/home_new/qsmx/runs/ChronoGeom/phase1/phase1_20261024_01`。协议998d873、训练实现b6e9f12、锁定规则与heldout预测da8ee34。正式复现依赖这些固定source snapshots；最终交付commit见Git历史。验收包括PHASE1_RESEARCH_PROTOCOL.md与其它四份报告。未自动继续Phase2或扩展MIT实验。
'''
 (docs/'PHASE1_RESEARCH_DECISION.md').write_text(decision)
 print('wrote four reports; rule beatsnone',int((rule.relative_to_none<0).sum()),'/',len(rule),'Mfits144 Tfits432')
if __name__=='__main__':main()
