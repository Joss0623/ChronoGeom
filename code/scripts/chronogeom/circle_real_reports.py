"""Generate evidence-linked reports; do not change frozen training or evaluation."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np

NAMES={'dec_embedding.temporal_embedding.hour_embed':'Hour lookup','dec_embedding':'Decoder embedding','decoder.layers.0.norm1':'Decoder norm1','decoder.norm':'Decoder final'}
def md(df):
    df=df.copy()
    if 'site' in df:df['site']=df.site.map(lambda a:NAMES.get(a,a))
    def val(x):return ('NA' if pd.isna(x) else f'{x:.5f}') if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def main(r,docs):
    docs.mkdir(parents=True,exist_ok=True)
    import shutil
    figout=docs/'circle_transfer/real01/figures'
    if not figout.exists():shutil.copytree(r/'summary/figures',figout)
    cfg=json.loads((r/'config.json').read_text());meta=json.loads((r/'dataset.json').read_text());s=pd.read_csv(r/'summary/intervention_summary.csv')
    g=pd.read_csv(r/'native_geometry.csv');m=pd.read_csv(r/'summary/matching_diagnostics.csv');ci=pd.read_csv(r/'summary/paired_contrasts.csv')
    counts=m[m.space.isin(['info_match','joint_match','recon_match'])].groupby('space')[['phase_match','reconstruction_match','test_phase_match','test_reconstruction_match','strict_joint_match']].sum().reset_index()
    f=pd.read_csv(r/'forecast.csv');bl=pd.read_csv(r/'baselines.csv');masks=pd.read_csv(r/'input_controls.csv');fail=pd.read_csv(r/'summary/failure_counts.csv')
    shared=f'''2026-10-24。预注册CIRCLE_REAL01_PROTOCOL.md与训练/原生代码固定 **b2656f3**；干预距离序列化单索引修复 **ffba596**（原失败完整保留）。TSLib官方源4e938a1767106324dd753b2a44832bf870a0252e，公共模型/Embed/attention/loader未改。唯一代码/home/home_new/qsmx/pycodes/ChronoGeom，分支chronogeom/circle-real-pilot01。全部原始run：`{r}`。'''
    quality=f'''{md(f)}

{md(bl)}'''
    native_fields=['pca2_fraction','cyclic_distance_rho','adjacent_ratio','nearest_two_hit','wrap_ratio','angular_consistency','winding','radius_cv','harmonic1_fraction','native_circle_gate']
    native=g.groupby(['stage','site'])[native_fields].mean().reset_index()
    scores=m[m.space.isin(['circle','cv_phase','info_match'])].groupby(['site','space'])[['test_phase_r2','test_capture','rank','principal_angle_max']].mean().reset_index()
    maincond=['noop','circle','cv_phase_norm','info_match_norm','joint_match_norm','residual_pca_norm','random0_norm','full_donor']
    inter=s[s.condition.isin(maincond)][['arm','site','condition','recovery','low','high','base_target_mse','recipient_truth_mse','truth_mse_change','phase_error_target_deg']]
    fidelity=s[s.condition.isin(['noop','circle','cv_phase_norm','info_match_norm','harmonic_mean_delta_norm'])][['arm','site','condition','amplitude','amplitude_change','offset_change','trend_change','nonharmonic_mse','recipient_truth_mse','delta_norm','nn_state_distance','target_centroid_distance','mahal']]
    residual=pd.read_csv(r/'summary/residual_fidelity_posthoc.csv');residual=residual[residual.condition.isin(['noop','circle','harmonic_mean_delta_norm'])]
    byshift=pd.read_csv(r/'summary/shift_availability_audit.csv');byshift=byshift[byshift.condition.isin(['noop','circle','cv_phase_norm','info_match_norm'])&byshift.site.eq(cfg['sites'][0])][['arm','shift','condition','recovery','recipient_truth_mse','recipient_time_information']]
    result=f'''# Circle Real-data Pilot01 — ETTh1真实数据验证

{shared}

## 当前结论

**已完成一个真实数据集上的模型训练、原生几何与实际激活干预；没有获得有效Circle机制迁移或预测收益的证据。** 这不是只有接口测试，但基础预测模型质量不足，使其不能承担对MIT机制的强推广判断。所有负结果保留，不为提高分数改变变量、周期、训练步数或test。

## 资产、数据与模型

已有外部TSLib的300个checkpoint未见与本协议匹配的官方Transformer；外部TTW改动模型不按名称猜测载入。依用户授权，GPU4新训三个官方小型Transformer（8177参数），各300步，batch32/Adam .001，L96/label48/H24，learned calendar、GELU、dropout0。没有新Encoder或Circle Loss。

使用[ETDataset官方ETTh1](https://github.com/zhouhaoyi/ETDataset/tree/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small)的OT单变量，hash f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066，与服务器既有ETTh1逐字节相同。服务器访问GitHub超时，Mac临时下载2.59MB官方文件用于核对后同步；无模型权重/大数据下载。最初检查的Electricity副本calendar来源未核验，不用它支持物理相位结论。

官方12/4/4×30天边界：train0:8640、val8640:11520、test11520:14400；尾部3020小时本轮不使用。train均值{meta['mean_train']:.8f}、populationSD{meta['std_train']:.8f}，只由train拟合；下列MSE均为该尺度标准化OT单位，乘SD平方可转回原OT单位。无RevIN。validation划分model-selection23/geometry-fit47/phase-calibration47窗口；test59窗口、9个14天block；报告的是该预设子集，不是官方全滑窗benchmark分数。每个H24完整覆盖24小时，起点时刻固定轮换，不依据观测选择。标准模型/训练/Clock-arm没有未来观测输入。

## 首要限制：基础预测显著弱于简单基线

{quality}

三个模型test MSE .2105–.2333，均值.21886；persistence .03283、seasonal24 .04795，模型平均约为persistence的6.67倍。仅train-mean更差不足以证明预测器合格。Checkpoint只按预留model_val选择，step250/200/200。原训练预算不延长、不用test重新调参。本轮不能据此判断优秀真实预测器是否会学Circle；小预算欠拟合、短验证段、时间漂移和calendar使用失准等解释尚未分开。

固定已训练模型的输入通道消融：

{md(masks.groupby('mode').mse.agg(['mean','std']).reset_index())}

calendar_zero MSE .09467反而低于baseline .21886，提示当前已训练函数使用calendar未稳定泛化；它仍高于persistence。置零为OOD操作，不等价于无calendar重新训练，不能唯一归因或拿它当新方法。

## 几何和监督信息定位

四层、三初始化，init/trained全覆盖。原生Circle gate **12/12未通过**；末层第一谐波份额约.0929（init .0891）、环距离rho .0164，未见合成Pilot02那样明显的周期集中。小时lookup只是被训练的类别查表，不能把它叫MLP或把固定窗口位置sinusoid叫日周期机制。

在小时层，监督Circle的test线性phase R²约.4126、purged-CV相位基.6176；但前者干预恢复较高。读出分数并不充分预测干预效果。后续层phase读出test R²多为负值，显示fit/val定位未稳定跨月份外推。全部数据/控制/半径/角单调/首尾/FFT见CIRCLE_REAL01_GEOMETRY.md。

相位分数匹配val12/12，但test仅7/12通过；重建+phase全条件joint仅 **1/12**通过。不能拿匹配失败的方向宣称已完全控制信息容量，也不能把本轮小时层Circle优于CV解读为特殊几何优势。

{md(counts)}

## 两类干预必须分开

Clock arm只错置±6h时间标签、历史数值不变，再用合法时间输入的激活恢复原模型预测；它检验错误metadata下的模型响应，不是自然预测提升。Natural arm取真实记录中±6h窗口的donor；两者历史、offset、趋势及实际未来都不同，不存在已知纯phase反事实。

**时刻可用性审计：Natural +6h donor的历史包含recipient预测时刻之后6小时内的观测。** 因此它只能用于离线机制分析，不能构成在线合法forecast或准确率提升；Natural −6h donor历史来自已可见过去，但预测对齐/插入激活仍是人工分析。原方案两个方向均保留，不能删除不利方向；逐方向表另存shift_availability_audit.csv。Clock arm与原始forecast基线未泄漏未来数值。该补充可用性说明不改变主实验或排除样本。

{md(inter)}

小时层Circle clock恢复.16145、natural .16579；末层分别−.04653、−.05341。Circle并没有将真实预测恢复成理想相位信号。Natural小时Circle使recipient MSE .21886→.22072，末层→.22691。恢复率衡量靠近另一个**模型预测**，不能等同真实观测精度。分母约.0161，绝对量另报，不能将百分比脱离原预测差异解释。

## 波形保真与统计边界

相位误差、振幅/offset/trend、残差和激活分布距离见CIRCLE_REAL01_INTERVENTIONS.md。实际OT的非24h残差可能是真实信号，不能叫噪声或毛刺自动消除。事后补充相对Noop的非谐波改变及相对实际观测的残差误差，不改变预注册主指标。末层人工纯谐波增量可代数保持原非谐波残差，但整体预测仍可能变差；平滑不等于有用机制。

CI按9个14天block整体重采样1000次seed1729；同源模型/两shift先平均，层/arm分开。59窗口不独立，±6donor及96小时历史仍有重叠；block依赖/边界无法严格排除，区间只是有限块、固定模型与拟合方向下的条件描述，未含重新训练/拟合不确定性，无普适显著性结论。前后半、逐seed、逐shift全表保留。

## 执行、失败与复现

GPU4 RTX4090：训练60.38s、原生审计18.63s、完整干预64.58s（约143.59s活动时间，验证/summary另计），训练峰值张量约82MiB、reserved约94MiB，不含CUDA上下文。三模型900次总更新，无长训练。84,960条记录、1440个预测包、30条件、60个Noop/末层Full控制全部精确0。

**61项测试通过**；四份数据逐位重生成、48组GPU真实干预及逐实例指标重放完全一致；所有rank2基正交误差≤1.78e−15，33984条norm控制最大相对误差6.05e−8。1535条全条件相位无定义仍保留MSE。Circle负恢复1028行，方向背离donor909行，没有因失败筛选样本。

首轮analyze在逐轨迹target-centroid-distance序列化时少了索引[i]而中止；原traceback、首个预测、基与旧源码移入interventions_failed_attempt01/，不删除。修复ffba596只改诊断标量索引，未改模型/方向/指标/数据；首个失败预测与完成版逐位相同。CIRCLE_REAL01_PROTOCOL.md仍是原冻结版本，报告声明此偏差。

完整checkpoint、优化器/RNG、数据、raw CSV、预测、candidate pools、失败日志留外部run；Git小表图在circle_transfer/real01/。本轮未公开推送代码。复现使用新run目录，CUDA_VISIBLE_DEVICES=4、CUBLAS_WORKSPACE_CONFIG=:4096:8，依次：

```bash
source scripts/chronogeom/activate_remote.sh
export CUDA_VISIBLE_DEVICES=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.chronogeom.circle_real_pilot train --config docs/chronogeom/circle_transfer/real01_config.json --run NEW_RUN
python -m scripts.chronogeom.circle_real_pilot geometry --config docs/chronogeom/circle_transfer/real01_config.json --run NEW_RUN
python -m scripts.chronogeom.circle_real_pilot analyze --config docs/chronogeom/circle_transfer/real01_config.json --run NEW_RUN
python -m scripts.chronogeom.circle_real_summary --run NEW_RUN
python -m scripts.chronogeom.circle_real_validate --run NEW_RUN
python -m scripts.chronogeom.circle_real_reports --run NEW_RUN --output-dir NEW_REPORTS
python -m unittest discover -s tests/chronogeom -p 'test_*.py' -v
```

train自动创建NEW_RUN，已有目录会拒绝覆盖。全部源commit、raw文件SHA与最终报告commit由外部run_manifest.json记录，不用自引用文档哈希。

## 决定

**当前真实数据pilot为负/受限结果，不支持声称Circle机制已有效迁移到现实预测。** 已建立真实模型的分析和干预链路，但预测器资格、phase probe跨时间稳定性、严格匹配均不足。下一步首先需要一个能可靠超过persistence的基础预测器，再谈特殊几何；若另行批准扩展，应重冻结训练/验证规则，保留未用尾部3020小时等真正独立测试，不能围绕已看test反复调到正结果。本轮按协议停止，不开发新Encoder，不自动扩训或增加benchmark。
'''
    geom=f'''# ETTh1 Circle Real01 — 原生几何与相位定位

{shared}

## 无监督原生审计

fit小时centroid经中心化SVD确定PCA2，仅正交投影；实际完整16D距离另保留，不白化、不按sin/cos监督旋转。小时标签用于分组和物理排序，不用于挑轴。3seed×init/trained×4固定层，不只展示最佳层。32次fit14天block bootstrap（约4个fit块）检查投影稳定性，1000次hour标签排列是结构负对照；有效样本块少，不能把稳定投影当圆。

三seed均值如下。近圆gate沿用Pilot02原阈值，不针对ETT调整；所有trained格子都失败。

{md(native)}

全seed数值（包括角单调、首尾边比、radiusCV、秩与稳定性）见circle_transfer/real01/tables/native_geometry.csv；原始centroid/distance/FFT与全部排列保留run/native/。PCA2方差占比不足.4，第一谐波约.09、rho接近零，远未组成原生近圆。均值hour geometry失败不证明所有低能量局部phase特征不存在，也不排除其他物理周期；本轮只检验预定义24小时尺度。

## 监督定位及时间泛化

所有子空间2维，仅fit/phase-val定方向；observable nuisance为历史mean/trend/std、相对horizon、month/weekday，不使用未来真实值。CV按连续周3折并purge全部L+H观测交叠，与合成样本的随机CV不同。当前fit/calibration/test属于不同月份，nuisance映射与phase decoder也可能漂移；负test R²是真实外推失败，不事后用test重中心化或重拟合。

{md(scores)}

小时lookup的CV phase读出优于Circle，后续层多数test R²低于0。由监督相位回归定位的Q不代表原始激活自发围成圆；随机类别查表也可读出hour identity。PCA形状、phase可读性和干预功能在本例没有形成统一链条。

## 更强匹配与不可辨识性

phase score/capture matching使用固定候选pool（随机平面及监督锚定混合，整体并非无监督方法）；重建指标仅残差能量份额，非全部任务信息。对同span旋转设排除阈值，实际rank/angles/overlap均保存。跨val/test通过计数（分母12）：

{md(counts)}

Info匹配的test仅7/12，joint仅1/12最终严格成立；不能将其余closest候选称严格等信息/等重建。报告所有失败，不只挑通过格。即使同rank/总扰动相同，拟合容量、条件信息、分布偏离也没有完全控制；Circle在小时层优于CV不能建立其特殊几何优势。

![固定seed0的原生PCA2，上init下trained，无白化](circle_transfer/real01/figures/native_geometry.png)

图严格按小时连接包括首尾，所有乱序与半径差异保留。颜色仅标hour，每列上下坐标范围一致；全部3seed表格与原始向量可追溯，不把一个图当统计证据。官方hour层是nn.Embedding，固定窗口索引Sinusoid仍存在，均不能称MLP自主发现日周期Circle。
'''
    intervention=f'''# ETTh1 Circle Real01 — 相位干预与预测保真

{shared}

## 解释边界

Clock arm的target是同数值历史、合法calendar模型预测；Natural arm的target是实际±6h窗口donor模型预测。都不是已知理想相位反事实。Natural +6h含recipient时刻之后的观测，只作离线机制诊断；−6h历史可用，也不能将人为horizon相位对齐称现实预测方法。主报告的合并shift结果不是在线forecast性能榜。

下面按方向单列小时层，所有其他层/条件亦在shift_availability_audit.csv：

{md(byshift)}

训练/未干预baseline与Clock arm仅使用各自时刻可用历史，没有未来观测输入。Natural donor本身为合法窗口前向，但插入另一个recipient不能自动继承信息合法性或on-manifold性质。Full在末层只证明hook正确恢复donor输出，不证明生成机制纯phase。

## 全部主要干预：相位与保真

phase误差相对target模型的第一谐波；实际观测的phase仅是24h局部OLS描述，没有真正latent phase Oracle。phase_defined要求预测和target幅度均≥.001训练标准化单位；全条件1535条角无定义，不移除MSE。表中NH是预测的非谐波能量，真实序列上越小不自动越好。

{md(fidelity)}

真实Natural arm小时Circle使MSE .21886→.22072，末层→.22691；仅向donor预测略靠近的恢复率不说明保留了recipient重要信息。预测的振幅、offset和趋势同时变动，二维空间不能被视为纯相位坐标。末层Circle/phase-match的负恢复不是隐藏成功；全部seed、两时间半段、随机/打乱/消融对照都保留。

## 非谐波保真的补充审计（事后、非主选择指标）

真实OT的非24h残差可含真实动态而不只是毛刺。为防止把平滑误当成功，另外计算相同联合线性设计Z=[1,h,sin,cos]的残差投影R=I−Z pinv(Z)：`||(prediction-baseline) R||^2`与`||(prediction-observed) R||^2`。这是对已保存输出的描述性补充，不改变预注册评价或选择任何模型。

{md(residual)}

Natural末层Circle的观测残差误差从.008571升至.011460，同时真实整体MSE也变差。谐波均值增量在末层因之后只有线性输出，代数上不改变非谐波残差；其恢复率却为−.29317，整体MSE约.23146。故“毛刺更少/保留残差”不是充分机制或预测价值证明。Clock arm同样报告，不能因噪声更小只选有利条件。

## 匹配、分布距离和统计

33984条norm控制最大相对偏差6.05e−8，全部rank2；但仅控制整段24×16总Frobenius扰动，不能完全控制逐token、phase信息或重建质量。nn距离取固定512个fit原始状态，hour-centroid与ridge Mahalanobis也是诊断；下游离centroid远不等于离真实流形远。小时查表partial混合通常不是24个合法码之一，接入自然donor也不保证联合网络状态自然。

恢复率的条件95%区间与配对Circle减控制差在intervention_summary/paired_contrasts.csv。以9个14天block重采样1000次，模型/shift共享同源窗口先平均；边界历史仍可能相关，CI仅条件、近似、块数少，不称跨架构统计显著。后续层test phase R²负值与joint匹配失败不能隐藏。

## 所有失败

{md(fail)}

合计Circle负恢复1028行、phase背离donor909行；Circle条件自身未出现undefined，其他条件1535条undefined全留。均值消融改变预测不证明Circle唯一必要性；不同模型/层不能视独立。完整30条件/84960行、1440预测NPZ及首轮程序失败均保留。

![全固定层、两arm的恢复率与14天块条件区间](circle_transfer/real01/figures/matched_interventions.png)

joint-match包含失败匹配候选，不是全部严格对照；0线与负结果保留。Full末层精确1是正控制，不是科学成功。

![预先固定的首个test样本、seed0、Natural +6h，包含实际观测和失败预测](circle_transfer/real01/figures/fixed_case.png)

示例未按好坏挑选，展示基线与干预预测的明显偏差；+6h为离线donor，不能拿该图当在线预测效果。其他所有案例另存，不通过换图制造正向迁移。

## 科研判断

本次证据是可重复的有限激活响应，不是特殊Circle机制或有用真实预测收益。下一步应先解决基础模型与时间泛化/匹配可靠性，不能堆Encoder模块或把当前负结果包装成成功。没有自动开始追加训练。
'''
    for name,body in {'CIRCLE_REAL01_RESULTS.md':result,'CIRCLE_REAL01_GEOMETRY.md':geom,'CIRCLE_REAL01_INTERVENTIONS.md':intervention}.items():
        path=docs/name
        if path.exists():raise FileExistsError(path)
        path.write_text(body)
    print('Wrote three real-data reports')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output-dir',default='docs/chronogeom');a=p.parse_args();main(Path(a.run),Path(a.output_dir))
