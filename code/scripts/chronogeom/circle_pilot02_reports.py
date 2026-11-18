"""Evidence-linked Pilot02 report generation; never selects a forecasting result."""
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LABELS={'dec_embedding.temporal_embedding.hour_embed':'Hour lookup','dec_embedding':'Decoder embedding','decoder.layers.0.norm1':'Decoder norm1','decoder.norm':'Decoder final'}
def md(df, decimals=4):
    def val(v):
        if isinstance(v,(float,np.floating)):return 'NA' if pd.isna(v) else f'{v:.{decimals}f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def pretty(df):
    df=df.copy()
    if 'site' in df:df['site']=df.site.map(lambda x:LABELS.get(x,x))
    return df

def main(r):
    docs=Path('docs/chronogeom');cfg=json.loads((r/'config.json').read_text());s=pd.read_csv(r/'summary/intervention_summary.csv')
    g=pd.read_csv(r/'native_geometry.csv');n=pd.read_csv(r/'summary/native_summary.csv');match=pd.read_csv(r/'summary/matching_diagnostics.csv')
    d=pd.read_csv(r/'interventions.csv');ci=pd.read_csv(r/'summary/paired_contrasts.csv');rep=pd.read_csv(r/'summary/cohort_repeats.csv')
    rows=[];harm=[]
    for stage in ['init','trained']:
        for si,site in enumerate(cfg['sites']):
            arr=[np.load(r/'native'/f'{seed}_{stage}_baseline_{si}.npz') for seed in cfg['seeds']]
            for a,b in itertools.combinations(range(3),2):
                upper=np.triu_indices(24,1)
                rows.append(dict(stage=stage,site=site,seed_a=a,seed_b=b,distance_spearman=float(spearmanr(arr[a]['distance_matrix'][upper],arr[b]['distance_matrix'][upper]).statistic)))
            for seed,a in zip(cfg['seeds'],arr):
                e=a['harmonic_energy']
                for k in range(1,len(e)):
                    harm.append(dict(stage=stage,site=site,seed=seed,k=k,energy=float(e[k]),nonzero_fraction=float(e[k]/e[1:].sum())))
    # Additional descriptive summaries use all predeclared observations, no reruns or selection.
    pd.DataFrame(rows).to_csv(r/'summary/cross_seed_distances.csv',index=False)
    pd.DataFrame(harm).to_csv(r/'summary/all_harmonics.csv',index=False)
    by=d.groupby(['seed','site','condition'])[['clean_pred_mse','base_clean_pred_mse','phase_error','amplitude_error','nonharmonic_mse','delta_norm']].mean().reset_index()
    by['phase_error']=d[d.phase_defined].groupby(['seed','site','condition']).phase_error.mean().reindex(pd.MultiIndex.from_frame(by[['seed','site','condition']])).to_numpy()
    by['recovery']=1-by.clean_pred_mse/by.base_clean_pred_mse;by.to_csv(r/'summary/intervention_by_model.csv',index=False)
    # Both primary and by-model phase summaries mask only undefined-amplitude angles.
    group=g[g['mode'].eq('baseline')&g.stage.eq('trained')]
    native=n[n['mode'].eq('baseline')][['stage','site','pca2_fraction','cyclic_distance_rho','adjacent_ratio','nearest_two_hit','angular_consistency','winding','radius_cv','harmonic1_fraction','native_circle_gate']]
    stable=group[['seed','site','wrap_ratio','stability_max_angle_deg','eigen_gap23','cyclic_distance_rho_permutation_p','harmonic1_fraction_permutation_p']]
    rank=match.groupby(['site','space'])[['rank','test_phase_r2','test_capture','overlap','principal_angle_max']].mean().reset_index()
    feasible=match[match.space.isin(['recon_match','info_match','joint_match'])].groupby(['site','space'])[['phase_match','reconstruction_match','test_phase_match','test_reconstruction_match','strict_joint_match']].sum().reset_index()
    mainconds=['circle','raw_pca_norm','residual_pca_norm','random0_norm','random1_norm','random2_norm','shuffled_norm','cv_phase_norm','recon_match_norm','info_match_norm','joint_match_norm']
    core=s[s.condition.isin(mainconds)][['site','condition','recovery','low','high','model_sd','phase_error_deg','amplitude_error','nonharmonic_mse']]
    fidelity=s[s.condition.isin(['noop','circle','cv_phase_norm','info_match_norm','conditional_full','conditional_circle','harmonic_mean_delta','harmonic_mean_delta_norm'])][['site','condition','phase_error_deg','amplitude','amplitude_error','offset_error','trend_error','nonharmonic_mse','clean_truth_mse','delta_norm','nn_state_distance','target_centroid_distance','mahal']]
    legacy=s[s.condition.isin(['legacy_circle','legacy_pca','legacy_random0_norm'])][['site','condition','recovery','low','high']]
    ablate=s[s.condition.isin(['noop','circle_ablate','residual_pca_ablate','info_match_ablate','random0_ablate_norm','full_ablate'])][['site','condition','corrupt_truth_mse','amplitude','phase_error_deg','phase_undefined','delta_norm']]
    qdiff=ci[ci.contrast.isin(['circle_minus_cv_phase_norm','circle_minus_info_match_norm'])]
    inp=pd.read_csv(r/'input_forecast.csv');trans=pd.read_csv(r/'linear_transport.csv');follow=pd.read_csv(r/'summary/calendar_follow.csv')
    root=f'`{r}`';shared=f'''2026-10-24。冻结协议/实验代码 commit **94fb6cb44da7840ced2c12fc18e8ff1e6a71ee50**；此前基线0cb4133，原训练代码2915ddc。唯一代码位置`/home/home_new/qsmx/pycodes/ChronoGeom`，TSLib官方源4e938a1767106324dd753b2a44832bf870a0252e。原始新run：{root}。本轮新增预测模型训练0、权重下载0，CPU2线程复用三个已训练及三个初始化checkpoint。所有预测均由现有官方TSLib模型实际前向产生。'''
    geom=f'''# Circle Pilot02 — 原生几何审计

{shared}

## 问题与操作定义

本分析不以sin/cos监督改变或挑选原始隐藏表示。按真实小时聚合24个16维centroid，以fit-centroid中心化SVD的前两轴作**无监督正交PCA2**；test仅投影，不白化、不按拟合角重排、不选择最好看的种子。小时分组/物理顺序用于评价，不用于学习圆形投影。全部未投影centroid、原生欧氏距离、FFT频谱和投影保留于native/。正交投影不保全部距离，因此同时报告full16D距离和PCA2解释率。有效秩阈值、退化处理固定在circle_pilot02_math.py。

原始DGP是24小时正弦+随机offset/趋势/噪声。全部24小时均在预测训练出现；新分析fit192、val96，test_a/b各96条独立轨迹。每个未来24h片段覆盖全部小时。不是未见小时/真实数据泛化，不能覆盖Pilot01排除四小时后的完整重建失败。

## 未训练/已训练对照：三seed均值

距离rho为full16D距离与环测地距离的Spearman；adjacent为相邻小时距离/所有非对角距离；nearest为真实循环两邻居在表征两近邻中的比例。角单调为连续角增量同号比例（全局方向不定）；winding是按真实小时连线的绝对圈数。k1为原始centroid沿hour轴的第一谐波能量/全部非零谐波能量。Nyquist不重复计数。

{md(pretty(native))}

**无任何一个模型×层通过冻结近圆gate。** gate为同时满足PCA2≥.8、rho≥.8、adjacent≤.4、neighbor≥.75、角同向≥.9、|winding|=1、radiusCV≤.25、k1≥.8，并要求跨3seed。它是预设操作性标准，失败不等于所有潜在局部二维特征都不存在。表内winding均值也不能当作单个闭合圆。

末层训练后k1约.519（init约.089）、rho约.568（init约.013），支持训练相关的周期偏向增强；其PCA2仅.619、邻居命中.313、角同向.625、半径CV.353，远未组成稳定原生圆。小时lookup仍主要为不规则高维查表码（PCA2 .317、k1 .163），不是MLP自然画圆。不能把监督投影的相位可读性R²约.614，替换为原生几何证据。

## 首尾、稳定性与排列负对照

下表为训练后的全部seed，32次源轨迹bootstrap的最大principal-angle稳定性摘要按原实现保存；查表层跨上下文确定，所以0°稳定本身不证明圆。seed0总embedding轴稳定性较弱（80.75°），不能挑选有利投影。p为各模型1000次固定随机hour排列的单侧经验值，最小1/1001；它检验小时顺序关联，不是圆的充分检验，未作多重比较校正，不汇总平均p当显著性。

{md(pretty(stable))}

wrap_ratio是23→0距离/其余相邻边平均；约.87–.93的跨seed均值不能挽救邻居/角单调的失败。初始化对应指标完整保留。两独立test cohort、三个seed均保留，后续层rho范围分别约.235–.618、.491–.656；没有只展示最佳层。跨seed只比较轴无关的距离矩阵：初始化相关约−.13至.04，训练后末层约.70至.73，而小时lookup仍约−.09至.05，提示任务相关结构跨seed增强而高维查表码仍不同。结果见summary/cross_seed_distances.csv；不能直接比较不同随机模型的神经元轴。

随机标签排列只重排向量，不重新监督拟合投影；半径CV、PCA方差这类排列不变指标不附会p值。随机初始化具有同架构、同固定位置编码，也没通过gate。监督Circle、CV相位probe属于另一个问题，不能拿来使此图变圆。

## 输入与位置混杂

官方DataEmbedding是value卷积+calendar查表+固定sinusoidal窗口位置编码。窗口内相对时刻在实现上是序列索引绝对位置，不是RoPE或成对relative-attention，也不等同24h物理周期。没有删除它再声称几何自发出现；本轮单独做position_zero敏感性分析，见CIRCLE_PILOT02_RESULTS.md。calendar_zero时hour lookup hook仍执行，随后父模块输出归零，所以该hook可读不代表信息仍进入模型。

## 结论

**有不依赖相位监督投影的周期结构增强证据，没有稳定原生近圆的证据。** 这否定本checkpoint上的强“原生Circle已复现”表述，不排除低能量局部特征、其他架构或真实周期系统。图固定seed0用于展示，完整三seed指标与所有负对照在CSV。

![原始centroid的无监督PCA2：上初始化，下训练后](circle_transfer/pilot02/figures/native_geometry.png)

![原生距离，逐矩阵除以其非对角均值，原始数值另存](circle_transfer/pilot02/figures/distances.png)

图没有白化或按相位监督旋转；每列上下使用相同坐标范围。颜色仅标记hour，连线严格按真实小时，包括首尾；杂乱线条未删。PNG/PDF与figure_provenance.json、原始24点均保留。
'''
    disent=f'''# Circle Pilot02 — 相位信息与预测保真解耦

{shared}

## 预设比较与指标

所有donor配对固定±6小时，c/b相同、噪声耦合；目标为恢复clean模型输出，**恢复率不等同真实预测改善或相位纯度**。恢复率=1−mean(intervened−clean预测)²/mean(corrupted−clean预测)²，可因平坦化而虚高，消融尤其不能这样解释。

逐条24步预测作联合OLS：offset + trend×h + sin(2πh/24) + cos(2πh/24) + residual。报告相对clean预测/真实相位的wrapped角误差、振幅及距真实振幅1误差、offset与trend真值误差、非谐波residual MSE、总MSE。offset在h=0定义，trend单位为每小时。H24上趋势与谐波不是正交设计；使用同一满秩联合分解，系数并非唯一物理因果分量。真实未来均值作主MSE，含噪声观测只用于baseline检查。振幅<1e−4时相位无定义，只排除角度均值，仍保留全部MSE/案例（全条件2305行），不得拿NaN当好相位。

所有主子空间rank2、共同fit192/val96，表示维度16，固定预测器参数8177不变。norm条件按每源轨迹全部24×16的Frobenius扰动匹配Circle，不是逐token或每坐标能量匹配。实际59904匹配行相对norm误差最大1.65e−8，无超1%失败。norm相等也不保证方向/分布偏离相等。

## 相位与非相位指标：全部层

phase_error_deg相对clean预测；MSE相对真实clean未来均值。nn为到固定512个fit原始状态最近距离/fit状态RMS半径；centroid为目标hour中心距离/同一尺度；mahal是pooled within-hour协方差加固定ridge后的平方Mahalanobis距离，均为诊断而非on-manifold证书。

{md(pretty(fidelity))}

小时层Circle：相位误差从90.05°降至15.99°，但平均振幅从.990降至.723；非谐波MSE从.003465升至.073648，offset误差.0727→.2204，trend .00465→.01561。**相位恢复与预测信息保真不同。** 末层Circle也有非谐波残差.027514，虽恢复率.9629。

## 为什么产生幅度损失和毛刺

二维投影替换 h_bad+(h_clean−h_bad)QQᵀ，将donor的两个坐标与recipient的14维信息和其他未替换路径混合；Circle从已知cos/sin监督定位的方向，并非完整hour码或全部预测充分统计量。在lookup上，所有partial Circle/匹配二维混合均远离合法的24个查表向量（outside-centroid-support=100%），这是该**单个查表模块**可严格验证的有限支持事实。在连续下游层，离24个centroid远不等于离数据流形远；nn/mahal也不能证明或排除它。

相同Circle平面内，用拟合谐波均值差Δφ Bphase代替原始donor投影，小时层非谐波MSE .073648→.015067；再把该差的扰动norm匹配原Circle，MSE=.018681、恢复率=.929759（Circle=.825616）、振幅=.861、相位误差7.70°。因此“特殊圆几何决定干预效果”不足；**同一个平面、同样总扰动大小，仅坐标更新规则不同就显著改变副作用**。它是监督构造的人工增量，不是自然激活或新预测方法，不能用来美化原失败。

末层之后恰为线性输出：Δy_h=Δφ_h Bphase W_out。因此拟合均值增量在输出端本来就是纯第一谐波，非谐波MSE维持baseline .003465、offset/trend也可保持。这有直接线性代数解释，不能当模型内部实现圆上旋转的独立证明。早期层经过非线性网络，残差/幅度耦合更明显。

这组观察支持partial替换中信息混合、更新方向、下游非线性兼容性共同参与失真，尚未唯一分解各自因果份额。离自然lookup有限支持与失真同时出现，但harmonic增量也离该支持、失真更小，故单一“off-support导致一切”解释同样不足。

## 同生成分布自然donor

conditional donor固定同c/b与目标clock，重新独立采样观测噪声，执行真实模型前向取激活；不是拿人工均值当自然向量。Full自然替换小时层恢复.9415，末层.9976；小时层conditional_circle与Circle完全相同，因为小时lookup不依赖噪声。下游conditional_circle也几乎相同，没消除partial混合副作用。

自然full与partial的维度/扰动不相等，不能将full更好单独归因于流形。donor完整状态来自原分布，但接入corrupted编码器/未替换路径后不保证联合激活来自自然条件分布；所条件化的是DGP的c/b/clock，不是完整已观测历史。保留所有分布诊断，不宣称自然donor自动on-manifold。

## 消融与失败

本轮消融在corrupted自然输入上向fit mean删除指定坐标。应相对该corrupted任务真值评价，而不是把向clean靠近误当相位恢复。结果为：

{md(pretty(ablate))}

小时层Circle消融原任务MSE从.009979升至.379228，info-match也升至.328469；等扰动random0为.151105。支持相位相关方向参与输出，不证明Circle是唯一必要机制。全消融常产生常数，角无定义；仍保留。Circle所有层中有1行相位方向错误（总embedding），没有负恢复率案例；这并不抹去幅度损失、毛刺、匹配失败与seed变异。完整165888条、全36条件和预测NPZ均保留。

![相位恢复与非谐波保真的权衡](circle_transfer/pilot02/figures/phase_fidelity.png)

点为固定三个模型、两shift、192源轨迹的描述均值，不是跨模型显著性；重合点不删，全部数字见表。
'''
    results=f'''# Circle Transfer Pilot02 — 实验结果

{shared}

## 执行与复现状态

先冻结CIRCLE_PILOT02_PROTOCOL.md再运行collect/geometry/analyze。GPU均繁忙，全部CPU2线程：采集2.30s、原生几何19.84s、干预110.24s；统计/完整性检查另计，无训练任务。Python3.10.4、torch2.6.0+cu124、numpy2.1.3。数据为独立fit192/val96/test_a96/test_b96，每轨迹L24/label12/H24；四个预设层、seed0/1/2、±6h、36条件，**165888条干预、864个预测包**。60条native评价含初始化、训练后、三种mask模式；baseline原生比较24条。所有未来观测为0输入，仅允许已知clock字段，无RevIN。

已有单元测试及新增几何/退化/负对照/接口/匹配/代数审计共**56项通过**。验证五份数据逐位重生成、48组真实checkpoint patch逐位重放、逐实例指标差<1e−12；24个Noop与6个末层Full positive control最大输出差均0；515个Pilot01原文件与六checkpoint哈希不变。检验日志unit_tests.log、validation.json。frozen source逐字节与94fb一致；summary/报告/验证脚本是随后处理，不能追溯性改变冻结指标。

## 1. 原生几何与监督定位是不同结论

原生近圆gate全部失败；末层k1 .519/rho .568提示周期偏向增强，但PCA2 .619、角一致.625不足以叫圆。初始/训练/随机标签/不同seed全部报告于CIRCLE_GEOMETRY_AUDIT.md。相位监督的Circle `phi=(cos,sin) -> R` 与CV逆向ridge `R -> phi` 是两种信息定位方式，不是自发几何发现。

## 2. 等秩/等扰动与更强对照

Circle为新共同fit数据、全部24小时重拟合；CV phase通过fit内三折源轨迹CV选ridge；residual PCA先去nuisance。Random0/1/2、shuffled保留。recon/info/joint从预设2560候选平面选，不依据test/预测结果；池含Circle/CV锚定混合，整体不是完全无监督候选族。与Circle同平面的旋转被排除；有效rank全部2（投影矩阵实际用正交基），不是仅参数声明。norm项每轨迹总扰动匹配，原始未norm行也完整保留。

{md(pretty(core))}

CI为**固定三模型与已拟合基条件下**，192个源轨迹分cohort bootstrap1000次seed1729；同轨迹两shift和三个模型先平均，层分别报告。model_sd是三个模型恢复率的样本SD。不能把165888行、24预测步或模型/层当IID。CI未包含重新训练/重拟合基的不确定性，三个seed不足以推断架构总体；未作多重比较校正，不用窄区间宣布普适显著性。

Circle减其他相位基的配对恢复率差：

{md(pretty(qdiff))}

CV_norm在前三层三seed均优于Circle，在末层三seed均弱于Circle；没有跨层稳定的Circle优势。线性相位分数匹配的控制与Circle小时恢复.8207/.8256、总embedding .7932/.7917、norm1 .9183/.9085、final .9483/.9629，结果有层依赖。完全共同信息量仍未严格匹配，不能反过来宣称两者已完全等价。

各平面实际phase score、重建capture、最大principal angle与Circle重叠（均值，逐seed另表）：

{md(pretty(rank[rank.space.isin(['circle','residual_pca','cv_phase','recon_match','info_match','joint_match'])]))}

capture=投影残差平方能量/全部残差平方能量，不是任务信息量。除rank/总norm外不能同时严格控制所有性质。各层三个模型的匹配通过数（分母3）：

{md(pretty(feasible))}

**Info-score匹配12/12通过val及test阈值，但不是逐小时误差或Shannon信息匹配；与重建联合匹配仅3/12最终通过。** 其余joint候选全部保留并标为失败，不把选出的closest候选称严格对照，也不只挑成功三格宣称优势；候选池有限，不证明其他符合约束的平面不存在。

## 3. 独立重复与旧基保留

旧Pilot01基不重拟合，在两组新轨迹上得到：

{md(pretty(legacy))}

小时层旧Circle .76545重复原.7654，旧PCA .44599、等扰动旧Random .31855接近原.449/.321，说明原局部功能结果可重复。新Circle .82562用全部24小时的独立分析fit，不能写成“修正后的原分数提高”，也不能覆盖原heldout-hour重建失败。

新fit关键比较在两个独立cohort（各96条）分别为：

{md(pretty(rep[rep.condition.isin(['circle','cv_phase_norm','info_match_norm'])][['site','condition','cohort','n','recovery']]))}

逐模型全部36条件见summary/intervention_by_model.csv，逐hour读出误差见phase_error_by_hour.csv；不把几何不规则或不利层剔除。

## 4. 输入calendar、固定位置与层间变换

固定模型输入通道消融，数值历史不变，所有mask同时施加encoder/decoder。原始test MSE（含观测噪声）：

{md(inp[inp.stage.eq('trained')][['seed','mode','mse','clean_mse']])}

baseline均值约.01138；calendar_zero约.70014，position_zero .06629，both_zero1.58308。这是固定已训练函数对输入通道的依赖，zero是OOD操作，不等价训练不含该通道的模型。calendar冲突（+6h）时保持全部数值历史，输出相位移动：

{md(follow)}

约89°而非保持历史相位，支持直接clock利用。所有calendar字段一起变动，因此不能独占归因于hour输入。official hour_embed是nn.Embedding，**不是MLP**；相对窗口索引的sinusoid位置编码仍存在，和24h物理相位不同。mask它会伤害性能，也不能证明circle内部计算。

总embedding→后续层的fit线性映射，在独立test的activation R²：

{md(pretty(trans))}

大量变异可线性传递，后续层周期集中度和干预响应也变化，故不能说每层只原样拷贝；但剩余非线性/归一化也不能自动解释为独立周期运算。当前checkpoint不能识别“没有calendar、没有固定位置时重新训练是否自发学圆”；本轮不为此重新训练。

## 5. 一个必须保留的代数陷阱（事后审计）

Circle前向回归子空间Q覆盖 `R.T @ Phi` 的列空间，因此fit上 `Phi.T @ R @ (I-Q@Q.T) = 0`。正交补的线性phase读出≈0由构造决定；不是独立证明所有相位信息被移除。**事后**补充只做代数/有限集合审计：24个hour centroid在14维补中仍全部不同；在确定性hour查表接口可从残余码用非线性查表恢复hour/phase。这证明“线性phase分数匹配”不能升级成“完整信息匹配”。在连续后续层，centroid不同不足以证明逐状态完全可解码。

原补空间结果phase_complement.csv原样保留，posthoc表与新增代数单元测试明确单列；未训练新的forecast模型或调整主protocol。其他新增描述表cross_seed_distances/all_harmonics只汇总既有数组。

## 6. 保真、负结果与范围

CIRCLE_PHASE_DISENTANGLEMENT.md给出所有phase/幅度/offset/trend/NH/MSE/分布距离和消融。相位恢复不等于幅度和非周期信息保持；同一个Circle平面内改变更新规则、等norm仍可大幅降低毛刺。全部数据/预测/失败保留：1行Circle相位方向错误、2305行全条件相位无定义、9/12 joint匹配失败、原生gate12/12失败；没有为优势改test或调周期。

正交patch只依赖QQᵀ，不能区分同一平面内Circle/ellipse的坐标命名。自然激活仅保证donor来自原模型前向，不保证与recipient其他路径的组合自然。两个独立cohort也只有同一个已知24h DGP；不支持现实forecast提升、LLM完整机制迁移或唯一圆形计算理论。

## 可复现命令

在全新run目录执行（脚本主动拒绝覆盖既有阶段目录；保留当前run）：

```bash
source scripts/chronogeom/activate_remote.sh
python -m scripts.chronogeom.circle_pilot02 collect --config docs/chronogeom/circle_transfer/pilot02_config.json --run NEW_RUN
python -m scripts.chronogeom.circle_pilot02 geometry --config docs/chronogeom/circle_transfer/pilot02_config.json --run NEW_RUN
python -m scripts.chronogeom.circle_pilot02 analyze --config docs/chronogeom/circle_transfer/pilot02_config.json --run NEW_RUN
# summary requires NEW_RUN/config.json copied byte-for-byte from frozen config
python -m scripts.chronogeom.circle_pilot02_summary --run NEW_RUN
python -m scripts.chronogeom.circle_pilot02_validate --run NEW_RUN
python -m unittest discover -s tests/chronogeom -p 'test_*.py' -v
```

core stages并不自动复制config，复现时先创建不存在的新目录并复制冻结config（mkdir应确认成功后再执行下一行）：`mkdir NEW_RUN`，`cp docs/chronogeom/circle_transfer/pilot02_config.json NEW_RUN/config.json`。报告/完整性validator还核对当前run保存的provenance与posthoc表；验证操作不覆盖科研结果。完整原始资产外置，Git只收代码、冻结配置、文档和小表/图/hash。最终报告提交commit在run_manifest.json记录，避免文档自引用哈希。

![所有固定层的匹配干预结果](circle_transfer/pilot02/figures/matched_interventions.png)

图中joint_match包含失败匹配，不是严格对照；置信区间仅源轨迹条件区间，全部seed差异见表，不因图中区间较窄而宣称跨训练稳定。
'''
    decision=f'''# Circle Pilot02 — 科研判断与下一步边界

{shared}

## 五个验收问题

1. **不依赖相位监督标签的原生圆形几何？未获得。** 原始16D centroid的周期关联在训练/后续层增强，是正结果；三个seed×四层均未通过预设原生近圆标准。原始PCA图不是圆，首尾/邻居/角单调/半径也未共同成立。不能用监督sin/cos投影替代发现。
2. **Circle优于匹配相位信息的其他二维表示？没有稳定证据。** 等rank和总扰动下，小时层Circle .8256，CV phase .8971，读出分数匹配.8207；前三层CV更好，末层Circle更好。匹配控制仅覆盖全局线性phase R²，不代表所有信息；重建+phase联合匹配仅3/12通过。这限制独立几何优越性的因果识别，不能宣称已完全排除信息/容量混杂。
3. **为什么相位干预损伤幅度并产生残差？** 两维donor与14维recipient和其他路径混合，不能保持完整合法hour码/完整任务充分信息，下游非线性进一步耦合。相同平面、等norm的谐波均值更新将小时NH MSE从.0736降至.0187、恢复升.9298，表明坐标更新规则和保真很重要。自然full donor更好但16维/扰动不同，不能据此唯一证明流形解释；完整原因尚不可辨识。
4. **哪些支持特殊几何，哪些仅支持一般phase利用？** 无监督周期集中度、相位probe/消融/替换共同支持当前网络的相位相关方向参与预测。只改calendar、历史不动，输出仍随clock约89°，是直接时间信息利用的强证据。没有验证圆上群运算、独特Circle机制、训练无calendar时的自发学习或对真实系统普适性。任意Q→QR不改变QQᵀ；空间内改名Circle/ellipse不会改变正交patch，所以此干预本身无法辨识几何坐标形状。
5. **是否值得真实时序迁移？有条件值得一个最小外部检验。** 迁移的价值是可解释的calendar/phase信息审计，不是已证明Circle新方法有效。优先一份配置完整、已有TSLib真实数据checkpoint，预先指定物理周期，沿用无监督原生几何、监督probe和CV/info-match对照，检查残差/趋势保真。不能直接启动大型benchmark、Circle Loss或新Encoder。

## 可识别与不可识别

已识别：固定checkpoint、当前合成DGP和独立轨迹上，指定接口的激活改动引起可重复预测变化；监督定位方向携带可用phase相关信息；低维更新可损害非phase保真。

未识别：同信息/同重建/同分布偏离条件下圆形状独有优势；未提供calendar的反事实训练；所有隐藏phase是否完全删除；对非平稳真实周期的泛化。Circle补空间零线性phase协方差是构造，24个小时身份仍可保留，不得夸大消融必要性。

输入法定可用未来timestamp与未来数值泄漏不同：本实验只用前者，但clock本身已提供真实phase。学习查表、固定窗口sinusoidal索引、task物理24h周期三者不能混同。小时lookup不是MLP，后续Transformer也是一个架构家族，不代表所有时序模型。

## 建议的最小下一步（本轮未执行）

找一份现有且可追踪的TSLib真实数据模型及其训练/验证/测试chronological split。周期只由训练段/领域定义，小时可用信息与历史输入明确。先冻结一个主要层/两辅助层与同样对照，跨独立日期/周期block估计不确定性，不能把重叠窗口当IID。优先分析真实连续观测与phase/offset/趋势响应，保留不含稳定周期的负例与错周期诊断。

若现有checkpoint缺少所需calendar接口，先报告资产缺口，再另定小模型对照预算；本轮没有申请或启动任何重训。若真实数据中监督phase定位和CV基都无稳定响应，或响应主要造成波形损坏，应停止“Circle特殊机制已迁移”的路线，将资产保留为一般phase信息审计，不靠加模块维持结论。

## 本轮决定

**继续有限的可解释性方法迁移；不进入Circle专有机制/新Encoder设计。** 这是对用户低成本实证目标的正向工程研究交付，并不要求虚构理论新颖性。Pilot01局部功能结果被独立重复，Pilot02将其更准确地限定为一般相位信息及更新兼容性；原生圆与独占几何机制仍未获支持。完成本轮后停止自动加实验，交由用户决定是否启动一个真实数据资产审计。
'''
    if (r/'gpu_replay/completion.json').exists():
        gp=json.loads((r/'gpu_replay/completion.json').read_text())
        results+='\n\n## 可选GPU可移植性核验\n\n用户要求GPU运行后，另存gpu_replay/protocol.json，60s上限，不改变CPU主实验；完成144组/27648条预测，GPU6 RTX4090，wall='+str(gp['wall_s'])+'s，最大CPU/GPU误差='+str(gp['cpu_gpu_max_error'])+'，失败组='+str(gp['failed_groups'])+'。它不是独立统计样本，详见gpu_replay/completion.json（张量显存不含CUDA上下文）。\n'
    outputs={'CIRCLE_GEOMETRY_AUDIT.md':geom,'CIRCLE_PHASE_DISENTANGLEMENT.md':disent,'CIRCLE_PILOT02_RESULTS.md':results,'CIRCLE_RESEARCH_DECISION.md':decision}
    for name,body in outputs.items():
        path=docs/name
        if path.exists():raise FileExistsError(path)
        path.write_text(body)
    print('Wrote',list(outputs))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(Path(p.parse_args().run))
