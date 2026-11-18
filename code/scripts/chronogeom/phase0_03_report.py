"""Assemble evidence tables, manifests and the Phase0-03 report from immutable runs."""
from pathlib import Path
import argparse,json,hashlib,shutil
import pandas as pd

def table(df):
    def cell(x):
        if pd.isna(x):return '—'
        if isinstance(x,float):return f'{x:.4f}'
        return str(x).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--docs',type=Path,required=True);args=ap.parse_args();run=args.run;docs=args.docs;dest=docs/'phase0_03/results';dest.mkdir(parents=True,exist_ok=False);a=run/'analysis_retry1';s=pd.read_csv(a/'condition_summary.csv');c=pd.read_csv(a/'paired_intervals.csv');gc=pd.read_csv(a/'group_counts.csv');ge=pd.read_csv(a/'group_effects.csv');raw=pd.read_json(run/'execution/interventions.jsonl',lines=True)
    for folder in [a,run/'supplement_fft']:
        for p in folder.iterdir():
            if p.is_file():shutil.copy2(p,dest/p.name)
    for folder,names in [(run/'execution',['metadata.json','unsupported.json','subspaces.json','config.json']),(run/'fft_execution',['metadata.json','preflight.json'])]:
        for n in names:shutil.copy2(folder/n,dest/(folder.name+'_'+n))
    main=s[(s.cohort=='fixed')&(s.site=='a')&(s.control_seed==42)]
    common=main[main['rank']==7];ops=common[(common.operation=='swap')&(common['mode']=='raw')]
    columns=['layer','method','effect_mean','accuracy','norm_mean','reconstruction_mse']
    eq=common[(common.operation=='swap')&(common['mode']=='equal_helix')]
    abl=main[((main['rank']==7)&main.method.isin(['helix','pca','fourier','random']))|main.method.isin(['full','noop','linear'])];abl=abl[(abl.operation=='ablate_mean')&(abl['mode'].isin(['raw','mean','zero']))]
    ci=c[(c.cohort=='fixed')&(c.site=='a')&(c['mode']=='equal_helix')&(c.bootstrap=='shared_nodes')&(c.bootstrap_seed==1729)]
    pos=s[(s.cohort=='fixed')&(s.method=='helix')&(s['rank']==7)&(s.operation=='swap')&(s['mode']=='raw')]
    groups=ge[(ge.cohort=='fixed')&(ge.site=='a')&(ge.method=='helix')&(ge['rank']==7)&(ge['mode']=='raw')&ge.group.str.startswith('carry_clean')]
    aux=c[(c.cohort!='fixed')&(c.operation=='swap')&(c.comparison=='helix_minus_pca')&(c.bootstrap=='shared_nodes')]
    cs=pd.read_csv(a/'control_seed_sensitivity.csv');cs=cs[(cs.site=='a')&(cs.operation=='swap')&(cs['mode']=='equal_helix')]
    rank=pd.read_csv(run/'supplement_fft/rank_sensitivity_intervals.csv')
    bs=c[(c.cohort=='fixed')&(c.operation=='swap')&(c['mode']=='equal_helix')&(c.comparison=='helix_minus_pca')]
    cases=raw[(raw.cohort=='fixed')&(raw.layer==1)&(raw.site=='a')&(raw.method=='helix')&(raw['rank']==7)&(raw.operation=='swap')&(raw['mode']=='raw')].sort_values('pair_id');example=pd.concat([cases[cases.target_correct].head(2),cases[~cases.target_correct].head(2)])[['pair_id','clean_a','corrupt_a','b','baseline_text','predicted_text','target_correct','ld_change']]
    report='''# ChronoGeom Phase 0-03 实验结果

日期：2026-10-24（Asia/Shanghai）。**Verified：真实冻结 GPT-J 的数值子空间替换、消融与分组重采样已执行；MIT 总体仍为 Partial Replication。** 本轮没有训练新模型、开发 Encoder、下载新权重或执行 TSLib 预测。

## 结论与证据边界

数值拟合出的子空间在早中层 a-token 位置具有条件性的算术因果相关性：替换推动 clean 答案，居中消融降低 clean 准确率，等范数随机方向效果很小。但 **Helix 不具独占优势**：相同有效秩7时 Fourier-only 的替换均值更高；Helix 相对 PCA 的消融差异区间包含零，rank1 时与 Linear/PCA 的差异也不明确。不能宣称完整 Helix/Clock 算法、普遍必要性、数值精度保持或预训练模型 OOD 已获证明。

## A. 研究记录修正

`notes/Research_Progress.md` 的历史顶部被错误沿用（“未下载权重/未执行新干预”）。已改为当前 Phase0-02/03 状态，历史迁移与 Phase0-01 段落明确标注为当时快照，原顶部逐字存入运行目录 `progress_stale_header_before.md`。根研究进度同步更新；没有删除旧原始实验。

## 环境、锁定协议与样本

唯一代码目录 `/home/home_new/qsmx/pycodes/ChronoGeom`，分支 `chronogeom/phase0-03`。推理源码提交 **656b19b9d59ed2b17d271cebde9e0801e7983c59**，预注册见 `PHASE0_03_PROTOCOL.md`；实际设备配置见本目录的 `phase0_03/results/execution_config.json`。原配置包含继承的 Phase0-02 字段，旧 `max_wall_seconds=3600` / `extra_conditions` 未用于本轮；实际限制 `gpu_wall_limit=600`、方法循环以本轮脚本为准。

复用 GPT-J revision `b71ae8bc86cac13154e03e92b5855203086b722e` 的官方 FP16 存储权重，本轮 **BF16 推理、FP32 缓存**，无量化、无优化训练。权重此前已完整 SHA256 校验，本轮复核相同大小/mtime，不重复下载。Python3.10.4 / torch2.6.0+cu124 / transformers5.12.1；GPU5 RTX4090。主实验233.92秒，独立整数捕获16.75秒，合计约250.67秒，峰值主模型11.31GiB，均在600秒预算内；两进程相继退出，不终止其他任务。

沿用 Phase0-02 的128个 Clean/Corrupted 候选及85个联合正确的固定pair，**不因本轮干预结果重新筛选**。拟合为79个 a 身份×4个 b=316提示，主要测试20个 a 身份独立；79/20数字范围仍在0..99，不是模型训练域外。模板为 `Output ONLY a number.\\n{a}+{b}=`，只使用单token操作数与单token答案且布局兼容的样本。Hook 是 **transformer block 的输入残差流**，索引0=embedding后、index1=首个block后，绝非“第1层输出”等模糊命名。所有预设索引[0,1,9,14,20,27]均报告。

额外候选预先写入 manifest，按原模型 clean/corrupt 都正确筛选；全部失败与未选原因也保留。train_ids 仅为拟合身份诊断，不算独立测试。range 对照仅使用拟合 a<80（实际观察上界76），range_in 使用界内留出身份，range_out 使用[80,88,89,96]四个留出身份；77–79处于界外空隙，没有混入范围内。下表列候选、保留和不同操作数身份数：

'''+table(gc[gc.group=='all'])+'''
## B. 等维度与等扰动对照

Helix 基采用固定十进制周期[2,5,10,100]+线性+bias；在整数上 sin(πa) 为零，因此设计列数不能当有效维度：Helix 的中心变化秩8、Fourier-only7、Linear1、Polynomial degree8为8、PCA8、Random8、Shuffled8。共同秩7比较六种可达到秩7的方法；Linear 不补零冒充7维，而在预设index1/9做七种方法真实rank1对照。另报告自然rank8，Fourier/Linear不参与不存在的rank8比较。

所有数值基只用训练数据拟合，方向按**训练拟合状态方差**排序截断；PCA只用同一训练激活；随机为正交方向；Shuffled按数值身份一致置换标签，重复a共享置换，不使用测试标签。相同干预维度不等于全部拟合自由度完全相同（原始基native8/7、PCA数据驱动）；本轮属于秩控制及扰动控制，不能宣称所有统计容量完全匹配。

替换为 h_corrupt + P(h_clean−h_corrupt)。下表 effect 是目标 Logit Difference 的增加，accuracy 是恢复 **clean** 答案的比例。MSE 是独立测试激活的投影重建误差（较低更好），各方法都使用真实 donor 隐藏状态投影；不是仅按数字预测隐藏状态的误差。所有六层和所有方法：

'''+table(ops[columns])+'''
![六层因果对照与消融](phase0_03/results/specificity_ablation.png)

等扰动补充按每个pair、每层，把其他方向实际BF16扰动L2匹配到Helix7。不是只匹配平均范数；6545个正目标干预的相对范数误差中位数2.43e-6、P95 4.01e-5、最大0.002438（0.244%），无误差>1%或无法匹配的非零目标。相同范数仍不保证相同信息量/分布或下游敏感性。没有事后调节重建误差以改善Helix结果。

'''+table(eq[['layer','method','effect_mean','accuracy','norm_mean']])+'''
秩敏感性：rank1时 Helix−Linear 的LD增益差，index1为0.1445，探索性区间[-0.1647,0.4603]；index9为−0.1252，[-0.3175,0.0469]。rank8有更强的Helix替换效果，但不能与rank7 Fourier跨容量宣布胜负。全部rank1/8配对区间：

'''+table(rank)+'''
## C. Helix 子空间消融（条件必要性）

主要消融：h_clean − P(h_clean−μ_train)，仅删除该子空间相对训练均值的变化，保留其正交补。随机/PCA消融使用同秩；Full mean/zero删除整个**同一token位置**激活，不是删除整层全部token；No-op作为零强度控制。

下表 effect 是 clean-vs-corrupt LD **下降**，正值表示损伤；accuracy 是保留原clean答案的比例。初始clean正确率100%，No-op保持100%；模型输出变化及完整预测文本见原始JSONL。各层原始消融结果（Linear为rank1，其余以表中rank为准）：

'''+table(abl[['layer','method','rank','mode','effect_mean','accuracy','norm_mean']])+'''
index1 Helix7消融准确率43/85=50.59%，PCA7为41/85=48.24%，Fourier7为42/85=49.41%；index9 Helix为45/85=52.94%，PCA为48/85=56.47%，Fourier为39/85=45.88%。它们均明显改变部分正确输出，**并不支持Helix独有必要性**。在a位置晚层20/27，即使Full ablation也影响很小：算术信息可能已传到其他位置；不能据此断言这些层对计算整体无用。

等强度PCA/Random补充是沿消融方向缩放的扰动，通常不再是精确投影删除，因此只能解释强度控制，不能把它与严格坐标置零的数学性质混同。替换支持“注入此信息能推动答案”的有限充分性，居中消融支持“此局部变化参与当前任务”的条件必要性；未覆盖备份路径、语义损伤、非线性流形和分布偏移。

## D. 位置、大小、进位、身份与范围

a/b/=三个位置均分析，几何标签始终为a。b位置代表上下文中的a信息，**并非独立拟合b的数值几何**；b与a本身的数值相同/不同不决定因果角色。index0的=位置恒定，native rank0，4个不可执行条件在unsupported.json明确标记，不填充或伪造结果。Helix7替换的所有实际位置结果：

'''+table(pos[['layer','site','n','effect_mean','accuracy','norm_mean']])+'''
效果从a的index1/9到=的index14呈位置依赖，不能把单一token位置的深度衰减当全模型算法结论。b/last的相同模板上下文与有限b采样可能产生相关性。

完整分组计数（包括零样本组）见 `phase0_03/results/group_counts.csv`，逐层逐方法效果见group_effects.csv，均为固定数据，不挑最佳层。固定集clean无进位50对、有进位35对；进位定义为个位相加≥10，另保留clean/corrupt四种进位转换。所有六层 Helix7分组：

'''+table(groups[['group','layer','operation','n','effect_mean','accuracy','norm_mean']])+'''
有进位的clean消融退化均值在index1/9更大，但子组大小、操作数分布和baseline难度不同，未作“进位导致更强依赖”的独立因果结论。clean<25仅1对，不能据数值大小趋势推广。四个数值quartile、身份交叉和各范围完整计数见CSV。

附加队列的 Helix−PCA rank7原始替换差：

'''+table(aux[['cohort','layer','n_pairs','min_a_clusters','insufficient','effect_point','low95','high95']])+'''
range_out只有4个a身份，标记insufficient；即使某个区间不跨零也不据此宣称OOD成立。range比较是**拟合器范围外**，非GPT-J预训练分布外，也没有测试浮点数、精度保持或大整数。

## E. 统计稳健性与不确定性

不把27138条重复层/方法记录当独立样本。每个比较先按相同pair_id配对；主分析对不同操作数**数值身份**统一重采样，clean_a/corrupt_a/b共用同一identity权重，同一pair内部同身份只乘一次，保留跨角色共享。固定集85对仅64个操作数union身份，clean20/corrupt19/b51。

每次2000次percentile resamples，主bootstrap seeds1729/1730/1731；还提供按clean、corrupt、b单角色cluster的敏感性。层不pool，跨层沿用相同重采样权重。仅条件于固定模型、拟合子空间和已筛选提示图；不涵盖重新拟合、模型初始化或预训练数据不确定性。shared-node graph扩展是**探索性重采样**，没有保证有限样本95%覆盖率，也未校正多重比较，因此表中low95/high95是2.5/97.5百分位区间，不能据不跨零宣称统计显著。

相关方法来源：[Owen 2007, The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111)、[Owen & Eckles 2012](https://arxiv.org/abs/1106.2125)。其交叉数组结论不能直接作为本三角色共享节点图bootstrap的理论保证；单角色分组也不能完整解决其他角色依赖。

等扰动（ablate表为Helix真实消融与其他缩放方向对照）、seed1729、所有层的配对差，正值表示Helix替换更强/消融损伤更大：

'''+table(ci[['layer','operation','comparison','n_pairs','effect_point','low95','high95']])+'''
例如index1/9 Helix−PCA等扰动替换分别+2.4408 [0.8417,4.0954]、+2.8188 [1.4719,4.4517]；Helix−Fourier为−0.7211 [−2.0172,0.1453]、−0.6562 [−1.7833,0.1206]。不能声称Helix胜过Fourier。Helix−PCA等扰动消融差−0.0924 [−0.7869,0.5919]、+0.6514 [−0.0090,1.3773]，必要性优势不确定。

全部bootstrap seed和单角色敏感性（Helix−PCA等扰动替换）：

'''+table(bs[['layer','bootstrap','bootstrap_seed','effect_point','low95','high95','zero_weight_draws']])+'''
随机方向/标签控制种子42/43/44，仅预设index1/9，种子敏感性如下；未重新训练GPT-J，不能称为独立模型seed复现：

'''+table(cs[['layer','method','control_seed','n','effect_mean','accuracy','norm_mean']])+'''
样本选择敏感性也不稳定：uniform43及b-stratified44的Helix−PCA区间均跨零；这些仍复用同20个留出身份，不是独立身份复制。保留小样本和所有失败，不能以主集较好结果覆盖这些不确定性。

## F. 独立隐藏状态数值FFT与拟合诊断

A–E质量门通过后补做预设361个连续整数0..360，最后数字token、index0/1；361个全部单token，没有插值/缺失删除，不以非均匀缓存直接FFT。保留原始与线性去趋势所有非负频率功率（未归一化平均|FFT|²，非连续物理谱密度），以及零频。未做周期显著性检验。

index1标准化0..99隐藏状态的PC1与a相关平方**0.901977**，index0为0.860557。与原论文约0.902一致的量是描述性PC1相关平方，**不是留出R²**，也非所有几何机制的复现成功。相同独立输入下，训练原79身份、测试其余21（含45）的Helix隐藏状态重建R²仅0.12520；原Phase0-01 500行缓存多上下文/身份样本指标0.1879仍保留，数据分布不同，不能替换该结果或宣称其错误。

'''+table(pd.read_csv(run/'supplement_fft/standalone_numeric_fits.csv'))+'''
![完整数值FFT诊断](phase0_03/results/standalone_fft.png)

index1去趋势谱有约10.0278整数周期的分量（f=36/361=0.09972），但全谱最大非零功率在最低频1/361，100整数周期不落在361点DFT精确bin上；存在趋势、谱泄漏、谐波和词元结构。没有宣称所有[2,5,10,100]均为模型独立基础周期。数字十进制周期与连续时间序列真实物理周期不可混同。

## 质量、失败、原始记录与可复现性

28项单元测试通过，涵盖有效秩、常量隐藏状态、消融正交补/幂等、BF16范数、共享身份权重、固定分层采样；最终运行记录tests_final.txt。27138条新真实干预记录全部有限；固定85对基线漂移0，所有附加已选样本基线也100%正确；No-op logits变化0、index0 Full替换100%恢复clean答案。主测试包含真实模型hook控制，非仅单元测试。

完整原始行保存每次baseline/预测文本、clean/corrupt数值与答案、LD、正确性、实际/理想扰动范数与舍入误差；成功/失败均保留。以下例子按pair_id各取前2个成功/失败（仅展示，不影响统计），clean目标答案为clean_a+b：

'''+table(example)+'''
主GPU推理正常结束并写好metadata后，首次CPU汇总第二个基线因DataFrame重复set_index引发KeyError；保留execution/analysis部分目录和gpu_job.log。仅把Helix配对索引设置移至方法循环外，**没有改提示、选择集、推理结果或重复主GPU实验**。正确CPU汇总另存analysis_retry1，772个区间完成。临时汇总检查命令出现的shell引号/打印列名错误不影响这些结果。首次supplement未发现尚未执行的FFT缓存，记录executed=false；后续补充FFT另存supplement_fft并明确为executed=true，不覆盖原状态。

运行根目录：`/home/home_new/qsmx/runs/ChronoGeom/phase0_03/phase0_03_20261024_01`。execution为主原始数据；analysis_retry1为正确统计；fft_execution为独立捕获；supplement_fft为CPU补充分析。source_snapshot保存原推理时源码；execution/metadata有输入/代码哈希，原始分析源码哈希对应失败版本；analysis_retry1/provenance.json记录修复后哈希，不回写冒充旧版本。完整SHA256 manifest在run_manifest.json。小型表格、图和配置复制到本Git的phase0_03/results，不提交大激活缓存、模型权重或私人资料。

复现（新目录，先检查闲置GPU并调整device，不覆盖已有run）：

```bash
source scripts/chronogeom/activate_remote.sh
python scripts/chronogeom/phase0_03_run.py --config docs/chronogeom/phase0_03/config.json --output "$HOME/runs/ChronoGeom/phase0_03/NEW_RUN"
# 对原始数据独立重算CPU分析：
python scripts/chronogeom/phase0_03_analysis.py --run /ABS/RUN/execution --output /ABS/NEW_ANALYSIS
python scripts/chronogeom/phase0_03_supplement.py --run /ABS/RUN/execution --standalone /ABS/RUN/fft_execution/standalone_0_360.npz --output /ABS/NEW_SUPPLEMENT
python -m unittest discover -s tests/chronogeom -p 'test_*.py' -v
```

独立FFT脚本额外检查同run根的质量门和剩余预算，并用timeout120s执行；其输入依赖本轮run布局，不是通用训练程序。完整重跑主实验应使用修正后分析版本，推理代码仍由原656b19b9锁定。所有绘图已实际查看；均保留六个预设层，置信区间另图呈现，未以连线代表未测试中间层。

## Phase 0-04 建议与科学判断（Planned）

**值得继续研究“数值相关子空间何时参与算术”，暂不支持开发Helix优先的ChronoGeom Encoder。** 下阶段优先扩大真正独立数值身份/提示模板测试，预先定义身份互斥复制集；以Fourier-only为强基线，检验完整rank8的线性残差方向是否有增量因果作用，做子空间夹角/重叠、分布保持条件替换及恢复/rescue对照。仍须先区分局部干预充分性、条件必要性与完整计算路径，检查备份机制。

在拟合身份重采样外再增加子空间重拟合不确定性；range_out至少扩大身份数，另设计浮点精度和数学操作协议。若需扩展到大数多token、更多上下文或长期GPU，先披露预算再申请。MIT Circle真实SAE资产与Mistral/Llama干预仍未在本轮复现；本轮GPT-J证据不能替代它。Track T坚持TSLib，当前不启动。
'''
    (docs/'PHASE0_03_RESULTS.md').write_text(report)
    prov=dict(analysis_script_sha256=hashlib.sha256(Path(__file__).with_name('phase0_03_analysis.py').read_bytes()).hexdigest(),report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),raw_sha256=hashlib.sha256((run/'execution/interventions.jsonl').read_bytes()).hexdigest(),note='CPU repair only; raw inference unchanged; source snapshot records original code')
    (a/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n');shutil.copy2(a/'provenance.json',dest/'analysis_provenance.json')
    print('report',docs/'PHASE0_03_RESULTS.md','raw_rows',len(raw))
if __name__=='__main__':main()
