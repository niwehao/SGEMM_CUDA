# GEMM 实验 trace

- 每次 GEMM GPU 实验都要分别采集 Nsight Systems 的原生时间线 `.nsys-rep` 和 Nsight Compute 的原生 kernel 数据 `.ncu-rep` 或 `.ncu-repz`。保留原始文件；NCU 数据不是 Nsight Systems 时间线。
- 将每次实验的原始文件放在本项目 `traces/<唯一实验目录>/`，不得覆盖已有文件。从远端复制时核对 SHA-256。记录 workload、采集范围与工具版本；如果只采了部分 kernel launch，明确说明范围。
- 采集、权限或传输失败时保留已有 trace，并明确说明缺失部分，不能将缺少原始文件的实验标记为完整。旧实验如果未采集 trace，不能声称可以事后还原。
- 除非用户另行指定，GPU 实验只使用物理 GPU 0；运行前核对预约和占用。密码不得写入源码、参数、日志或 trace。
