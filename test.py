from evalscope import TaskConfig, run_task

# 这个配置会自动下载数据集并运行
task_cfg = TaskConfig(
    model='ollama/deepseek-coder',  # 或你用的模型
    api_url='http://localhost:11434',  # Ollama地址
    datasets=['swe_bench_verified_mini'],
    limit=1  # 先跑1个测试
)
run_task(task_cfg=task_cfg)