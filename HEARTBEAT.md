# HEARTBEAT.md

## 定时任务

### 每日任务汇总
- **任务ID**: `5cb530ed-d758-41ad-8f5a-645f61e3ade2`
- **名称**: daily-task-summary
- **频率**: 每 24 小时
- **触发**: systemEvent = "daily-task-summary"
- **动作**: 读取当天的 memory/YYYY-MM-DD.md，整理任务时间线，生成摘要

## 对话结束时的自动记录
- 检测对话中的任务关键词
- 记录任务开始/完成时间
- 追加到当天的记忆文件
