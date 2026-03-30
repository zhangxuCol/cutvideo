# OpenClaw 健康检查日志

## 检查时间
- 时间: 2026-03-20 23:33 CST (Asia/Shanghai)
- 任务ID: cron:bfd0a748-09fc-4152-819d-49a0ee40ad5e

## 检查结果
✅ **OpenClaw 服务运行正常**

### 服务状态
- Gateway 服务: 运行中 (pid 63055, state active)
- LaunchAgent: 已安装并已加载
- 会话数: 278 个活跃会话
- 默认模型: kimi-k2.5

### 通道状态
- Feishu: ✅ 已启用且正常 (2/2 账户已配置)

### 安全审计
- 严重问题: 0
- 警告: 2
  - 反向代理头未配置信任 (仅本地访问可忽略)
  - Feishu 文档创建权限警告
- 信息: 1

### 注意事项
- Gateway 当前仅绑定本地回环地址 (127.0.0.1:18789)
- Memory FTS 功能不可用 (缺少 fts5 模块)
- Tailscale: 关闭状态

## 结论
服务运行正常，无需重启或干预。
