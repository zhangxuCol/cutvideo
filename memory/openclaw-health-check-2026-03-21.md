# OpenClaw 健康检查日志

**检查时间:** 2026-03-21 08:16:00 (Asia/Shanghai)
**检查任务:** cron:bfd0a748-09fc-4152-819d-49a0ee40ad5e

## 检查结果

### 服务状态: ✅ 正常运行

- **Gateway 服务:** 正在运行 (pid 63055, state active)
- **LaunchAgent:** 已安装并已加载
- **Dashboard:** http://127.0.0.1:18789/
- **操作系统:** macOS 26.3 (arm64) · Node 23.11.0

### 渠道状态

| 渠道 | 状态 | 详情 |
|------|------|------|
| Feishu | ✅ ON | 已配置 · 2/2 账户正常 |

### 会话统计

- **总会话数:** 729
- **当前活跃会话:** 多个 direct 会话正在运行
- **默认模型:** kimi-k2.5 (205k 上下文)

### 安全审计

- **严重:** 0
- **警告:** 2
  - WARN: Reverse proxy headers are not trusted (loopback 模式下的预期警告)
  - WARN: Feishu doc create 权限提示
- **信息:** 1

### 注意事项

1. Gateway 当前运行在本地回环模式 (ws://127.0.0.1:18789)
2. Memory FTS 功能不可用 (缺少 fts5 模块) - 不影响核心功能
3. 如需外部访问 Dashboard，需配置 trustedProxies

## 结论

OpenClaw 服务运行正常，无需重启或干预。
