# OpenClaw Health Check Log - 2026-03-20 23:42 CST

## 检查时间
2026-03-20 23:42 (Asia/Shanghai)

## 执行命令
`openclaw status`

## 检查结果
✅ **服务状态：正常运行**

### 系统信息
- OS: macOS 26.3 (arm64) · Node 23.11.0
- Dashboard: http://127.0.0.1:18789/
- Channel: stable (default)
- Update: pnpm · npm latest 2026.3.13

### 服务状态
- Gateway service: LaunchAgent installed · loaded · running (pid 63055, state active)
- Node service: LaunchAgent not installed
- Agents: 1 active, 1 bootstrap file present, 287 sessions

### 频道状态
- Feishu: ✅ ON · OK · 2/2 账户已配置

### 安全审计
- 0 critical · 2 warn · 1 info
  - WARN: Reverse proxy headers are not trusted
  - WARN: Feishu doc create can grant requester permissions

### 会话状态
- 287 active sessions
- Default model: kimi-k2.5 (205k ctx)

## 结论
OpenClaw 服务运行正常，无需重启。
