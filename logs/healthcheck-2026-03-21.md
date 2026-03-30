# OpenClaw 健康检查日志

**时间**: 2026-03-21 06:23 AM (Asia/Shanghai)
**任务**: 定期健康检查 (cron:bfd0a748-09fc-4152-819d-49a0ee40ad5e)

## 检查结果

### 状态: ✅ 正常

**服务概览**:
- Dashboard: http://127.0.0.1:18789/
- OS: macOS 26.3 (arm64) · Node 23.11.0
- Gateway: 本地运行中 (pid 63055, state active)
- Channel: Feishu 已配置，2个账户正常
- Sessions: 638 active

**安全审计**: 0 critical · 2 warn · 1 info
- WARN: Reverse proxy headers 未配置信任
- WARN: Feishu doc create 权限提醒
- 以上警告不影响正常运行

**结论**: OpenClaw 服务运行正常，无需重启。
