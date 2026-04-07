# OpenClaw Health Check Log

**Timestamp:** 2026-03-20 22:32:00 (Asia/Shanghai)
**Job ID:** cron:bfd0a748-09fc-4152-819d-49a0ee40ad5e

## 检查结果

### 服务状态: ✅ 正常

- **Gateway Service:** 运行中 (pid 44044, state active)
- **LaunchAgent:** 已安装并已加载
- **Node Service:** 未安装（正常，非必需）
- **Feishu Channel:** 正常 (2/2 账号已配置)
- **Sessions:** 221 活跃会话

### 安全审计

- **严重问题:** 0
- **警告:** 2
  - 反向代理头未受信任（仅本地访问可忽略）
  - Feishu 文档创建权限警告
- **信息:** 1

### 注意事项

- Gateway 绑定在本地回环地址 (127.0.0.1:18789)
- Memory FTS 不可用 (缺少 fts5 模块)
- 无关键问题需要处理

## 结论

OpenClaw 服务运行正常，无需重启或干预。
