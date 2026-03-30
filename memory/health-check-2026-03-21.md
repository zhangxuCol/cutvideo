# OpenClaw Health Check Log - 2026-03-21 08:27 CST

## 检查结果

**状态**: ✅ 正常运行

**服务状态**:
- Gateway 服务: 已安装 · 已加载 · 运行中 (pid 63055, state active)
- 节点服务: LaunchAgent 未安装
- 会话数: 740 个活跃会话
- 默认模型: kimi-k2.5

**通道状态**:
- Feishu: ✅ 已启用 · 状态正常 · 2/2 账户已配置

**安全审计**:
- 严重: 0
- 警告: 2
  - 反向代理头不受信任 (gateway.trustedProxies 为空)
  - Feishu 文档创建可授予请求者权限
- 信息: 1

**内存状态**:
- 内存文件: 0
- 缓存: 开启 (0)
- FTS: 不可用 (缺少 fts5 模块)

## 结论

OpenClaw 服务运行正常，无需重启。
