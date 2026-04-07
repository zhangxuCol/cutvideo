# 代码规范

## 快速参考

本文档包含以下主要部分：
- [开发流程](#开发流程)
  - [开发前准备阶段](#开发前准备阶段)
  - [开发新功能规范](#开发新功能规范)
  - [开发完成后的审查流程](#开发完成后的审查流程)
- [编码规范](#编码规范)
  - [通用规范](#通用规范)
  - [组件类规范](#组件类规范)
  - [测试类规范](#测试类规范)
  - [元素定位规范](#元素定位规范)
  - [日志记录规范](#日志记录规范)
  - [装饰器使用规范](#装饰器使用规范)
  - [代码复杂度控制](#代码复杂度控制)
- [质量保证](#质量保证)
  - [测试覆盖率要求](#测试覆盖率要求)
  - [异常处理和错误码规范](#异常处理和错误码规范)
  - [性能和安全规范](#性能和安全规范)
  - [代码审查标准](#代码审查标准)
  - [重构规范](#重构规范)
  - [技术债务管理](#技术债务管理)
- [工程实践](#工程实践)
  - [依赖管理规范](#依赖管理规范)
  - [环境配置规范](#环境配置规范)
  - [文档更新要求](#文档更新要求)
  - [版本控制规范](#版本控制规范)
  - [发布和部署规范](#发布和部署规范)
  - [团队协作规范](#团队协作规范)

## 文档更新日志

### 2024-01-XX
- 添加开发前准备阶段
- 添加开发新功能规范
- 添加开发完成后的审查流程
- 添加代码审查标准
- 添加重构规范
- 添加技术债务管理
- 添加团队协作规范
- 优化文档结构

### 2024-01-XX
- 添加异常处理和错误码规范
- 添加性能和安全规范
- 添加文档更新要求
- 添加代码复杂度控制
- 添加测试覆盖率要求
- 添加依赖管理规范
- 添加环境配置规范
- 添加发布和部署规范
- 添加版本控制规范

---

# 开发流程

## 开发前准备阶段

### 1. 需求分析和理解

在开始编码之前，必须完成以下工作：

- **需求理解**
  - 仔细阅读需求文档，确保完全理解业务逻辑
  - 识别功能的核心目标和边界条件
  - 列出所有功能点和验收标准

- **疑问澄清**
  - 记录需求中的不明确之处
  - 与产品经理或需求提出者沟通确认
  - 确保对需求的理解无歧义

- **需求拆分**
  - 将复杂需求拆分为可独立开发的小任务
  - 识别任务之间的依赖关系
  - 确定开发优先级

### 2. 技术方案设计

在编码前必须进行技术方案设计：

- **方案设计**
  - 设计技术实现方案，包括架构和关键算法
  - 评估方案的可行性和风险
  - 考虑性能、安全、可维护性等因素

- **接口设计**
  - 定义清晰的接口规范
  - 设计合理的参数和返回值
  - 考虑接口的扩展性

- **数据结构设计**
  - 设计合理的数据模型
  - 考虑数据的存储和查询效率
  - 定义数据验证规则

### 3. 影响范围评估

在开发前评估变更的影响范围：

- **代码影响**
  - 识别需要修改的现有代码
  - 评估对现有功能的影响
  - 确定需要同步修改的模块

- **测试影响**
  - 识别需要修改的测试用例
  - 确定需要新增的测试用例
  - 评估回归测试的范围

- **文档影响**
  - 识别需要更新的文档
  - 确定需要新增的文档
  - 规划文档更新的时机

### 4. 依赖关系分析

分析开发任务的依赖关系：

- **任务依赖**
  - 识别任务之间的依赖关系
  - 确定任务的执行顺序
  - 识别可以并行开发的任务

- **资源依赖**
  - 确认所需的开发资源
  - 协调相关开发人员
  - 安排开发时间表

- **外部依赖**
  - 确认外部接口的可用性
  - 协调第三方服务的支持
  - 准备测试数据和环境

## 开发新功能规范

### 方法查找和复用

在开发新功能之前，必须遵循以下步骤：

1. **检查现有方法**
   - 在 `components/` 目录下的组件类中查找是否已有相关方法
   - 在 `utils/page_actions.py` 基础类中查找是否已有通用方法
   - 检查其他组件类中是否有可复用的方法

2. **方法复用优先级**
   - 优先使用 `PageActions` 类中的通用方法
   - 其次使用现有组件类中的方法
   - 只有在确认没有合适方法时，才创建新方法

3. **创建新方法的条件**
   - 确认所有现有方法都无法满足需求
   - 新方法具有通用性，可被其他功能复用
   - 新方法符合项目命名规范和代码风格

4. **方法重写规范**
   - 如果现有方法功能相近但不完全满足需求，优先考虑扩展现有方法
   - 重写方法时保持方法签名一致
   - 添加清晰的文档说明重写原因

### 重复代码抽取规范

在开发过程中，如果发现重复性很高的代码，必须进行抽取和重构：

1. **识别重复代码**
   - 相同或相似的代码块出现3次及以上
   - 多个方法中有相同的逻辑结构
   - 多个组件类中有相似的方法实现

2. **抽取原则**
   - 将重复代码抽取为独立方法
   - 方法应具有通用性和可复用性
   - 方法命名应清晰表达其功能
   - 方法参数应灵活，支持不同场景

3. **抽取位置**
   - 通用方法放在 `utils/page_actions.py`
   - 组件相关方法放在对应的组件类中
   - 工具类方法放在 `utils/` 目录下

4. **示例**

```python
# ❌ 错误做法：重复代码
def get_top_up_button(self):
    """获取充值按钮"""
    locator = self.locators["top_up_button"]
    selector_type = None if locator.startswith("text=") else "css"
    element = self.page_actions.find_element(locator, selector_type)
    if not element:
        raise Exception(f"Element not found for locator: {locator}")
    return element

def get_transaction_history(self):
    """获取交易历史"""
    locator = self.locators["transaction_history"]
    selector_type = None if locator.startswith("text=") else "css"
    element = self.page_actions.find_element(locator, selector_type)
    if not element:
        raise Exception(f"Element not found for locator: {locator}")
    return element

# ✅ 正确做法：抽取公共方法
def get_element(self, locator_name):
    """获取元素的通用方法"""
    self.logger.info(f"获取{locator_name}")
    locator = self.locators[locator_name]
    selector_type = None if locator.startswith("text=") else "css"
    element = self.page_actions.find_element(locator, selector_type)
    if not element:
        raise Exception(f"Element not found for locator: {locator}")
    return element

def get_top_up_button(self):
    """获取充值按钮"""
    return self.get_element("top_up_button")

def get_transaction_history(self):
    """获取交易历史"""
    return self.get_element("transaction_history")
```

### 开发完成后的审查流程

开发完成后，必须按照以下流程进行审查：

1. **代码审查**
   - 检查新增代码是否符合项目编码规范
   - 检查是否有重复代码需要抽取
   - 检查方法命名是否符合命名规范
   - 检查是否使用了现有方法而非重复创建
   - 检查日志记录是否完整
   - 检查异常处理是否合理

2. **需求验证**
   - 确认代码实现了所有需求功能
   - 确认没有遗漏需求点
   - 确认功能实现符合预期
   - 运行相关测试用例验证功能

3. **代码检查**
   - 运行代码检查工具（如 pylint、flake8）
   - 修复所有警告和错误
   - 确保代码无语法错误
   - 确保导入正确无冗余

4. **审查结果处理**

   **情况一：符合需求且无错误**
   - 代码审查通过
   - 需求验证通过
   - 代码检查无报错
   - ✅ 开发完成，可以提交

   **情况二：不符合需求**
   - 代码审查发现问题
   - 或需求验证未通过
   - ⚠️ 需要重新开发
   - 修改代码后重新进入审查流程

   **情况三：代码有报错**
   - 代码检查发现错误
   - 修复所有错误
   - 修复后重新进入审查流程

5. **审查流程图**
```
开始开发
    ↓
开发完成
    ↓
代码审查 ──→ 发现问题 ──→ 修改代码
    ↓                    ↓
需求验证 ────────────┘
    ↓
代码检查 ──→ 发现错误 ──→ 修复错误
    ↓                    ↓
通过所有检查 ──────────┘
    ↓
开发完成
```

**注意事项**：
- 审查必须严格执行，不可跳过任何步骤
- 发现问题必须立即修复，不可累积
- 每次修改后都要重新进行完整审查
- 保持审查记录，便于追溯

**示例**：

```python
# ✅ 正确做法：先查找现有方法
def test_click_button(self):
    """测试点击按钮"""
    # 检查 PageActions 是否有 click_element 方法
    # 检查组件类是否有现成的点击方法
    if hasattr(self.page_actions, 'click_element'):
        self.page_actions.click_element(locator, selector_type)
    else:
        # 只有在没有合适方法时才创建新方法
        self._create_and_click_button(locator)

# ❌ 错误做法：直接创建新方法而不检查现有方法
def test_click_button(self):
    """测试点击按钮"""
    # 直接创建新方法，没有检查是否已有类似方法
    self._create_new_click_method(locator)
```

## 通用规范

### 1. 文件命名

- 测试文件：以 `test_` 开头或以 `_test.py` 结尾
- 组件文件：以 `Component.py` 结尾，使用驼峰命名法
- 工具文件：使用小写字母和下划线

示例：
```
✅ profile_test.py
✅ ProfileComponent.py
✅ page_actions.py
❌ ProfileTest.py
❌ profilecomponent.py
❌ PageActions.py
```

### 2. 类命名

- 测试类：以 `Test` 开头，使用驼峰命名法
- 组件类：以 `Component` 结尾，使用驼峰命名法
- 工具类：使用驼峰命名法

示例：
```python
✅ class TestProfile:
✅ class ProfileComponent:
✅ class PageActions:
❌ class profile_test:
❌ class profilecomponent:
❌ class page_actions:
```

### 3. 方法命名

- 测试方法：以 `test_` 开头，使用小写字母和下划线
- 组件方法：使用小写字母和下划线，以动词开头
- 工具方法：使用小写字母和下划线

示例：
```python
✅ def test_top_up_button(self):
✅ def get_top_up_button(self):
✅ def find_element(self, locator):
❌ def TestTopUpButton(self):
❌ def GetTopUpButton(self):
❌ def FindElement(self, locator):
```

### 4. 变量命名

- 使用小写字母和下划线
- 变量名应具有描述性

示例：
```python
✅ top_up_button
✅ transaction_history
❌ topUpButton
❌ transactionHistory
```

### 5. 常量命名

- 使用大写字母和下划线
- 在 `config/` 目录下定义

示例：
```python
✅ PROFILE_URL
✅ BASE_URL
❌ profileUrl
❅ baseUrl
```

## 组件类规范

### 1. 组件类结构

```python
class ProfileComponent:
    """个人中心页面组件类"""

    def __init__(self, page):
        """初始化组件"""
        self.page = page
        self.page_actions = PageActions(page)
        self.logger = LoggerUtils.get_default_logger()
        self.locators = PROFILE_PAGE
        self.logger.info("初始化个人中心页面组件")

    def get_xxx(self):
        """获取XXX元素"""
        self.logger.info("获取XXX")
        locator = self.locators["xxx"]
        selector_type = None if locator.startswith("text=") else "css"
        element = self.page_actions.find_element(locator, selector_type)
        if not element:
            raise Exception(f"Element not found for locator: {locator}")
        return element
```

### 2. 组件方法规范

- 所有方法都应有文档字符串
- 使用 `self.logger` 记录日志
- 使用 `self.page_actions` 进行元素操作
- 使用 `self.locators` 获取定位器
- 如果元素不存在，抛出异常

### 3. 元素查找规范

```python
# ✅ 正确做法
def get_xxx(self):
    """获取XXX元素"""
    self.logger.info("获取XXX")
    locator = self.locators["xxx"]
    selector_type = None if locator.startswith("text=") else "css"
    element = self.page_actions.find_element(locator, selector_type)
    if not element:
        raise Exception(f"Element not found for locator: {locator}")
    return element

# ❌ 错误做法
def get_xxx(self):
    element = self.page.ele(locator)
    return element
```

## 测试类规范

### 1. 测试类结构

```python
@pytest.mark.profile
class TestProfile:
    """个人中心测试类"""

    @pytest.fixture(autouse=True)
    def setup(self, page):
        """测试夹具，在每个测试方法前执行"""
        self.logger = LoggerUtils.get_default_logger()
        self.page = page
        self.profile = ProfileComponent(self.page)
        self.page.get(PROFILE_URL)
        self.logger.info(f"导航到个人中心页面: {PROFILE_URL}")
        try:
            self.profile.wait_for_page_load()
            self.logger.info("个人中心页面加载完成")
        except Exception as e:
            self.logger.error(f"等待个人中心页面加载时出错: {str(e)}")
            raise

    @pytest.mark.smoke
    def test_xxx(self):
        """测试XXX"""
        self.logger.info("测试XXX")
        xxx = self.profile.get_xxx()
        assert xxx is not None, "未找到XXX"
        self.logger.info("XXX存在")
        xxx.click()
        self.logger.info("成功点击XXX")
        self.page.wait(5)
        self.profile.click_back_button()
        self.logger.info("返回上一页")
        self.page.wait(2)
```

### 2. 测试方法规范

- 所有测试方法都应有文档字符串
- 使用 `@pytest.mark.xxx` 标记测试类型
- 使用 `assert` 进行断言
- 使用 `self.logger` 记录日志
- 测试方法应包含完整的测试流程

### 3. 断言规范

```python
# ✅ 正确做法
assert xxx is not None, "未找到XXX"
assert xxx.text == "预期文本", "文本不匹配"

# ❌ 错误做法
if xxx is None:
    raise Exception("未找到XXX")
```

## 元素定位规范

### 1. 定位器优先级

1. 优先使用 `text=` 定位器
2. 其次使用 CSS 选择器
3. 最后使用 XPath

### 2. 定位器定义

在 `config/locators.py` 中定义：

```python
PROFILE_PAGE = {
    "top_up_button": "text=Top UP",
    "transaction_history": "text=Transaction History",
    "vip_avatar": "div.relative > div > div.gap-xl img",
}
```

### 3. 选择器类型判断

```python
selector_type = None if locator.startswith("text=") else "css"
```

## 日志记录规范

### 1. 日志级别

- `info`: 记录正常操作流程
- `debug`: 记录调试信息
- `warning`: 记录警告信息
- `error`: 记录错误信息

### 2. 日志内容

```python
# ✅ 正确做法
self.logger.info("获取充值按钮")
self.logger.info("成功点击充值按钮")
self.logger.error("未找到充值按钮")

# ❌ 错误做法
self.logger.info("get top up button")
self.logger.info("click top up button")
self.logger.error("top up button not found")
```

## 装饰器使用规范

### 1. 元素等待装饰器

```python
@element_wait_decorator(wait_type="exists", timeout=20, raise_err=False)
def find_element(self, locator, selector_type=None):
    """查找单个元素"""
    element = self._get_element(locator, selector_type)
    return element if element else None
```

### 2. 装饰器参数

- `wait_type`: 等待类型，支持 "clickable"（可点击）和 "exists"（存在）
- `timeout`: 超时时间（秒），默认为 8 秒
- `raise_err`: 未找到元素时是否抛出异常，默认为 False

### 3. 注意事项

- 不要手动调用 `element.wait.clickable()` 或 `element.wait.displayed()`
- 让装饰器自动处理等待逻辑

## 异常处理和错误码规范

### 1. 统一的异常处理策略

**异常处理原则**：
- 预期异常应该被捕获和处理
- 未预期异常应该被记录并重新抛出
- 避免使用裸 except 语句
- 异常处理应该包含恢复逻辑

**异常层级**：
```python
class BaseException(Exception):
    """基础异常类"""
    pass

class BusinessException(BaseException):
    """业务异常基类"""
    pass

class ElementNotFoundException(BusinessException):
    """元素未找到异常"""
    pass

class TimeoutException(BusinessException):
    """超时异常"""
    pass
```

### 2. 自定义异常类型

**异常命名规范**：
- 使用描述性的名称
- 以 `Exception` 结尾
- 体现异常的性质

**异常属性**：
- 包含错误码
- 包含错误信息
- 可选包含原始异常

**示例**：
```python
class ElementNotFoundException(BusinessException):
    """元素未找到异常

    Attributes:
        element_name: 元素名称
        locator: 定位器
        message: 错误信息
    """
    def __init__(self, element_name, locator):
        self.element_name = element_name
        self.locator = locator
        self.message = f"Element {element_name} not found with locator: {locator}"
        super().__init__(self.message)
```

### 3. 错误码定义

**错误码格式**：
- 使用 5 位数字编码
- 格式：`模块码(2位) + 错误类型(1位) + 序列号(2位)`
- 模块码：01-99
- 错误类型：1-系统错误，2-业务错误，3-输入错误

**错误码示例**：
```
01001 - 元素定位失败
01002 - 元素等待超时
01003 - 页面加载失败
02001 - 充值失败
02002 - 登录失败
03001 - 参数验证失败
03002 - 输入格式错误
```

**错误码定义文件**：
```python
# config/error_codes.py

ERROR_CODES = {
    # 元素相关错误 (01)
    "01001": {"code": "ELEMENT_NOT_FOUND", "message": "元素未找到"},
    "01002": {"code": "ELEMENT_TIMEOUT", "message": "元素等待超时"},
    "01003": {"code": "PAGE_LOAD_FAILED", "message": "页面加载失败"},

    # 业务逻辑错误 (02)
    "02001": {"code": "PAYMENT_FAILED", "message": "充值失败"},
    "02002": {"code": "LOGIN_FAILED", "message": "登录失败"},

    # 输入验证错误 (03)
    "03001": {"code": "INVALID_PARAM", "message": "参数验证失败"},
    "03002": {"code": "INVALID_FORMAT", "message": "输入格式错误"},
}
```

### 4. 错误信息格式

**标准格式**：
```python
{
    "error_code": "01001",
    "error_type": "ELEMENT_NOT_FOUND",
    "message": "元素未找到",
    "detail": "充值按钮未找到，定位器：text=Top UP",
    "timestamp": "2024-01-01 12:00:00",
    "stack_trace": "..."  # 仅在开发环境返回
}
```

**日志格式**：
```python
logger.error(
    f"Error occurred: {error_code} - {error_type}",
    extra={
        "error_code": error_code,
        "error_type": error_type,
        "message": message,
        "detail": detail,
        "timestamp": datetime.now().isoformat()
    }
)
```

### 5. 异常处理示例

```python
# ✅ 正确做法
def get_element(self, locator_name):
    """获取元素"""
    try:
        locator = self.locators[locator_name]
        element = self.page_actions.find_element(locator)
        if not element:
            raise ElementNotFoundException(
                element_name=locator_name,
                locator=locator
            )
        return element
    except KeyError as e:
        self.logger.error(f"Locator not found: {locator_name}")
        raise ElementNotFoundException(
            element_name=locator_name,
            locator=None
        ) from e
    except Exception as e:
        self.logger.error(f"Unexpected error: {str(e)}")
        raise

# ❌ 错误做法
def get_element(self, locator_name):
    """获取元素"""
    try:
        element = self.page_actions.find_element(self.locators[locator_name])
        return element
    except:
        return None  # 裸 except，隐藏了所有异常
```

## 代码审查标准

### 审查维度

**功能正确性**：
- [ ] 功能实现符合需求
- [ ] 边界条件处理正确
- [ ] 异常情况处理完善
- [ ] 业务逻辑准确无误

**代码质量**：
- [ ] 命名符合规范
- [ ] 结构清晰合理
- [ ] 无重复代码
- [ ] 复杂度在合理范围

**可维护性**：
- [ ] 注释充分且准确
- [ ] 文档字符串完整
- [ ] 代码易于理解
- [ ] 模块职责单一

**测试覆盖**：
- [ ] 单元测试充分
- [ ] 集成测试完整
- [ ] 边界条件测试
- [ ] 异常路径测试

**安全性**：
- [ ] 无安全漏洞
- [ ] 敏感信息处理正确
- [ ] 输入验证完善
- [ ] 输出编码正确

### 审查流程

1. **自我审查**
   - 开发者完成代码后先自我审查
   - 检查是否符合编码规范
   - 运行所有测试用例
   - 检查代码复杂度

2. **同行审查**
   - 提交 PR 后请求同行审查
   - 审查者检查代码质量
   - 提出改进建议
   - 开发者根据反馈修改

3. **审查确认**
   - 所有问题解决后确认通过
   - 添加审查意见和标签
   - 合并代码到主分支

### 审查反馈

**反馈原则**：
- 提供具体、建设性的反馈
- 指出问题的同时给出建议
- 对代码改进表示赞赏
- 尊重开发者的工作

**反馈格式**：
```markdown
## 问题
1. [严重] 问题描述
   - 位置：文件名:行号
   - 建议：改进建议

2. [一般] 问题描述
   - 位置：文件名:行号
   - 建议：改进建议

## 优点
- 列出代码的优点

## 总体评价
- 对代码的整体评价
```

## 重构规范

### 重构时机

**触发条件**：
- 代码重复率超过 30%
- 方法复杂度超过 20
- 类规模超过 1000 行
- 添加新功能困难
- 修改 bug 时影响范围过大
- 代码难以理解

**最佳时机**：
- 新功能开发前
- Bug 修复后
- 代码审查发现问题时
- 定期技术债务偿还时

### 重构原则

**基本原则**：
- 保持功能不变
- 小步快跑，频繁提交
- 每次重构都有测试保护
- 重构后运行所有测试

**重构准则**：
- 不添加新功能
- 不修改现有功能
- 只改进代码结构
- 提高代码质量

### 重构步骤

1. **准备阶段**
   - 识别需要重构的代码
   - 编写测试用例（如果没有）
   - 确保所有测试通过
   - 创建重构分支

2. **重构阶段**
   - 选择重构目标
   - 逐步进行重构
   - 每次小改动后运行测试
   - 保持测试通过

3. **验证阶段**
   - 运行所有测试
   - 进行代码审查
   - 验证功能未改变
   - 性能测试（如需要）

4. **完成阶段**
   - 提交代码
   - 更新文档
   - 合并到主分支
   - 删除重构分支

### 常见重构技术

**提取方法**：
```python
# ❌ 重构前
def process_user(user):
    if user:
        if user.get("name"):
            name = user["name"].strip().title()
            if len(name) > 50:
                name = name[:50]
            return name
    return None

# ✅ 重构后
def process_user(user):
    if not user:
        return None
    name = user.get("name")
    return normalize_name(name) if name else None

def normalize_name(name):
    """规范化用户名"""
    name = name.strip().title()
    return name[:50] if len(name) > 50 else name
```

**提取类**：
```python
# ❌ 重构前
class Order:
    def __init__(self, items):
        self.items = items

    def calculate_total(self):
        total = 0
        for item in self.items:
            total += item.price * item.quantity
        return total

    def calculate_tax(self):
        total = self.calculate_total()
        return total * 0.1

# ✅ 重构后
class Order:
    def __init__(self, items):
        self.items = items
        self.calculator = OrderCalculator(items)

    def calculate_total(self):
        return self.calculator.calculate_total()

    def calculate_tax(self):
        return self.calculator.calculate_tax()

class OrderCalculator:
    def __init__(self, items):
        self.items = items

    def calculate_total(self):
        return sum(item.price * item.quantity for item in self.items)

    def calculate_tax(self):
        return self.calculate_total() * 0.1
```

## 技术债务管理

### 债务识别

**常见债务类型**：
- 临时解决方案
- 硬编码的值
- 过时的代码
- 缺少测试的代码
- 性能瓶颈
- 不一致的命名
- 缺少文档
- 复杂的逻辑

**识别方法**：
- 代码审查时记录
- 性能分析时发现
- 测试覆盖率报告
- 团队反馈
- 用户反馈

### 债务记录

**记录内容**：
- 债务描述
- 影响范围
- 优先级（高/中/低）
- 预估修复成本
- 创建时间
- 负责人

**记录模板**：
```markdown
## 技术债务描述

### 基本信息
- **ID**: TD-001
- **标题**: 用户登录模块缺少单元测试
- **优先级**: 高
- **状态**: 待处理
- **负责人**: 张三
- **创建时间**: 2024-01-01

### 债务描述
用户登录模块缺少单元测试，覆盖率低于 50%，导致修改时容易引入 bug。

### 影响范围
- 文件: auth/login.py
- 影响功能: 用户登录、密码重置
- 风险等级: 高

### 修复计划
- 添加单元测试用例
- 目标覆盖率: 80%
- 预估工时: 2 天

### 备注
需要在下次迭代中处理，优先级高。
```

### 债务偿还

**偿还策略**：
- 制定偿还计划
- 按优先级逐步偿还
- 每次迭代预留时间
- 偿还后更新文档

**偿还流程**：
1. 选择高优先级债务
2. 制定偿还方案
3. 在迭代中分配时间
4. 实施偿还
5. 验证效果
6. 更新债务记录

**预防措施**：
- 代码审查时发现并记录
- 避免引入新债务
- 定期评估债务状况
- 团队共同维护债务清单

## 性能和安全规范

### 1. 性能优化要求

**代码性能**：
- 避免不必要的循环和递归
- 使用高效的数据结构和算法
- 合理使用缓存机制
- 避免频繁的数据库查询

**资源使用**：
- 及时释放不再使用的资源
- 避免内存泄漏
- 控制并发线程/进程数量
- 优化大文件处理

**页面性能**：
- 减少不必要的页面等待时间
- 使用条件等待而非固定等待
- 优化元素定位策略
- 减少不必要的页面操作

### 2. 资源使用限制

**浏览器资源**：
- 单个测试用例的执行时间不超过 5 分钟
- 单个测试文件的总执行时间不超过 30 分钟
- 并发测试时浏览器实例不超过 4 个

**内存使用**：
- 单个测试用例的内存增长不超过 100MB
- 长时间运行的测试需要定期释放内存
- 避免在循环中创建大对象

**网络资源**：
- 控制并发请求数量
- 实现请求重试机制
- 添加请求超时设置
- 避免重复请求相同资源

### 3. 敏感信息处理

**敏感信息定义**：
- 用户密码和密钥
- 个人身份信息（PII）
- 支付信息
- API 密钥和令牌

**处理原则**：
- 敏感信息不得硬编码在代码中
- 使用环境变量或配置文件管理
- 日志中不得记录敏感信息
- 测试数据必须脱敏处理

**示例**：
```python
# ❌ 错误做法
password = "my_password_123"
logger.info(f"User password: {password}")

# ✅ 正确做法
password = os.getenv("USER_PASSWORD")
logger.info("User password entered")
```

### 4. 输入验证和输出编码

**输入验证**：
- 所有外部输入必须验证
- 验证数据类型和格式
- 检查数据长度和范围
- 过滤特殊字符和恶意内容

**输出编码**：
- 所有输出必须正确编码
- 避免直接输出用户输入
- 使用参数化查询防止注入
- 对特殊字符进行转义

**示例**：
```python
# ❌ 错误做法
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# ✅ 正确做法
query = "SELECT * FROM users WHERE name = %s"
cursor.execute(query, (user_input,))
```

### 5. 安全检查清单

开发完成后必须检查以下安全项：
- [ ] 无硬编码的敏感信息
- [ ] 所有输入都经过验证
- [ ] 所有输出都经过编码
- [ ] 使用了安全的认证机制
- [ ] 实现了适当的权限控制
- [ ] 日志中不包含敏感信息
- [ ] 依赖库无已知安全漏洞
- [ ] 使用了 HTTPS 等安全协议

## 文档更新要求

### 1. 代码变更时的文档同步

**文档同步原则**：
- 代码变更必须同步更新相关文档
- 文档更新应与代码变更在同一 PR 中
- 重大变更需要更新架构文档
- 新增功能需要更新使用文档

**需要更新的文档类型**：
- API 文档：接口变更时更新
- 用户文档：功能变更时更新
- 开发文档：代码结构变更时更新
- 测试文档：测试用例变更时更新

### 2. API 文档维护

**API 文档内容**：
- 方法签名和参数说明
- 返回值类型和说明
- 异常类型和触发条件
- 使用示例

**文档格式**：
```python
def get_element(self, locator_name: str) -> Element:
    """获取页面元素

    Args:
        locator_name: 定位器名称，在 locators 中定义

    Returns:
        Element: 页面元素对象

    Raises:
        ElementNotFoundException: 元素未找到时抛出

    Example:
        >>> element = get_element("top_up_button")
        >>> element.click()
    """
```

### 3. 变更日志记录

**变更日志格式**：
```markdown
## [版本号] - YYYY-MM-DD

### 新增
- 添加了充值按钮功能
- 添加了用户登录验证

### 修改
- 优化了元素定位策略
- 改进了错误处理机制

### 修复
- 修复了页面加载超时问题
- 修复了元素定位失败的问题

### 删除
- 移除了旧的报告生成器
```

### 4. 注释和文档字符串标准

**注释规范**：
- 复杂逻辑必须添加注释
- 注释说明"为什么"而非"是什么"
- 注释与代码同步更新
- 避免无用的注释

**文档字符串规范**：
- 所有公共方法必须有文档字符串
- 使用 Google 或 NumPy 风格
- 包含参数、返回值、异常说明
- 提供使用示例

**示例**：
```python
def find_element(self, locator: str, selector_type: str = None) -> Element:
    """查找页面元素

    Args:
        locator: 元素定位器，支持 text= 或 CSS 选择器
        selector_type: 选择器类型，None 表示 text=，"css" 表示 CSS 选择器

    Returns:
        Element: 找到的元素对象，未找到返回 None

    Raises:
        ValueError: locator 为空时抛出

    Example:
        >>> element = find_element("text=Login")
        >>> element = find_element("div.button", "css")
    """
```

## 代码复杂度控制

### 1. 圈复杂度限制

**复杂度标准**：
- 简单方法：1-5
- 中等方法：6-10
- 复杂方法：11-20（需要重构）
- 极复杂方法：>20（必须重构）

**降低复杂度的方法**：
- 提取方法：将复杂逻辑提取为独立方法
- 提前返回：减少嵌套层级
- 策略模式：使用多态替代复杂条件
- 状态机：使用状态机处理复杂状态转换

### 2. 方法长度限制

**长度标准**：
- 简单方法：1-20 行
- 中等方法：21-50 行
- 长方法：51-100 行（需要拆分）
- 超长方法：>100 行（必须拆分）

**拆分原则**：
- 按功能拆分
- 按抽象层次拆分
- 提取公共逻辑
- 使用辅助方法

### 3. 类规模控制

**规模标准**：
- 小型类：<200 行
- 中型类：200-500 行
- 大型类：500-1000 行（需要拆分）
- 超大类：>1000 行（必须拆分）

**拆分原则**：
- 按职责拆分（单一职责原则）
- 提取接口
- 使用组合替代继承
- 提取公共基类

### 4. 嵌套层级限制

**层级标准**：
- 最大嵌套层级：不超过 4 层
- 推荐嵌套层级：不超过 2 层

**降低嵌套的方法**：
- 提前返回
- 提取方法
- 使用卫语句
- 使用多态

**示例**：
```python
# ❌ 错误做法：嵌套过深
def process_data(data):
    if data:
        if data.get("items"):
            for item in data["items"]:
                if item.get("valid"):
                    if item.get("active"):
                        process_item(item)

# ✅ 正确做法：降低嵌套
def process_data(data):
    if not data or not data.get("items"):
        return

    for item in data["items"]:
        if is_valid_item(item):
            process_item(item)

def is_valid_item(item):
    return item.get("valid") and item.get("active")
```

## 测试覆盖率要求

### 1. 单元测试覆盖率标准

**覆盖率要求**：
- 核心业务逻辑模块：覆盖率 ≥ 80%
- 工具类和公共方法：覆盖率 ≥ 90%
- 组件类：覆盖率 ≥ 70%
- 整体项目覆盖率：≥ 75%

**测试原则**：
- 每个公共方法必须有对应的测试用例
- 每个异常路径必须有测试覆盖
- 边界条件和异常情况必须测试
- 复杂逻辑必须有充分的测试用例

### 2. 集成测试要求

**测试范围**：
- 组件之间的交互
- 页面流程的完整性
- 与外部系统的集成
- 数据的一致性

**测试用例**：
- 正常流程测试
- 异常流程测试
- 边界条件测试
- 并发场景测试

### 3. 测试用例编写规范

**测试用例结构**：
1. 准备（Arrange）：准备测试数据和环境
2. 执行（Act）：执行被测试的功能
3. 断言（Assert）：验证结果是否符合预期

**命名规范**：
- 使用 `test_` 前缀
- 描述清晰，体现测试意图
- 格式：`test_<被测试功能>_<测试场景>`

**示例**：
```python
def test_click_button_success(self):
    """测试点击按钮成功场景"""
    # Arrange: 准备测试数据
    button = self.page.get_button()

    # Act: 执行操作
    button.click()

    # Assert: 验证结果
    assert self.page.is_success_displayed()
```

### 4. 测试数据管理

**测试数据来源**：
- 优先使用 mock 数据
- 其次使用测试数据库
- 避免使用生产数据

**数据管理原则**：
- 测试数据与生产数据隔离
- 每次测试前后清理数据
- 使用固定且可重复的测试数据
- 敏感数据必须脱敏

**测试数据组织**：
- 测试数据文件放在 `tests/data/` 目录
- 使用 JSON 或 YAML 格式存储
- 按功能模块分类组织
- 添加清晰的注释说明

### 5. 测试执行和报告

**测试执行**：
- 本地开发：运行相关模块的测试
- 提交前：运行全部测试
- CI/CD：自动运行全部测试

**测试报告**：
- 生成详细的测试报告
- 包含覆盖率统计
- 标记失败的测试用例
- 提供测试执行日志

**失败处理**：
- 测试失败必须立即修复
- 不能忽略失败的测试
- 更新测试用例时必须更新文档

## 依赖管理规范

### 1. 第三方库引入规范

**引入原则**：
- 优先使用标准库
- 选择成熟、稳定的第三方库
- 避免引入功能重复的库
- 评估库的维护状态和社区活跃度

**引入流程**：
1. 评估需求，确认必须引入第三方库
2. 对比多个候选库，选择最优方案
3. 检查库的许可证和安全性
4. 在团队中讨论并获得批准
5. 更新 requirements.txt 并记录引入原因

### 2. 版本管理策略

**版本锁定**：
- 使用 `requirements.txt` 锁定版本
- 生产环境使用固定版本号
- 开发环境可使用宽松版本号

**版本格式**：
```text
# 固定版本（推荐生产环境使用）
pytest==7.4.0

# 兼容版本（推荐开发环境使用）
pytest>=7.4.0,<8.0.0

# 最低版本（不推荐）
pytest>=7.4.0
```

**版本更新**：
- 定期检查依赖库的更新
- 评估更新带来的影响
- 重要更新需要充分测试
- 记录版本变更和原因

### 3. 安全漏洞检查

**检查工具**：
- 使用 `pip-audit` 检查已知漏洞
- 使用 `safety` 检查依赖安全性
- 使用 `bandit` 检查代码安全问题

**检查频率**：
- 每次引入新依赖时检查
- 定期（如每月）检查现有依赖
- CI/CD 流程中自动检查

**漏洞处理**：
- 发现漏洞立即评估影响
- 高危漏洞必须立即修复
- 更新到安全版本或寻找替代方案
- 记录漏洞处理过程

### 4. 依赖更新流程

**更新检查清单**：
- [ ] 确认更新的必要性
- [ ] 查看更新日志和变更说明
- [ ] 评估向后兼容性
- [ ] 在测试环境充分测试
- [ ] 更新相关文档
- [ ] 记录更新原因和影响

## 环境配置规范

### 1. 开发环境配置

**配置文件**：
- 使用 `.env` 文件存储环境变量
- 将 `.env.example` 提交到版本控制
- 将 `.env` 添加到 `.gitignore`

**配置项**：
```bash
# .env.example
DEBUG=True
LOG_LEVEL=DEBUG
BASE_URL=https://dev.example.com
DB_HOST=localhost
DB_PORT=5432
```

### 2. 测试环境配置

**配置要求**：
- 使用独立的测试数据库
- 使用模拟的外部服务
- 配置详细的日志级别
- 禁用或限制发送邮件等外部通知

### 3. 生产环境配置

**配置要求**：
- 使用环境变量而非配置文件
- 禁用调试模式
- 使用生产数据库
- 配置适当的日志级别
- 启用性能监控

**安全配置**：
- 使用 HTTPS
- 配置防火墙规则
- 限制数据库访问
- 定期更新密钥和证书

### 4. 配置文件管理

**配置优先级**：
1. 环境变量（最高优先级）
2. 配置文件
3. 默认值（最低优先级）

**配置加载**：
```python
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 配置类
class Config:
    DEBUG = os.getenv("DEBUG", "False") == "True"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    BASE_URL = os.getenv("BASE_URL", "https://example.com")
```

## 发布和部署规范

### 1. 发布前检查清单

**代码检查**：
- [ ] 所有代码审查通过
- [ ] 所有测试用例通过
- [ ] 代码覆盖率达标
- [ ] 无已知的安全漏洞
- [ ] 无性能问题

**文档检查**：
- [ ] API 文档已更新
- [ ] 用户文档已更新
- [ ] 变更日志已记录
- [ ] 配置说明已更新

**环境检查**：
- [ ] 测试环境验证通过
- [ ] 配置文件已准备
- [ ] 数据库迁移已准备
- [ ] 回滚方案已准备

### 2. 版本号规范

**语义化版本**：
格式：`主版本号.次版本号.修订号`

- 主版本号：不兼容的 API 修改
- 次版本号：向下兼容的功能性新增
- 修订号：向下兼容的问题修正

**预发布版本**：
格式：`主版本号.次版本号.修订号-预发布标识`

- `alpha`：内部测试版
- `beta`：公测版
- `rc`：发布候选版

**示例**：
```
1.0.0 - 首个稳定版
1.1.0 - 新增功能
1.1.1 - 修复 bug
2.0.0 - 重大更新
2.0.0-rc1 - 候选版
```

### 3. 回滚策略

**回滚触发条件**：
- 严重 bug 影响核心功能
- 性能严重下降
- 数据损坏或丢失
- 安全漏洞

**回滚流程**：
1. 立即停止新版本服务
2. 恢复到上一个稳定版本
3. 恢复数据库备份（如需要）
4. 验证系统功能正常
5. 分析问题原因
6. 修复后重新发布

**回滚准备**：
- 每次发布前准备回滚方案
- 保留最近 3 个版本的备份
- 定期测试回滚流程
- 记录回滚操作日志

### 4. 监控和告警

**监控指标**：
- 系统资源使用（CPU、内存、磁盘）
- 应用性能（响应时间、吞吐量）
- 业务指标（成功率、错误率）
- 自定义业务指标

**告警规则**：
- 错误率超过阈值
- 响应时间超过阈值
- 系统资源使用超过阈值
- 关键业务指标异常

**告警通知**：
- 即时通知：严重问题
- 定期汇总：一般问题
- 通知渠道：邮件、短信、即时通讯工具

## 团队协作规范

### 代码所有权

**模块负责人**：
- 每个主要模块指定负责人
- 负责人负责模块的设计和维护
- 重大变更需要与负责人协商
- 负责人审查相关 PR

**协作原则**：
- 遵循代码审查流程
- 尊重他人的代码
- 重大变更提前沟通
- 及时响应审查意见

### 知识分享

**技术分享会**：
- 定期组织技术分享会
- 分享新技术和最佳实践
- 记录分享内容
- 鼓励团队成员参与

**文档化**：
- 重要决策必须文档化
- 记录架构演进过程
- 分享最佳实践
- 维护知识库

### 问题解决

**问题跟踪**：
- 使用 issue 跟踪问题
- 清晰描述问题
- 提供复现步骤
- 标记优先级和严重程度

**响应时间**：
- 严重问题：24 小时内响应
- 一般问题：48 小时内响应
- 低优先级：一周内响应

**解决方案**：
- 记录解决方案
- 更新相关文档
- 防止问题再次发生
- 分享解决经验

### 沟通规范

**会议规范**：
- 会议前发送议程
- 会议中做好记录
- 会议后发送纪要
- 明确行动项和负责人

**即时通讯**：
- 工作时间及时响应
- 重要事项使用邮件确认
- 避免打扰他人
- 尊重他人时间

**文档沟通**：
- 重要决策文档化
- 使用统一的文档格式
- 定期更新文档
- 保持文档准确性

## 版本控制规范

### 1. 分支管理策略

**分支命名规范**：
- 主分支：`main` 或 `master`
- 开发分支：`feature/功能描述`
- 修复分支：`fix/问题描述`
- 发布分支：`release/版本号`
- 热修复分支：`hotfix/问题描述`

**分支使用规则**：
- `main/master`：只包含稳定的、可发布的代码
- `feature/*`：用于开发新功能，从 `main/master` 分出
- `fix/*`：用于修复 bug，从 `main/master` 分出
- `release/*`：用于准备发布，从 `main/master` 分出
- `hotfix/*`：用于紧急修复生产问题，从 `main/master` 分出

**分支工作流程**：
1. 从 `main/master` 创建功能分支
2. 在功能分支上开发和测试
3. 提交代码并创建 Pull Request
4. 代码审查通过后合并到 `main/master`
5. 删除已合并的功能分支

### 2. 提交信息规范

**提交信息格式**：
```
<类型>(<范围>): <简短描述>

<详细描述>

<关联 issue>
```

**类型（type）**：
- `feat`：新功能
- `fix`：bug 修复
- `docs`：文档更新
- `style`：代码格式调整（不影响功能）
- `refactor`：重构（既不是新功能也不是修复）
- `perf`：性能优化
- `test`：测试相关
- `chore`：构建过程或辅助工具的变动

**示例**：
```
feat(profile): 添加充值按钮功能

- 添加充值按钮定位器
- 实现充值按钮点击方法
- 添加充值按钮测试用例

Closes #123
```

**提交原则**：
- 每次提交只做一件事
- 提交信息清晰描述变更内容
- 提交前进行代码检查
- 提交前确保测试通过

### 3. 合并请求（PR）规范

**PR 标题**：
- 使用与提交信息相同的格式
- 简洁明了地描述变更内容

**PR 描述模板**：
```markdown
## 变更类型
- [ ] 新功能
- [ ] Bug 修复
- [ ] 重构
- [ ] 文档更新
- [ ] 性能优化
- [ ] 其他：

## 变更说明
描述本次 PR 的主要变更内容

## 测试情况
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试通过
- [ ] 添加了新的测试用例

## 检查清单
- [ ] 代码符合项目规范
- [ ] 添加了必要的文档
- [ ] 更新了相关文档
- [ ] 无不必要的代码改动
- [ ] 无调试代码和注释

## 关联 Issue
Closes #<issue 号>
```

**PR 审查要点**：
- 代码质量和可读性
- 是否符合项目规范
- 是否有重复代码
- 测试覆盖是否充分
- 文档是否完整

### 4. 代码审查流程

**审查要求**：
- 每个 PR 必须至少经过 1 人审查
- 审查者必须熟悉相关代码
- 重要变更需要多人审查

**审查内容**：
- 功能实现是否符合需求
- 代码质量和可维护性
- 是否遵循项目规范
- 测试是否充分
- 是否有潜在的安全问题

**审查反馈**：
- 提供具体、建设性的反馈
- 指出问题的同时给出建议
- 对审查意见及时响应

## 完整检查清单

### 开发前
- [ ] 需求已理解
- [ ] 技术方案已设计
- [ ] 影响范围已评估
- [ ] 依赖关系已分析
- [ ] 开发环境已准备

### 开发中
- [ ] 使用现有方法
- [ ] 遵循命名规范
- [ ] 添加日志记录
- [ ] 处理异常情况
- [ ] 编写测试用例
- [ ] 添加文档字符串
- [ ] 控制代码复杂度

### 开发后
- [ ] 代码审查通过
- [ ] 测试用例通过
- [ ] 文档已更新
- [ ] 无安全漏洞
- [ ] 无性能问题
- [ ] 代码覆盖率达标

### 发布前
- [ ] 所有检查项通过
- [ ] 回滚方案已准备
- [ ] 监控已配置
- [ ] 团队已通知
- [ ] 配置文件已准备
- [ ] 数据库迁移已准备

### 代码质量
- [ ] 命名符合规范
- [ ] 结构清晰合理
- [ ] 无重复代码
- [ ] 复杂度在合理范围
- [ ] 注释充分且准确
- [ ] 文档字符串完整
- [ ] 代码易于理解
- [ ] 模块职责单一

### 测试覆盖
- [ ] 单元测试充分
- [ ] 集成测试完整
- [ ] 边界条件测试
- [ ] 异常路径测试
- [ ] 测试数据已准备
- [ ] 测试报告已生成

### 安全性
- [ ] 无硬编码敏感信息
- [ ] 所有输入都经过验证
- [ ] 所有输出都经过编码
- [ ] 使用了安全的认证机制
- [ ] 实现了适当的权限控制
- [ ] 日志中不包含敏感信息
- [ ] 依赖库无已知安全漏洞
- [ ] 使用了 HTTPS 等安全协议

### 性能
- [ ] 无不必要的循环和递归
- [ ] 使用高效的数据结构和算法
- [ ] 合理使用缓存机制
- [ ] 避免频繁的数据库查询
- [ ] 及时释放不再使用的资源
- [ ] 避免内存泄漏
- [ ] 控制并发线程/进程数量
- [ ] 优化大文件处理

### 文档
- [ ] API 文档已更新
- [ ] 用户文档已更新
- [ ] 开发文档已更新
- [ ] 测试文档已更新
- [ ] 变更日志已记录
- [ ] 配置说明已更新

### 版本控制
- [ ] 分支命名符合规范
- [ ] 提交信息清晰描述变更
- [ ] 提交前代码检查通过
- [ ] 提交前测试通过
- [ ] PR 描述完整
- [ ] 代码审查通过

## 常见错误

### 1. 手动等待元素

```python
# ❌ 错误做法
element = self.page.ele(locator)
element.wait.clickable(timeout=5)

# ✅ 正确做法
element = self.page_actions.find_element(locator, selector_type)
# 装饰器会自动处理等待
```

### 2. 直接使用 page 对象

```python
# ❌ 错误做法
element = self.page.ele(locator)

# ✅ 正确做法
element = self.page_actions.find_element(locator, selector_type)
```

### 3. 不使用日志

```python
# ❌ 错误做法
def get_xxx(self):
    element = self.page_actions.find_element(locator, selector_type)
    return element

# ✅ 正确做法
def get_xxx(self):
    """获取XXX元素"""
    self.logger.info("获取XXX")
    element = self.page_actions.find_element(locator, selector_type)
    if not element:
        raise Exception(f"Element not found for locator: {locator}")
    return element
```
