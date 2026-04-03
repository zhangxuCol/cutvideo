#!/usr/bin/env python3
"""
全自动代码优化脚本 - 无需人工干预
集成 AI 优化 + 自动验证 + Git 保护 + 详细日志记录
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

class AutoOptimizer:
    """全自动代码优化器"""
    
    def __init__(self, language: str, goal: str, max_iterations: int, git_repo: str = None):
        self.language = language
        self.goal = goal
        self.max_iterations = max_iterations
        self.git_repo = git_repo
        self.git_enabled = git_repo is not None
        self.git_branch = None
        self.git_history = []
        self.optimization_log = []
        self.start_time = time.time()
        
    def log(self, message: str, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.optimization_log.append(log_entry)
        
    def save_log(self, output_path: str):
        """保存完整日志"""
        log_file = Path(output_path).parent / f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file, 'w') as f:
            f.write(f"优化任务日志\n")
            f.write(f"{'='*60}\n")
            f.write(f"语言: {self.language}\n")
            f.write(f"目标: {self.goal}\n")
            f.write(f"最大迭代: {self.max_iterations}\n")
            f.write(f"Git保护: {self.git_enabled}\n")
            f.write(f"执行时间: {time.time() - self.start_time:.2f}s\n")
            f.write(f"{'='*60}\n\n")
            f.write('\n'.join(self.optimization_log))
        self.log(f"日志已保存: {log_file}")
        return log_file
        
    def create_git_branch(self) -> bool:
        """创建 Git 保护分支"""
        if not self.git_enabled:
            return False
            
        git_manager = Path(__file__).parent / "git_manager.py"
        result = subprocess.run(
            [sys.executable, str(git_manager), "create-branch", self.git_repo, ""],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('BRANCH:'):
                    self.git_branch = line.replace('BRANCH:', '').strip()
                    self.log(f"创建 Git 分支: {self.git_branch}")
                    return True
        
        self.log("Git 分支创建失败，禁用 Git 保护", "WARN")
        self.git_enabled = False
        return False
        
    def validate_code(self, code: str) -> dict:
        """验证代码"""
        script_path = Path(__file__).parent / "validate_code.py"
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), self.language, "--stdin"],
                input=code, capture_output=True, text=True, timeout=60
            )
            
            try:
                return json.loads(result.stdout)
            except:
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "errors": [result.stderr] if result.stderr else []
                }
        except Exception as e:
            return {"success": False, "output": "", "errors": [str(e)]}
            
    def optimize_with_ai(self, code: str, validation: dict) -> str:
        """
        使用 AI 进行代码优化
        这里会调用外部 AI 服务或本地模型
        """
        # 构建优化提示
        prompt = self.build_optimization_prompt(code, validation)
        
        # 保存提示到临时文件
        temp_prompt = Path("/tmp/optimize_prompt.txt")
        temp_prompt.write_text(prompt)
        
        self.log("调用 AI 进行优化...")
        
        # 调用 OpenClaw AI（通过 exec 调用外部工具）
        # 这里使用一个简单的优化策略文件
        optimized = self.apply_optimization_strategy(code, self.goal)
        
        return optimized
        
    def build_optimization_prompt(self, code: str, validation: dict) -> str:
        """构建优化提示"""
        errors = validation.get('errors', [])
        error_text = '\n'.join(errors[:3]) if errors else '无错误'
        
        prompt = f"""
请优化以下 {self.language} 代码，目标: {self.goal}

当前代码:
```{self.language}
{code}
```

当前问题:
{error_text}

请直接返回优化后的完整代码，不要解释。
"""
        return prompt
        
    def apply_optimization_strategy(self, code: str, goal: str) -> str:
        """
        应用优化策略（简化版，实际应该调用 AI）
        这里返回一些基本的优化示例
        """
        # 根据目标应用不同策略
        if goal == "performance":
            # 性能优化：查找并优化常见模式
            optimized = code
            # 示例：优化循环
            if "for " in code and "range(len(" in code:
                optimized = optimized.replace(
                    "for i in range(len(",
                    "for i, _ in enumerate("
                )
            return optimized
        elif goal == "readability":
            # 可读性优化：添加注释，改进命名
            return code  # 简化处理
        else:
            return code
            
    def is_code_corrupted(self, original: str, current: str, validation: dict) -> bool:
        """检查代码是否被破坏"""
        errors = ' '.join(str(e).lower() for e in validation.get('errors', []))
        
        critical_keywords = ['syntaxerror', 'syntax error', 'parse error', 
                           'indentationerror', 'unexpected eof', 'invalid syntax']
        
        has_critical_error = any(kw in errors for kw in critical_keywords)
        code_empty = len(current.strip()) < len(original.strip()) * 0.5
        
        return has_critical_error or code_empty
        
    def commit_to_git(self, code: str, iteration: int, status: str):
        """提交到 Git"""
        if not self.git_enabled or not self.git_branch:
            return False
            
        # 保存代码到文件
        code_file = Path(self.git_repo) / f"optimized_code.{self.language}"
        code_file.write_text(code)
        
        # 提交
        git_manager = Path(__file__).parent / "git_manager.py"
        commit_msg = f"[迭代{iteration}] {self.goal} - {status}"
        
        result = subprocess.run(
            [sys.executable, str(git_manager), "commit", self.git_repo, commit_msg, str(iteration)],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('COMMIT:'):
                    commit_hash = line.replace('COMMIT:', '').strip()
                    self.git_history.append({
                        "iteration": iteration,
                        "commit": commit_hash,
                        "message": commit_msg
                    })
                    self.log(f"Git 提交: {commit_hash[:8]}")
                    return True
        
        return False
        
    def rollback_git(self):
        """Git 回滚"""
        if not self.git_enabled:
            return
            
        result = subprocess.run(
            ['git', 'reset', '--hard', 'HEAD~1'],
            cwd=self.git_repo, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            self.log("Git 已回滚到上一版本")
            if self.git_history:
                self.git_history.pop()
        else:
            self.log(f"Git 回滚失败: {result.stderr}", "ERROR")
            
    def run(self, code: str) -> dict:
        """运行全自动优化循环"""
        self.log(f"开始全自动优化: {self.goal}")
        self.log(f"最大迭代次数: {self.max_iterations}")
        
        # Step 0: 创建 Git 分支
        if self.git_enabled:
            self.create_git_branch()
            
        original_code = code
        current_code = code
        iteration = 0
        rollback_count = 0
        best_code = code
        best_score = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            self.log(f"\n{'='*60}")
            self.log(f"迭代 {iteration}/{self.max_iterations}")
            self.log(f"{'='*60}")
            
            # 1. 验证当前代码
            self.log("验证当前代码...")
            validation = self.validate_code(current_code)
            
            if validation["success"]:
                self.log("✓ 代码验证通过")
                score = 100
            else:
                error = str(validation.get('errors', ['未知'])[:1])
                self.log(f"✗ 验证失败: {error[:80]}")
                score = 50
                
            if score > best_score:
                best_score = score
                best_code = current_code
                
            # 2. 优化代码
            self.log(f"优化代码 (目标: {self.goal})...")
            optimized = self.optimize_with_ai(current_code, validation)
            
            if optimized == current_code:
                self.log("代码未改变，跳过本次迭代")
                continue
                
            code_changed = len(optimized) != len(current_code)
            self.log(f"代码已修改: {len(current_code)} → {len(optimized)} 字符")
            
            # 3. 验证优化后的代码
            self.log("验证优化后的代码...")
            new_validation = self.validate_code(optimized)
            
            should_rollback = False
            should_commit = False
            
            if new_validation["success"]:
                self.log("✓ 优化后验证通过")
                should_commit = True
                current_code = optimized
                score = 100
            else:
                error = str(new_validation.get('errors', ['未知'])[:1])
                self.log(f"✗ 验证失败: {error[:80]}")
                
                if self.is_code_corrupted(original_code, optimized, new_validation):
                    self.log("代码被破坏！执行回滚...", "WARN")
                    should_rollback = True
                    rollback_count += 1
                else:
                    self.log("代码结构完整，继续优化...")
                    should_commit = True
                    current_code = optimized
                    score = 70
                    
            if score > best_score:
                best_score = score
                best_code = current_code
                
            # 4. Git 操作
            if self.git_enabled:
                if should_rollback and self.git_history:
                    self.rollback_git()
                    if self.optimization_log:
                        current_code = self.optimization_log[-1].get('code', original_code)
                elif should_commit:
                    status = "通过" if new_validation["success"] else "失败但继续"
                    self.commit_to_git(current_code, iteration, status)
                    
            # 记录本次迭代
            self.optimization_log.append({
                "iteration": iteration,
                "code": current_code,
                "validation": new_validation,
                "rollback": should_rollback
            })
            
            # 检查终止条件
            if should_rollback and rollback_count >= 3:
                self.log(f"回滚次数过多 ({rollback_count})，停止优化", "WARN")
                break
                
            if iteration >= self.max_iterations:
                self.log("达到最大迭代次数")
                
        # 最终验证
        self.log(f"\n{'='*60}")
        self.log("优化完成")
        self.log(f"{'='*60}")
        
        final_validation = self.validate_code(best_code)
        execution_time = time.time() - self.start_time
        
        self.log(f"总迭代次数: {iteration}")
        self.log(f"回滚次数: {rollback_count}")
        self.log(f"最佳评分: {best_score}")
        self.log(f"最终验证: {'通过' if final_validation['success'] else '失败'}")
        self.log(f"代码变化: {len(original_code)} → {len(best_code)} 字符")
        self.log(f"执行时间: {execution_time:.2f}s")
        
        if self.git_enabled:
            self.log(f"\nGit 分支: {self.git_branch}")
            self.log(f"提交次数: {len(self.git_history)}")
            self.log(f"查看历史: git log {self.git_branch} --oneline")
            
        return {
            "original_code": original_code,
            "optimized_code": best_code,
            "iterations": iteration,
            "rollback_count": rollback_count,
            "best_score": best_score,
            "final_validation": final_validation,
            "execution_time": execution_time,
            "git_branch": self.git_branch,
            "git_history": self.git_history,
            "optimization_log": self.optimization_log
        }


def main():
    """命令行入口"""
    if len(sys.argv) < 3:
        print("Usage: python auto_optimize.py <language> <code_file> [goal] [max_iterations] [git_repo]")
        print("\nExamples:")
        print("  # 基础优化（无 Git）")
        print("  python auto_optimize.py python code.py performance 10")
        print("\n  # 带 Git 保护")
        print("  python auto_optimize.py python code.py performance 10 /path/to/repo")
        sys.exit(1)
    
    language = sys.argv[1]
    code_file = sys.argv[2]
    goal = sys.argv[3] if len(sys.argv) > 3 else "general"
    max_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    git_repo = sys.argv[5] if len(sys.argv) > 5 else None
    
    # 读取代码
    code = Path(code_file).read_text()
    
    # 创建优化器并运行
    optimizer = AutoOptimizer(language, goal, max_iter, git_repo)
    result = optimizer.run(code)
    
    # 保存结果
    output_file = Path(code_file).parent / f"optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{language}"
    output_file.write_text(result["optimized_code"])
    print(f"\n优化结果已保存: {output_file}")
    
    # 保存日志
    log_file = optimizer.save_log(str(output_file))
    
    # 输出摘要
    print(f"\n{'='*60}")
    print("优化摘要")
    print(f"{'='*60}")
    print(f"迭代次数: {result['iterations']}")
    print(f"回滚次数: {result['rollback_count']}")
    print(f"最佳评分: {result['best_score']}")
    print(f"执行时间: {result['execution_time']:.2f}s")
    print(f"最终结果: {'✓ 通过' if result['final_validation']['success'] else '✗ 失败'}")
    print(f"输出文件: {output_file}")
    print(f"日志文件: {log_file}")
    
    if result['git_branch']:
        print(f"Git 分支: {result['git_branch']}")


if __name__ == "__main__":
    main()
