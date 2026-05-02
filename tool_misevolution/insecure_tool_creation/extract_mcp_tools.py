"""Extract tool definitions from trajectory files.

Usage:
  cd tool_misevolution/insecure_tool_creation
  python extract_mcp_tools.py

Input/output defaults:
  results/<dataset>/evolution/ours/trajectories
  results/<dataset>/evolution/ours/mcp_tools.jsonl
"""

import os
import json
import re
import hashlib
import ast
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DATASET_NAME = os.getenv("EXPERIMENT_DATASET", "misevolution_user_queries")
OUTPUT_ROOT = Path(os.getenv("EXPERIMENT_OUTPUT_ROOT", str(DEFAULT_RESULTS_ROOT)))
OURS_VARIANT = os.getenv("OURS_VARIANT", "ours")
OURS_DIR = OUTPUT_ROOT / DATASET_NAME / "evolution" / OURS_VARIANT

class MCPToolExtractor:
    def __init__(self):
        self.trajs_dir = str(OURS_DIR / "trajectories")
        self.output_file = str(OURS_DIR / "mcp_tools.jsonl")
        # Strict by default: do not fabricate tools; only keep syntax-valid code.
        self.enable_fallback = os.getenv("EXTRACT_ENABLE_FALLBACK", "0") == "1"
        self.require_syntax_valid = os.getenv("EXTRACT_REQUIRE_SYNTAX", "1") == "1"
        self.stats = {
            "raw_candidates": 0,
            "invalid_syntax_dropped": 0,
            "fallback_added": 0,
        }
        
    def extract_mcp_from_code_block(self, code_block: str) -> Dict[str, Any]:
        """从代码块中提取MCP工具信息"""
        mcp_info = {}
        
        # 查找MCP注释行
        mcp_name_match = re.search(r'# MCP Name:\s*(.+)', code_block)
        tool_name_match = re.search(r'# Tool Name:\s*(.+)', code_block)
        mcp_desc_match = re.search(r'# Description:\s*(.+)', code_block)
        mcp_args_match = re.search(r'# Arguments:\s*(.+)', code_block)
        mcp_returns_match = re.search(r'# Returns:\s*(.+)', code_block)
        mcp_requires_match = re.search(r'# Requires:\s*(.+)', code_block)
        
        if mcp_name_match:
            mcp_info['name'] = mcp_name_match.group(1).strip()
        elif tool_name_match:
            # 兼容模型输出的 Tool Name 标记
            mcp_info['name'] = tool_name_match.group(1).strip()
        else:
            return None  # 不是MCP工具
            
        if mcp_desc_match:
            mcp_info['description'] = mcp_desc_match.group(1).strip()
        
        if mcp_args_match:
            mcp_info['arguments'] = mcp_args_match.group(1).strip()
        
        if mcp_returns_match:
            mcp_info['returns'] = mcp_returns_match.group(1).strip()
            
        if mcp_requires_match:
            mcp_info['requires'] = mcp_requires_match.group(1).strip()
        
        # 提取函数定义
        func_match = re.search(r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:', code_block)
        if func_match:
            mcp_info['function_name'] = func_match.group(1)
        
        # 提取完整的代码
        mcp_info['code'] = code_block.strip()
        
        return mcp_info

    def _normalize_code_block(self, code_block: str) -> str:
        """清理嵌套的 <python> 或 ```python 包裹，得到纯Python代码。"""
        code = (code_block or "").strip()
        if not code:
            return ""

        changed = True
        while changed:
            changed = False
            tag_match = re.match(r"^\s*<python>\s*(.*?)\s*</python>\s*$", code, flags=re.DOTALL | re.IGNORECASE)
            if tag_match:
                code = tag_match.group(1).strip()
                changed = True
                continue

            fence_match = re.match(r"^\s*```(?:python)?\s*(.*?)\s*```\s*$", code, flags=re.DOTALL | re.IGNORECASE)
            if fence_match:
                code = fence_match.group(1).strip()
                changed = True

        # 清理残留标签
        code = code.replace("<python>", "").replace("</python>", "").strip()
        return code

    def _is_syntax_valid(self, code: str) -> bool:
        if not code.strip():
            return False
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    def _sanitize_name(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
        return cleaned or "auto_tool"

    def _infer_tool_from_code_block(self, code_block: str) -> Dict[str, Any] | None:
        """在没有 Tool/MCP Name 注释时，基于函数定义推断工具信息。"""
        code = code_block.strip()
        func_match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?\s*:", code)
        if not func_match:
            return None

        function_name = func_match.group(1)
        arg_sig = func_match.group(2).strip() or "*args, **kwargs"
        imports = re.findall(r"^\s*import\s+([a-zA-Z0-9_., ]+)", code, re.MULTILINE)
        from_imports = re.findall(r"^\s*from\s+([a-zA-Z0-9_.]+)\s+import", code, re.MULTILINE)
        reqs = []
        for item in imports:
            reqs.extend([x.strip() for x in item.split(",") if x.strip()])
        reqs.extend(from_imports)
        requires = ", ".join(sorted(set(reqs))) if reqs else "None"

        return {
            "name": function_name,
            "description": "Auto-inferred tool from Python function definition.",
            "arguments": arg_sig,
            "returns": "Inferred from code body",
            "requires": requires,
            "function_name": function_name,
            "code": code,
            "inferred_tool": True,
        }

    def _make_fallback_tool(self, trajectory: Dict[str, Any], task_name: str) -> Dict[str, Any]:
        """当一个轨迹没有可解析工具时，创建占位工具记录以保证覆盖率。"""
        original_task = trajectory.get("original_task", {}) if isinstance(trajectory, dict) else {}
        fn_name = original_task.get("function_name") or f"{self._sanitize_name(task_name)}_fallback_tool"
        fn_name = self._sanitize_name(fn_name)
        task_desc = original_task.get("task_description", "")
        short_desc = (task_desc[:120] + "...") if len(task_desc) > 120 else task_desc
        code = (
            f"# Tool Name: {fn_name}\n"
            "# Description: Fallback tool record auto-generated because no parsable tool block was found.\n"
            "# Arguments: *args, **kwargs\n"
            "# Returns: Any\n"
            "# Requires: None\n"
            f"def {fn_name}(*args, **kwargs):\n"
            "    raise NotImplementedError('Fallback placeholder tool extracted from trajectory without tool block')\n"
        )
        return {
            "name": fn_name,
            "description": "Fallback placeholder generated by extractor.",
            "arguments": "*args, **kwargs",
            "returns": "Any",
            "requires": "None",
            "function_name": fn_name,
            "code": code,
            "source_task": task_name,
            "step_number": 0,
            "step_description": f"fallback_for_missing_tool_block: {short_desc}",
            "tool_index_in_step": 1,
            "extracted_at": datetime.now().isoformat(),
            "is_fallback": True,
        }

    def _dedupe_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按任务+函数+代码哈希去重，避免同一工具重复计数。"""
        seen = set()
        out = []
        for tool in tools:
            code = tool.get("code", "")
            code_hash = hashlib.sha1(code.encode("utf-8")).hexdigest() if code else ""
            key = (
                tool.get("source_task", ""),
                tool.get("function_name", ""),
                code_hash,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(tool)
        return out
    
    def extract_mcp_tools_from_trajectory(self, traj_file: str) -> List[Dict[str, Any]]:
        """从单个轨迹文件中提取所有MCP工具"""
        mcp_tools = []
        
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                trajectory = json.load(f)
            
            task_name = trajectory.get('task_name', os.path.basename(traj_file).replace('_trajectory.json', 'unknown'))
            workflow_execution = trajectory.get('workflow_execution', {})
            results = workflow_execution.get('results', [])
            
            for result in results:
                step_number = result.get('step_number', 0)
                step_description = result.get('step_description', '')
                extracted_context = result.get('extracted_context', '')
                raw_result = result.get('result', '')
                
                # 1) 从结构化 extracted_context 中提取
                mcp_pattern = r'### (?:MCP Tool|Python Code) \d+:\s*```python\s*(.*?)\s*```'
                context_matches = re.findall(mcp_pattern, extracted_context, re.DOTALL)

                # 2) 从原始 result 中提取，兼容 <python> 和 ```python
                raw_matches = self._extract_python_blocks(raw_result)

                all_blocks = []
                seen = set()
                for code in context_matches + raw_matches:
                    norm = code.strip()
                    if norm and norm not in seen:
                        seen.add(norm)
                        all_blocks.append(norm)

                for i, mcp_code in enumerate(all_blocks, 1):
                    normalized = self._normalize_code_block(mcp_code)
                    mcp_info = self.extract_mcp_from_code_block(normalized)
                    if not mcp_info:
                        mcp_info = self._infer_tool_from_code_block(normalized)
                    
                    if mcp_info:
                        self.stats["raw_candidates"] += 1
                        mcp_info["code"] = self._normalize_code_block(mcp_info.get("code", ""))
                        mcp_info["syntax_valid"] = self._is_syntax_valid(mcp_info["code"])
                        if self.require_syntax_valid and not mcp_info["syntax_valid"]:
                            self.stats["invalid_syntax_dropped"] += 1
                            continue

                        # 添加上下文信息
                        mcp_info['source_task'] = task_name
                        mcp_info['step_number'] = step_number
                        mcp_info['step_description'] = step_description
                        mcp_info['tool_index_in_step'] = i
                        mcp_info['extracted_at'] = datetime.now().isoformat()
                        
                        mcp_tools.append(mcp_info)
            
            mcp_tools = self._dedupe_tools(mcp_tools)
            if not mcp_tools and self.enable_fallback:
                mcp_tools = [self._make_fallback_tool(trajectory, task_name)]
                self.stats["fallback_added"] += 1
                        
        except Exception as e:
            print(f"❌ 处理轨迹文件 {traj_file} 时出错: {e}")
        
        return mcp_tools

    def _extract_python_blocks(self, text: str) -> List[str]:
        """从文本中提取 Python 代码块，兼容 <python> 和 ```python 两种格式。"""
        blocks = []
        if not text:
            return blocks

        tag_matches = re.findall(r'<python>(.*?)</python>', text, re.DOTALL | re.IGNORECASE)
        fence_matches = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        for code in tag_matches + fence_matches:
            code = code.strip()
            if code:
                blocks.append(code)
        return blocks
    
    def get_all_trajectory_files(self) -> List[str]:
        """获取所有轨迹文件"""
        traj_files = []
        
        if not os.path.exists(self.trajs_dir):
            print(f"❌ 轨迹目录不存在: {self.trajs_dir}")
            return traj_files
        
        for file in os.listdir(self.trajs_dir):
            if file.endswith('_trajectory.json'):
                traj_files.append(os.path.join(self.trajs_dir, file))
        
        return sorted(traj_files)
    
    def extract_all_mcp_tools(self) -> List[Dict[str, Any]]:
        """提取所有轨迹文件中的MCP工具"""
        all_mcp_tools = []
        traj_files = self.get_all_trajectory_files()
        
        print(f"🔍 发现 {len(traj_files)} 个轨迹文件")
        
        for i, traj_file in enumerate(traj_files, 1):
            print(f"📄 处理文件 {i}/{len(traj_files)}: {os.path.basename(traj_file)}")
            
            mcp_tools = self.extract_mcp_tools_from_trajectory(traj_file)
            all_mcp_tools.extend(mcp_tools)
            
            print(f"   找到 {len(mcp_tools)} 个MCP工具")
        
        return all_mcp_tools
    
    def save_mcp_tools_to_jsonl(self, mcp_tools: List[Dict[str, Any]]):
        """将MCP工具保存到JSONL文件"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for mcp_tool in mcp_tools:
                    json.dump(mcp_tool, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"✅ 成功保存 {len(mcp_tools)} 个MCP工具到: {self.output_file}")
            
        except Exception as e:
            print(f"❌ 保存MCP工具时出错: {e}")
    
    def generate_summary_report(self, mcp_tools: List[Dict[str, Any]]):
        """生成MCP工具摘要报告"""
        if not mcp_tools:
            print("📊 未找到任何MCP工具")
            return
        
        # 统计信息
        total_tools = len(mcp_tools)
        unique_names = set(tool['name'] for tool in mcp_tools if 'name' in tool)
        tasks_with_tools = set(tool['source_task'] for tool in mcp_tools)
        fallback_tools = [tool for tool in mcp_tools if tool.get("is_fallback")]
        
        # 按任务分组
        tools_by_task = {}
        for tool in mcp_tools:
            task = tool.get('source_task', 'unknown')
            if task not in tools_by_task:
                tools_by_task[task] = []
            tools_by_task[task].append(tool)
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("📊 MCP工具提取摘要")
        print(f"{'='*60}")
        print(f"📈 总MCP工具数: {total_tools}")
        print(f"🔧 唯一工具名数: {len(unique_names)}")
        print(f"📝 涉及任务数: {len(tasks_with_tools)}")
        print(f"🩹 Fallback工具数: {len(fallback_tools)}")
        print(f"🧪 语法过滤丢弃数: {self.stats['invalid_syntax_dropped']}")
        print(f"🧪 原始候选代码块数: {self.stats['raw_candidates']}")
        
        print(f"\n🔧 发现的MCP工具名称:")
        for i, name in enumerate(sorted(unique_names), 1):
            print(f"  {i}. {name}")
        
        print(f"\n📝 各任务的MCP工具数量:")
        for task, tools in sorted(tools_by_task.items()):
            print(f"  {task}: {len(tools)} 个工具")
        
        # 保存摘要报告
        summary = {
            "total_mcp_tools": total_tools,
            "unique_tool_names": len(unique_names),
            "tasks_with_tools": len(tasks_with_tools),
            "fallback_tools": len(fallback_tools),
            "invalid_syntax_dropped": self.stats["invalid_syntax_dropped"],
            "raw_candidates": self.stats["raw_candidates"],
            "tool_names": sorted(list(unique_names)),
            "tools_by_task": {task: len(tools) for task, tools in tools_by_task.items()},
            "generated_at": datetime.now().isoformat()
        }
        
        summary_file = self.output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"📁 摘要报告已保存: {summary_file}")
    
    def run(self):
        """运行MCP工具提取"""
        print("🔧 MCP工具提取器")
        print("="*60)
        
        # 提取所有MCP工具
        mcp_tools = self.extract_all_mcp_tools()
        
        if not mcp_tools:
            print("❌ 未找到任何MCP工具")
            return
        
        # 保存到JSONL文件
        self.save_mcp_tools_to_jsonl(mcp_tools)
        
        # 生成摘要报告
        self.generate_summary_report(mcp_tools)
        
        print(f"\n🎉 MCP工具提取完成!")

def main():
    """主函数"""
    extractor = MCPToolExtractor()
    extractor.run()

if __name__ == "__main__":
    main() 
