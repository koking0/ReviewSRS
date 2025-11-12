"""
用户故事歧义检测通用工具模块

该模块提供了用于各种歧义类型检测的通用函数和类，
包括API调用、评估指标计算、数据可视化等功能。
"""

import pandas as pd
import numpy as np
import json
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 从 prompts 模块导入提示词配置
from prompts import (
    AMBIGUITY_TYPE_CONFIGS,
    generate_ambiguity_prompt,
    get_ambiguity_config,
    get_recommended_sample_size,
    get_all_ambiguity_types
)

# 配置信息
CONFIG = {
    "base_url": "https://api.zhizengzeng.com/v1/",
    "api_key": "sk-zk20f741becece1c055c848225093b2e458662329a0f1016"
}

# 模型列表
DEFAULT_MODELS = [
    "gpt-3.5-turbo",
    # "gemini-2.5-flash",
    "deepseek-chat"
]

class AmbiguityDetector:
    """
    歧义检测基类，提供通用的检测功能
    """

    def __init__(self, ambiguity_type: str, models: List[str] = None):
        """
        初始化歧义检测器

        Args:
            ambiguity_type: 歧义类型（如'semantic', 'scope', 'actor'等）
            models: 要使用的模型列表

        Raises:
            ValueError: 如果ambiguity_type不被支持
        """
        if ambiguity_type not in AMBIGUITY_TYPE_CONFIGS:
            raise ValueError(f"Unknown ambiguity type: {ambiguity_type}. Supported types: {list(AMBIGUITY_TYPE_CONFIGS.keys())}")

        self.ambiguity_type = ambiguity_type
        self.ambiguity_config = AMBIGUITY_TYPE_CONFIGS[ambiguity_type]
        self.models = models or DEFAULT_MODELS
        self.results = []

    def get_ambiguity_config(self) -> Dict:
        """
        获取当前歧义类型的配置

        Returns:
            歧义类型的配置字典
        """
        return get_ambiguity_config(self.ambiguity_type)

    def get_recommended_sample_size(self) -> int:
        """
        获取当前歧义类型的推荐样本大小

        Returns:
            推荐的样本大小
        """
        return get_recommended_sample_size(self.ambiguity_type)

    def generate_prompt(self, user_story: str) -> str:
        """
        为用户故事生成对应歧义类型的提示词

        Args:
            user_story: 用户故事文本

        Returns:
            生成的提示词
        """
        return generate_ambiguity_prompt(user_story, self.ambiguity_type)

    def call_llm(self, prompt: str, model: str) -> Dict:
        """
        调用大模型API的通用函数

        Args:
            prompt: 提示词
            model: 模型名称

        Returns:
            解析后的JSON响应
        """
        try:
            client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            content = response.choices[0].message.content

            # 尝试解析JSON格式的响应
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回默认值
                return self._get_default_response()

        except Exception as e:
            print(f"API call failed ({model}): {str(e)}")
            return self._get_default_response()

    def _get_default_response(self) -> Dict:
        """
        获取默认响应格式
        """
        has_ambiguity_key = f"Has {self.ambiguity_type.title()} Ambiguity"
        return {
            has_ambiguity_key: "true",
            "Ambiguous Parts": "unknown",
            "Reasoning": "parsing/api error",
            "Suggested Improvement": "unknown"
        }

    def evaluate_detection(self, y_true: List[bool], y_pred: List[str]) -> Tuple[float, float, float]:
        """
        计算检测任务的评估指标

        Args:
            y_true: 真实标签列表
            y_pred: 预测标签列表

        Returns:
            (precision, recall, f1)
        """
        true_positives = false_positives = false_negatives = 0

        for true_label, pred_label in zip(y_true, y_pred):
            pred_bool = pred_label.lower() == "true"

            if true_label and pred_bool:
                true_positives += 1
            elif not true_label and pred_bool:
                false_positives += 1
            elif true_label and not pred_bool:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def load_and_prepare_data(self, file_path: str, create_balanced: bool = True) -> pd.DataFrame:
        """
        加载并准备数据

        Args:
            file_path: 数据文件路径
            create_balanced: 是否创建平衡数据集

        Returns:
            准备好的数据集
        """
        try:
            df_full = pd.read_excel(file_path, sheet_name='User_Stories')

            # 选择相关的列
            column_name = f"{self.ambiguity_type.title()}Ambiguity"
            df = df_full[['StoryID', 'StoryText', column_name]].copy()
            df.columns = ['StoryID', 'StoryText', 'HasAmbiguity']

            # 统计信息
            ambiguity_count = df['HasAmbiguity'].sum()
            non_ambiguity_count = len(df) - ambiguity_count

            print(f"{self.ambiguity_type.title()}歧义统计:")
            print(f"  有{self.ambiguity_type}歧义: {ambiguity_count} ({ambiguity_count/len(df)*100:.1f}%)")
            print(f"  无{self.ambiguity_type}歧义: {non_ambiguity_count} ({non_ambiguity_count/len(df)*100:.1f}%)")

            if create_balanced and ambiguity_count > 0 and non_ambiguity_count > 0:
                # 创建平衡数据集
                min_count = min(ambiguity_count, non_ambiguity_count)
                positive = df[df['HasAmbiguity'] == True].sample(n=min_count, random_state=42)
                negative = df[df['HasAmbiguity'] == False].sample(n=min_count, random_state=42)
                df_balanced = pd.concat([positive, negative]).reset_index(drop=True)
                print(f"平衡测试集: {len(df_balanced)} 个用户故事")

                # 统计最终数据集中HasAmbiguity的分布
                final_true_count = (df_balanced['HasAmbiguity'] == True).sum()
                final_false_count = (df_balanced['HasAmbiguity'] == False).sum()
                print(f"  HasAmbiguity=True: {final_true_count} ({final_true_count/len(df_balanced)*100:.1f}%)")
                print(f"  HasAmbiguity=False: {final_false_count} ({final_false_count/len(df_balanced)*100:.1f}%)")

                # 打乱数据顺序
                df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

                return df_balanced
            else:
                print(f"使用所有可用数据: {len(df)} 个用户故事")

                # 统计最终数据集中HasAmbiguity的分布
                final_true_count = (df['HasAmbiguity'] == True).sum()
                final_false_count = (df['HasAmbiguity'] == False).sum()
                print(f"  HasAmbiguity=True: {final_true_count} ({final_true_count/len(df)*100:.1f}%)")
                print(f"  HasAmbiguity=False: {final_false_count} ({final_false_count/len(df)*100:.1f}%)")

                # 打乱数据顺序
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)

                return df

        except Exception as e:
            print(f"加载数据时出错: {e}")
            return pd.DataFrame()

    def process_dataset(self, df: pd.DataFrame, model: str, sample_size: int = 30,
                        prompt_generator: callable = None) -> Dict:
        """
        处理数据集并评估模型性能

        Args:
            df: 数据集
            model: 模型名称
            sample_size: 样本大小
            prompt_generator: 提示词生成函数

        Returns:
            包含预测结果和指标的结果字典
        """
        if prompt_generator is None:
            raise ValueError("必须提供prompt_generator函数")

        results = {
            "model": model,
            "ambiguity_type": self.ambiguity_type,
            "predictions": [],
            "metrics": {}
        }

        print(f"Processing model {model}...")
        df_sample = df.head(sample_size)

        for idx, row in df_sample.iterrows():
            # 生成提示词
            prompt = prompt_generator(row['StoryText'])

            # 调用模型
            prediction = self.call_llm(prompt, model)

            # 确保预测结果格式正确
            has_ambiguity_key = f"Has {self.ambiguity_type.title()} Ambiguity"
            if has_ambiguity_key not in prediction:
                prediction[has_ambiguity_key] = "true"

            results["predictions"].append({
                "story_id": row['StoryID'],
                "story_text": row['StoryText'],
                "true_has_ambiguity": row['HasAmbiguity'],
                "pred_has_ambiguity": prediction[has_ambiguity_key].lower(),
                "ambiguous_parts": prediction.get("Ambiguous Parts", "unknown"),
                "reasoning": prediction.get("Reasoning", "unknown")
            })

            # 进度显示
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(df_sample)} samples")

        # 计算指标
        true_labels = [item["true_has_ambiguity"] for item in results["predictions"]]
        pred_labels = [item["pred_has_ambiguity"] for item in results["predictions"]]

        precision, recall, f1 = self.evaluate_detection(true_labels, pred_labels)

        results["metrics"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        print(f"  {self.ambiguity_type.title()} Ambiguity Detection - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        return results

    def run_evaluation(self, df: pd.DataFrame, sample_size: int = 30,
                      prompt_generator: callable = None) -> List[Dict]:
        """
        运行所有模型的评估

        Args:
            df: 数据集
            sample_size: 样本大小
            prompt_generator: 提示词生成函数

        Returns:
            所有模型的评估结果列表
        """
        print(f"=== {self.ambiguity_type.title()}歧义检测评估 ===")
        print(f"测试数据集: {len(df)} 个用户故事")

        all_results = []

        for model in self.models:
            result = self.process_dataset(df, model, sample_size, prompt_generator)
            all_results.append(result)

            # 添加延迟以避免API限制
            time.sleep(1)

        return all_results

    def create_visualization(self, results: List[Dict], save_prefix: str = None):
        """
        创建结果可视化

        Args:
            results: 评估结果列表
            save_prefix: 保存文件的前缀
        """
        if not results:
            print("没有结果可以可视化")
            return

        models = [r["model"] for r in results]
        precisions = [r["metrics"]["precision"] for r in results]
        recalls = [r["metrics"]["recall"] for r in results]
        f1_scores = [r["metrics"]["f1"] for r in results]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{self.ambiguity_type.title()} Ambiguity Detection Analysis', fontsize=16, fontweight='bold')

        x = np.arange(len(models))
        width = 0.35

        # 精确率和召回率
        ax1.bar(x - width/2, precisions, width, label='Precision', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title(f'{self.ambiguity_type.title()} Ambiguity Detection - Precision & Recall')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 1)

        # 添加数值标签
        for i, (p, r) in enumerate(zip(precisions, recalls)):
            ax1.text(i - width/2, p + 0.01, f'{p:.2f}', ha='center', va='bottom')
            ax1.text(i + width/2, r + 0.01, f'{r:.2f}', ha='center', va='bottom')

        # F1分数
        bars = ax2.bar(models, f1_scores, alpha=0.8, color='gold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('F1 Score')
        ax2.set_title(f'{self.ambiguity_type.title()} Ambiguity Detection - F1 Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 1)

        # 添加数值标签
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.2f}', ha='center', va='bottom')

        plt.tight_layout()

        # 保存图表
        if save_prefix:
            filename = f'{save_prefix}_{self.ambiguity_type}_ambiguity_results.png'
        else:
            filename = f'{self.ambiguity_type}_ambiguity_detection_results.png'

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, results: List[Dict], filename: str = None):
        """
        保存评估结果

        Args:
            results: 评估结果列表
            filename: 保存文件名
        """
        if filename is None:
            filename = f'{self.ambiguity_type}_ambiguity_evaluation_results.json'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"结果已保存到 {filename}")

    def print_summary(self, results: List[Dict]):
        """
        打印评估总结

        Args:
            results: 评估结果列表
        """
        print(f"\n=== {self.ambiguity_type.title()}歧义检测评估总结 ===")
        for result in results:
            model = result["model"]
            metrics = result["metrics"]
            print(f"\nModel: {model}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1']:.3f}")


def create_comparison_visualization(all_results: Dict[str, List[Dict]], save_filename: str = "ambiguity_comparison.png"):
    """
    创建所有歧义类型的比较可视化

    Args:
        all_results: 所有歧义类型的结果字典 {ambiguity_type: results}
        save_filename: 保存文件名
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('All Ambiguity Types Detection Performance Comparison', fontsize=16, fontweight='bold')

    # 准备数据
    ambiguity_types = list(all_results.keys())
    models = DEFAULT_MODELS[:3]  # 假设使用前3个模型

    # 1. 精确率比较
    precision_data = []
    for amb_type in ambiguity_types:
        results = all_results[amb_type]
        precision_data.append([r["metrics"]["precision"] for r in results])

    precision_df = pd.DataFrame(precision_data, index=ambiguity_types, columns=models)
    precision_df.plot(kind='bar', ax=axes[0, 0], alpha=0.8)
    axes[0, 0].set_title('Precision Comparison Across Ambiguity Types')
    axes[0, 0].set_ylabel('Precision Score')
    axes[0, 0].legend(title='Model')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. 召回率比较
    recall_data = []
    for amb_type in ambiguity_types:
        results = all_results[amb_type]
        recall_data.append([r["metrics"]["recall"] for r in results])

    recall_df = pd.DataFrame(recall_data, index=ambiguity_types, columns=models)
    recall_df.plot(kind='bar', ax=axes[0, 1], alpha=0.8, color='orange')
    axes[0, 1].set_title('Recall Comparison Across Ambiguity Types')
    axes[0, 1].set_ylabel('Recall Score')
    axes[0, 1].legend(title='Model')
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. F1分数比较
    f1_data = []
    for amb_type in ambiguity_types:
        results = all_results[amb_type]
        f1_data.append([r["metrics"]["f1"] for r in results])

    f1_df = pd.DataFrame(f1_data, index=ambiguity_types, columns=models)
    f1_df.plot(kind='bar', ax=axes[1, 0], alpha=0.8, color='green')
    axes[1, 0].set_title('F1 Score Comparison Across Ambiguity Types')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend(title='Model')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 4. 热力图 - 最佳F1分数
    best_f1_scores = []
    for amb_type in ambiguity_types:
        results = all_results[amb_type]
        best_f1 = max([r["metrics"]["f1"] for r in results])
        best_f1_scores.append(best_f1)

    # 创建热力图数据
    heatmap_data = f1_df.T

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                ax=axes[1, 1], cbar_kws={'label': 'F1 Score'})
    axes[1, 1].set_title('F1 Score Heatmap (Models vs Ambiguity Types)')
    axes[1, 1].set_xlabel('Ambiguity Type')
    axes[1, 1].set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_errors(results: List[Dict], ambiguity_type: str, num_examples: int = 3):
    """
    分析错误案例

    Args:
        results: 评估结果
        ambiguity_type: 歧义类型
        num_examples: 显示的错误案例数量
    """
    print(f"\n=== {ambiguity_type.title()}歧义错误分析 ===")

    # 选择最佳模型的结果进行分析
    best_model = max(results, key=lambda x: x["metrics"]["f1"])
    model_name = best_model["model"]
    predictions = best_model["predictions"]

    print(f"\n分析模型: {model_name} (F1: {best_model['metrics']['f1']:.3f})")

    # 找出错误案例
    false_positives = []
    false_negatives = []

    for pred in predictions:
        if pred["true_has_ambiguity"] == False and pred["pred_has_ambiguity"] == "true":
            false_positives.append(pred)
        elif pred["true_has_ambiguity"] == True and pred["pred_has_ambiguity"] == "false":
            false_negatives.append(pred)

    print(f"\n误报 (False Positives): {len(false_positives)} 个")
    print(f"漏报 (False Negatives): {len(false_negatives)} 个")

    # 显示示例
    print(f"\n误报示例 (模型认为有歧义，实际无歧义):")
    for i, fp in enumerate(false_positives[:num_examples]):
        print(f"\n{i+1}. 用户故事: {fp['story_text']}")
        print(f"   模型识别的歧义部分: {fp['ambiguous_parts']}")
        print(f"   模型推理: {fp['reasoning']}")

    print(f"\n漏报示例 (实际有歧义，模型未检测到):")
    for i, fn in enumerate(false_negatives[:num_examples]):
        print(f"\n{i+1}. 用户故事: {fn['story_text']}")
        print(f"   模型推理: {fn['reasoning']}")