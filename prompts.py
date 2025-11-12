"""
提示词配置模块

该模块集中管理所有与歧义检测相关的提示词配置，包括：
1. 歧义类型检测的提示词模板
2. 消歧方法的提示词
3. 伪代码生成的提示词
"""

# ============================================================================
# 标准提示词模板（用于歧义检测）
# ============================================================================

PROMPT_TEMPLATE_BASE = """**Background:** {background_definition}

**Common examples of {ambiguity_type_lower} ambiguity include:**
{examples}

**Role:** You are a professional software requirements analyst specializing in detecting {ambiguity_type_lower} ambiguity in software requirements and user stories.

**Task:** Analyze the following user story and determine if it contains {ambiguity_type_lower} ambiguity.

User Story: "{user_story}"

Please analyze and provide:
{analysis_points}

Output in JSON format:
{{
    "Has {ambiguity_type} Ambiguity": "...",
    {output_fields}
}}

**Important Notes:**
{important_notes}"""


# ============================================================================
# 各种歧义类型的配置
# ============================================================================

AMBIGUITY_TYPE_CONFIGS = {
    "semantic": {
        "display_name": "Semantic Ambiguity",
        "background_definition": """Semantic ambiguity in user stories occurs when words, phrases, or entire sentences can be interpreted in multiple ways due to the multiple meanings of terms, unclear context, or vague expressions. This type of ambiguity is particularly problematic in software requirements as it can lead to different developers implementing different features based on their interpretation.""",

        "examples": """- Example: As a user, I want to manage my account so that I can handle my information.
  Reasoning: Vague verbs "manage" and "handle" - unclear what specific actions are needed.
  Suggested Improvement:  As a customer, I want to update my profile information so that I can keep my contact details current.""",

        "analysis_points": """1. **Has Semantic Ambiguity**: "true" if the story contains semantic ambiguity, "false" if it's clear and unambiguous
2. **Ambiguous Parts**: List specific words/phrases that are ambiguous
3. **Reasoning**: Brief explanation of why those parts are ambiguous
4. **Suggested Improvement**: How to make the story clearer""",

        "output_fields": '"Ambiguous Parts": "...",\n    "Reasoning": "...",\n    "Suggested Improvement": "..."',

        "important_notes": """- "Has Semantic Ambiguity" should be exactly "true" or "false" (lowercase)
- Focus on semantic ambiguity, not other types of ambiguity""",

        "sample_size": 50
    },

    "scope": {
        "display_name": "Scope Ambiguity",
        "background_definition": """Scope ambiguity in user stories occurs when the boundaries, conditions, or extent of functionality are unclear or can be interpreted in multiple ways. This leads to uncertainty about what exactly should be implemented and what the system should or should not do.""",

        "examples": """- Example: As a customer, I want to search for products so that I can find items.
  Reasoning: Unclear search criteria, filters, or result format - scope is too broad.
  Suggested Improvement:  As a customer, I want to search for products by name and category with price filters so that I can find items within my budget.""",

        "analysis_points": """1. **Has Scope Ambiguity**: "true" if the story contains scope ambiguity, "false" if it's clear and unambiguous
2. **Ambiguous Parts**: List specific parts that create scope uncertainty
3. **Missing Information**: What boundaries, conditions, or limits need clarification
4. **Suggested Questions**: Questions that would help clarify the scope""",

        "output_fields": '"Ambiguous Parts": "...",\n    "Missing Information": "...",\n    "Suggested Questions": "..."',

        "important_notes": """- "Has Scope Ambiguity" should be exactly "true" or "false"
- Focus specifically on scope-related ambiguity, not other types
- Consider whether a developer would have uncertainty about implementation boundaries""",

        "sample_size": 40
    },

    "actor": {
        "display_name": "Actor Ambiguity",
        "background_definition": """Actor ambiguity in user stories occurs when the participants, users, or system roles involved in the story are unclear or can be interpreted in multiple ways. This creates uncertainty about who performs actions, who receives results, and what permissions are required.""",

        "examples": """- Example: As a user, I want to access reports so that I can view data.
  Reasoning: Generic "user" role - unclear which type of user and what access level.
  Suggested Improvement:  As a sales manager, I want to access monthly sales reports so that I can review team performance.""",

        "analysis_points": """1. **Has Actor Ambiguity**: "true" if the story contains actor ambiguity, "false" if it's clear and unambiguous
2. **Ambiguous Roles**: List specific roles or actors that are unclear
3. **Missing Clarifications**: What role definitions or permissions need to be specified
4. **Suggested Improvements**: How to make the actor roles more specific""",

        "output_fields": '"Ambiguous Roles": "...",\n    "Missing Clarifications": "...",\n    "Suggested Improvements": "..."',

        "important_notes": """- "Has Actor Ambiguity" should be exactly "true" or "false"
- Focus specifically on actor/role-related ambiguity
- Consider whether developers would be uncertain about who performs actions or has permissions""",

        "sample_size": 30
    },

    "dependency": {
        "display_name": "Dependency Ambiguity",
        "background_definition": """Dependency ambiguity in user stories occurs when external dependencies, system integrations, or component relationships are unclear or undefined. This creates uncertainty about what external systems, APIs, or components are required for implementation.""",

        "examples": """- Example: As a user, I want to integrate with the system so that data flows.
  Reasoning: Unclear which system and what integration dependencies exist.
  Suggested Improvement:  As a developer, I want to integrate with the CRM API so that customer data syncs automatically.""",

        "analysis_points": """1. **Has Dependency Ambiguity**: "true" if the story contains dependency ambiguity, "false" if dependencies are clear
2. **External Dependencies**: List external systems or services that are mentioned but unclear
3. **Missing Integration Details**: What technical details or specifications need clarification
4. **Suggested Clarifications**: Specific questions to resolve dependency uncertainty""",

        "output_fields": '"External Dependencies": "...",\n    "Missing Integration Details": "...",\n    "Suggested Clarifications": "..."',

        "important_notes": """- "Has Dependency Ambiguity" should be exactly "true" or "false"
- Focus specifically on external dependencies and integration ambiguity
- Consider whether developers would need clarification about external systems or APIs""",

        "sample_size": 20
    },

    "priority": {
        "display_name": "Priority Ambiguity",
        "background_definition": """Priority ambiguity in user stories occurs when the importance, urgency, or relative priority of features is unclear or undefined. This creates uncertainty about implementation order, resource allocation, and business value assessment.""",

        "examples": """- Example: As a user, I want enhanced security so that I feel safe.
  Reasoning: No clear priority or urgency specified - "enhanced" is vague.
  Suggested Improvement:  As a user, I want two-factor authentication so that my account is protected from unauthorized access.""",

        "analysis_points": """1. **Has Priority Ambiguity**: "true" if the story contains priority ambiguity, "false" if priority is clear
2. **Ambiguous Priority Statements**: List specific phrases that create priority uncertainty
3. **Missing Priority Context**: What business value, urgency, or relative importance needs clarification
4. **Suggested Priority Framework**: How the priority could be clearly defined""",

        "output_fields": '"Ambiguous Priority Statements": "...",\n    "Missing Priority Context": "...",\n    "Suggested Priority Framework": "..."',

        "important_notes": """- "Has Priority Ambiguity" should be exactly "true" or "false"
- Focus specifically on priority, urgency, and importance ambiguity
- Consider whether product managers would need clarification for roadmap planning""",

        "sample_size": 20
    }
}


# ============================================================================
# 消歧方法的提示词（用于生成消歧）
# ============================================================================

DISAMBIGUATION_PROMPT_TEMPLATE = """
Objective
Analyze the given software requirement for ambiguities based on the description itself and the provided input context.
If the software requirement is ambiguous, your task is to clarify it by interpreting the ambiguous concepts, specifying necessary conditions, or using other methods.
Provide all possible disambiguations.

Important Rules
1. Perform detailed analyses before concluding whether the software requirement is clear or ambiguous.
2. Output disambiguations in the specified format.
3. Some seemingly unambiguous software requirements are actually ambiguous given that particular input context. So, do not forget to leverage the input to analyze whether the software requirement is underspecified.
4. Always generate exactly 10 disambiguated software requirements, even if some interpretations seem similar. Focus on capturing different aspects, perspectives, and implementation details.
5. Each disambiguated requirement should be a complete, standalone user story with the format "As a [user role], I want [functionality] so that [benefit]."

Example Analysis:
Original User Story: "As a store owner, I want to track order in order to find desired items with detailed specifications and comprehensive requirements"

Disambiguations:
1. As a store owner, I want a system that tracks the status of each customer order (from placement to delivery) so that I can locate specific items within those orders and view their detailed specifications and any customer-specified requirements associated with each item.
2. As a store owner, I want a searchable inventory module that lets me find desired product items by their detailed specifications (e.g., size, color, material) and comprehensive requirements (e.g., regulatory compliance, warranty, safety certifications).
3. As a store owner, I want a workflow tracking feature that monitors the fulfillment process of orders, ensuring that each shipped item meets its detailed specifications and all comprehensive requirements (such as packaging standards or shipping regulations) before dispatch.
4. As a store owner, I want reporting that aggregates order data to identify items that are frequently ordered (desired items) and provides detailed specifications and compliance requirements for those items to aid in inventory planning and supplier negotiations.
5. As a store owner, I want real-time tracking of order processing stages so that I can quickly identify bottlenecks and ensure all items meet their specifications before proceeding to the next stage.
6. As a store owner, I want an integrated search functionality across orders and inventory that allows me to find items based on their technical specifications and track which orders contain specific items.
7. As a store owner, I want automated validation checks that verify all items in an order meet their specified requirements and detailed specifications before shipping or delivery.
8. As a store owner, I want a dashboard that displays comprehensive order metrics including item specifications, compliance requirements, and fulfillment status to monitor order efficiency.
9. As a store owner, I want the ability to track individual items within multi-item orders, monitoring their specifications and requirements throughout the fulfillment process.
10. As a store owner, I want historical data analysis of order patterns and item specifications to help forecast demand and ensure inventory meets comprehensive requirements.

Output Format
Your output should follow this format:
Disambiguations:
1. [First disambiguated software requirement]
2. [Second disambiguated software requirement]
3. [Third disambiguated software requirement]
4. [Fourth disambiguated software requirement]
5. [Fifth disambiguated software requirement]
6. [Sixth disambiguated software requirement]
7. [Seventh disambiguated software requirement]
8. [Eighth disambiguated software requirement]
9. [Ninth disambiguated software requirement]
10. [Tenth disambiguated software requirement]

If the software requirement is clear and unambiguous, still provide 10 disambiguations that represent different implementation approaches, technical specifications, or operational scenarios.

Here is the software requirement you have to clarify:
{REQUIREMENT}
"""


# ============================================================================
# 伪代码生成的提示词（用于生成伪代码）
# ============================================================================

PSEUDOCODE_PROMPT_TEMPLATE = """
You are a software architect tasked with converting software requirements (user stories) into pseudocode.

Objective:
Convert the given software requirement into detailed pseudocode that describes the implementation logic and system behavior.

Guidelines:
1. Create clear, well-structured pseudocode that reflects the requirement
2. Include function definitions, control flow, and data structures
3. Focus on the functional logic, not syntax from a specific language
4. Use meaningful variable and function names
5. Include comments explaining key steps
6. The pseudocode should be implementation-independent but technically sound

Output Format:
Provide only the pseudocode without any additional explanation. Start directly with the pseudocode.

Here is the software requirement to convert to pseudocode:
{REQUIREMENT}
"""


# ============================================================================
# 辅助函数
# ============================================================================

def generate_ambiguity_prompt(user_story: str, ambiguity_type: str) -> str:
    """
    根据歧义类型参数生成对应的提示词

    Args:
        user_story: 用户故事文本
        ambiguity_type: 歧义类型（'semantic', 'scope', 'actor', 'dependency', 'priority'）

    Returns:
        生成的提示词

    Raises:
        ValueError: 如果ambiguity_type不被支持
    """
    if ambiguity_type not in AMBIGUITY_TYPE_CONFIGS:
        raise ValueError(f"Unknown ambiguity type: {ambiguity_type}. Supported types: {list(AMBIGUITY_TYPE_CONFIGS.keys())}")

    config = AMBIGUITY_TYPE_CONFIGS[ambiguity_type]

    return PROMPT_TEMPLATE_BASE.format(
        background_definition=config["background_definition"],
        ambiguity_type_lower=ambiguity_type.lower(),
        ambiguity_type=config["display_name"],
        examples=config["examples"],
        analysis_points=config["analysis_points"],
        output_fields=config["output_fields"],
        important_notes=config["important_notes"],
        user_story=user_story
    )


def get_ambiguity_config(ambiguity_type: str) -> dict:
    """
    获取指定歧义类型的配置

    Args:
        ambiguity_type: 歧义类型

    Returns:
        歧义类型的配置字典

    Raises:
        ValueError: 如果ambiguity_type不被支持
    """
    if ambiguity_type not in AMBIGUITY_TYPE_CONFIGS:
        raise ValueError(f"Unknown ambiguity type: {ambiguity_type}. Supported types: {list(AMBIGUITY_TYPE_CONFIGS.keys())}")

    return AMBIGUITY_TYPE_CONFIGS[ambiguity_type]


def get_recommended_sample_size(ambiguity_type: str) -> int:
    """
    获取指定歧义类型的推荐样本大小

    Args:
        ambiguity_type: 歧义类型

    Returns:
        推荐的样本大小

    Raises:
        ValueError: 如果ambiguity_type不被支持
    """
    config = get_ambiguity_config(ambiguity_type)
    return config.get("sample_size", 30)


def get_all_ambiguity_types() -> list:
    """
    获取所有支持的歧义类型列表

    Returns:
        歧义类型列表
    """
    return list(AMBIGUITY_TYPE_CONFIGS.keys())
