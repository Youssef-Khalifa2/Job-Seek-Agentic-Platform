from langchain_aws import ChatBedrock
from langchain_mistralai import ChatMistralAI
import config

class LLMRegistry:
    """Central registry for all LLMs used in the system (December 2025)."""

    @staticmethod
    def get_haiku():
        """Claude Haiku 4.5 - Fast, cheap, structured tasks (200K context)."""
        return ChatBedrock(
            model_id="anthropic.claude-haiku-4-5-20251001-v1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.3, "max_tokens": 4096}
        )

    @staticmethod
    def get_sonnet():
        """Claude Sonnet 4.5 - Complex reasoning, editing (200K context)."""
        return ChatBedrock(
            model_id="anthropic.claude-sonnet-4-5-20250929-v1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.5, "max_tokens": 8192}
        )

    @staticmethod
    def get_nova_micro():
        """Amazon Nova Micro - Ultra-cheap routing/verification ($0.035/1M)."""
        return ChatBedrock(
            model_id="amazon.nova-micro-v1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.2, "max_tokens": 2048}
        )

    @staticmethod
    def get_nova_lite():
        """Amazon Nova Lite - Cheap semantic matching ($0.06/1M, 300K context)."""
        return ChatBedrock(
            model_id="amazon.nova-lite-v1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.3, "max_tokens": 4096}
        )

    @staticmethod
    def get_gpt_oss120():
        """GPT-OSS-120B - Strong instruction following for editing ($1.00/1M)."""
        return ChatBedrock(
            model_id="openai.gpt-oss-120b-1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.5, "max_tokens": 8192}
        )

    @staticmethod
    def get_gpt_oss20():
        """GPT-OSS-120B - Strong instruction following for editing ($1.00/1M)."""
        return ChatBedrock(
            model_id="openai.gpt-oss-20b-1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.5, "max_tokens": 8192}
        )

    @staticmethod
    def get_nova_pro():
        """Amazon Nova Pro - Strong instruction following for editing ($1.00/1M)."""
        return ChatBedrock(
            model_id="amazon.nova-pro-v1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.5, "max_tokens": 8192}
        )
    
    @staticmethod
    def get_deepseek_v3():
        """DeepSeek V3 - Strong instruction following for editing ($1.00/1M)."""
        return ChatBedrock(
            model_id="deepseek.v3-v1:0",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.2, "max_tokens": 2048}
        )

    @staticmethod
    def get_pixtral_large():
        """Pixtral Large - FREE vision model from Mistral for layout analysis."""
        return ChatMistralAI(
            model="pixtral-large-2411",
            mistral_api_key=config.MISTRAL_API_KEY,
            temperature=0.3,
            max_tokens=4096
        )

    @staticmethod
    def get_ministral_3b():
        """Ministral 3B - FREE fast model for grammar/style checks."""
        return ChatMistralAI(
            model="ministral-3b-2410",
            mistral_api_key=config.MISTRAL_API_KEY,
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )

    @staticmethod
    def get_mistral_small():
        """Fast, JSON mode, free tier."""
        return ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=config.MISTRAL_API_KEY,
            response_format={"type": "json_object"},
            temperature=0.3
        )

    
    @staticmethod
    def get_mistral_medium():
        """Fast, JSON mode, free tier."""
        return ChatMistralAI(
            model="mistral-medium-latest",
            mistral_api_key=config.MISTRAL_API_KEY,
            response_format={"type": "json_object"},
            temperature=0.3
        )

    @staticmethod
    def get_mistral_large():
        """Fast, JSON mode, free tier."""
        return ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=config.MISTRAL_API_KEY,
            response_format={"type": "json_object"},
            temperature=0.3
        )