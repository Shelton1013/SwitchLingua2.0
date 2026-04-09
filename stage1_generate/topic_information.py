"""
SwitchLingua 2.0 — Topic Information Provider System
话题信息获取系统：为对话 Agent 注入真实的、领域相关的信息

核心设计：
  不同话题调用不同的 API，获取具体的、可讨论的信息。
  例如：technology 话题 → 调用 HackerNews API 获取最新科技新闻
       academic 话题 → 调用 arXiv API 获取最新论文
       finance 话题 → 调用 NewsAPI 获取财经新闻

  这些信息注入到对话 prompt 中，让两个 Agent 有具体的"谈资"，
  避免生成空洞、模板化的对话。

Provider 体系：
  - Tier 1 (无需 API key): HackerNews, arXiv, Wikipedia
  - Tier 2 (需免费 API key): NewsAPI, TMDB, Semantic Scholar
  - Fallback: DuckDuckGo 网页搜索（无需 API key）

使用方式：
    router = TopicRouter("provider_config.yaml")
    snippets = router.fetch("technology")
    # → [InformationSnippet(title="...", content="...", source="hackernews"), ...]
"""

import json
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import yaml

try:
    import requests
except ImportError:
    raise ImportError("需要安装 requests: pip install requests")

logger = logging.getLogger("topic_information")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class InformationSnippet:
    """一条话题信息片段，用于注入对话 prompt"""
    title: str              # 标题
    content: str            # 内容摘要（100-300 字符）
    source: str             # 来源 provider 名
    url: str = ""           # 原始链接（可选）
    language: str = "en"    # 信息语言
    relevance: float = 1.0  # 相关度 0-1


# ============================================================
# Base Provider
# ============================================================

class BaseProvider(ABC):
    """
    信息 Provider 的抽象基类。
    每个 provider 负责从一个特定 API 获取信息并转化为 InformationSnippet。
    """

    def __init__(self, config: dict, api_keys: dict, global_config: dict):
        """
        参数：
        - config: provider 的特定配置（从 provider_config.yaml 中读取）
        - api_keys: 全局 API key 字典
        - global_config: 全局配置（超时、缓存等）
        """
        self.config = config
        self.api_keys = api_keys
        self.timeout = global_config.get("request_timeout", 15)
        self.max_length = global_config.get("snippet_max_length", 300)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SwitchLingua/2.0 Research Bot"
        })

    @abstractmethod
    def fetch(self, params: dict) -> list[InformationSnippet]:
        """
        从 API 获取信息。
        子类实现具体的 API 调用逻辑。
        返回 InformationSnippet 列表。
        """
        pass

    def _truncate(self, text: str) -> str:
        """将文本截断到 max_length"""
        if len(text) <= self.max_length:
            return text
        return text[:self.max_length - 3] + "..."

    def _clean_html(self, text: str) -> str:
        """去除 HTML 标签"""
        return re.sub(r'<[^>]+>', '', text).strip()


# ============================================================
# Tier 1: 无需 API Key 的 Provider
# ============================================================

class HackerNewsProvider(BaseProvider):
    """
    HackerNews API — 获取最新科技新闻/讨论。
    API 文档: https://github.com/HackerNews/API
    无需 API key，无速率限制。

    适用话题: technology, academic, work
    """

    BASE_URL = "https://hacker-news.firebaseio.com/v0"

    def fetch(self, params: dict) -> list[InformationSnippet]:
        category = params.get("category", "top")  # top / best / new
        limit = params.get("limit", 10)

        try:
            # 获取 story ID 列表
            url = f"{self.BASE_URL}/{category}stories.json"
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            story_ids = resp.json()[:limit]
        except Exception as e:
            logger.warning(f"HackerNews: 获取 story 列表失败: {e}")
            return []

        snippets = []
        for sid in story_ids[:limit]:
            try:
                item_url = f"{self.BASE_URL}/item/{sid}.json"
                resp = self.session.get(item_url, timeout=self.timeout)
                item = resp.json()

                if not item or item.get("type") != "story":
                    continue

                title = item.get("title", "")
                # HackerNews story 通常没有 text，用 title + score + comments 构建
                score = item.get("score", 0)
                comments = item.get("descendants", 0)
                url = item.get("url", "")

                content = (
                    f"{title}。"
                    f"这个帖子获得了 {score} 分和 {comments} 条评论。"
                )

                snippets.append(InformationSnippet(
                    title=title,
                    content=self._truncate(content),
                    source="hackernews",
                    url=url,
                    language="en",
                ))
            except Exception:
                continue

        return snippets


class ArxivProvider(BaseProvider):
    """
    arXiv API — 获取最新学术论文信息。
    API 文档: https://info.arxiv.org/help/api/
    无需 API key。

    适用话题: academic, technology
    """

    BASE_URL = "https://export.arxiv.org/api/query"

    def fetch(self, params: dict) -> list[InformationSnippet]:
        categories = params.get("categories", ["cs.CL", "cs.AI"])
        max_results = params.get("max_results", 10)

        # 构建 arXiv 查询
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        query_params = {
            "search_query": cat_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            resp = self.session.get(
                self.BASE_URL, params=query_params, timeout=self.timeout
            )
            resp.raise_for_status()
            text = resp.text
        except Exception as e:
            logger.warning(f"arXiv: API 调用失败: {e}")
            return []

        # 简单解析 Atom XML（不引入 lxml 依赖）
        snippets = []
        entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
        for entry in entries:
            title_m = re.search(r'<title[^>]*>(.*?)</title>', entry, re.DOTALL)
            summary_m = re.search(r'<summary[^>]*>(.*?)</summary>', entry, re.DOTALL)
            link_m = re.search(r'<id>(.*?)</id>', entry)

            if not title_m:
                continue

            title = self._clean_html(title_m.group(1)).replace("\n", " ").strip()
            summary = ""
            if summary_m:
                summary = self._clean_html(summary_m.group(1)).replace("\n", " ").strip()
            url = link_m.group(1).strip() if link_m else ""

            content = f"{title}。{summary}" if summary else title

            snippets.append(InformationSnippet(
                title=title,
                content=self._truncate(content),
                source="arxiv",
                url=url,
                language="en",
            ))

        return snippets


class WikipediaProvider(BaseProvider):
    """
    Wikipedia REST API — 获取百科知识。
    API 文档: https://en.wikipedia.org/api/rest_v1/
    无需 API key。

    适用话题: 所有话题的通用知识补充
    """

    BASE_URL = "https://en.wikipedia.org/api/rest_v1"

    def fetch(self, params: dict) -> list[InformationSnippet]:
        search_topics = params.get("search_topics", [])
        snippets = []

        for topic in search_topics:
            try:
                # 搜索相关页面
                search_url = (
                    f"{self.BASE_URL}/page/summary/{topic.replace(' ', '_')}"
                )
                resp = self.session.get(search_url, timeout=self.timeout)

                if resp.status_code == 404:
                    # 页面不存在，尝试搜索
                    search_api = (
                        f"https://en.wikipedia.org/w/api.php"
                        f"?action=query&list=search&srsearch={topic}"
                        f"&format=json&utf8=1&srlimit=3"
                    )
                    resp2 = self.session.get(search_api, timeout=self.timeout)
                    results = resp2.json().get("query", {}).get("search", [])
                    for r in results[:1]:
                        snippets.append(InformationSnippet(
                            title=r.get("title", ""),
                            content=self._truncate(
                                self._clean_html(r.get("snippet", ""))
                            ),
                            source="wikipedia",
                            language="en",
                        ))
                    continue

                resp.raise_for_status()
                data = resp.json()

                title = data.get("title", topic)
                extract = data.get("extract", "")

                if extract:
                    snippets.append(InformationSnippet(
                        title=title,
                        content=self._truncate(extract),
                        source="wikipedia",
                        url=data.get("content_urls", {})
                              .get("desktop", {})
                              .get("page", ""),
                        language="en",
                    ))
            except Exception as e:
                logger.debug(f"Wikipedia: '{topic}' 查询失败: {e}")
                continue

        return snippets


# ============================================================
# Tier 2: 需要 API Key 的 Provider
# ============================================================

class NewsAPIProvider(BaseProvider):
    """
    NewsAPI — 获取全球新闻。
    https://newsapi.org/ — 免费 100 次/天
    需要 API key。

    适用话题: work, finance, daily_life, 以及任何新闻类话题
    """

    BASE_URL = "https://newsapi.org/v2"

    def fetch(self, params: dict) -> list[InformationSnippet]:
        api_key = self.api_keys.get("newsapi", "")
        if not api_key:
            logger.debug("NewsAPI: 未配置 API key，跳过")
            return []

        category = params.get("category", "general")
        query = params.get("query", "")

        try:
            if query:
                url = f"{self.BASE_URL}/everything"
                req_params = {
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": 10,
                    "apiKey": api_key,
                    "language": "en",
                }
            else:
                url = f"{self.BASE_URL}/top-headlines"
                req_params = {
                    "category": category,
                    "pageSize": 10,
                    "apiKey": api_key,
                    "language": "en",
                }

            resp = self.session.get(url, params=req_params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"NewsAPI: 调用失败: {e}")
            return []

        snippets = []
        for article in data.get("articles", []):
            title = article.get("title", "")
            description = article.get("description", "") or ""
            content = article.get("content", "") or ""
            # 优先用 description，更简洁
            body = description if description else content

            if title and body:
                snippets.append(InformationSnippet(
                    title=title,
                    content=self._truncate(f"{title}。{body}"),
                    source="newsapi",
                    url=article.get("url", ""),
                    language="en",
                ))

        return snippets


class TMDBProvider(BaseProvider):
    """
    TMDB API — 获取热门电影/电视信息。
    https://www.themoviedb.org/settings/api — 免费
    需要 API key。

    适用话题: entertainment
    """

    BASE_URL = "https://api.themoviedb.org/3"

    def fetch(self, params: dict) -> list[InformationSnippet]:
        api_key = self.api_keys.get("tmdb", "")
        if not api_key:
            logger.debug("TMDB: 未配置 API key，跳过")
            return []

        try:
            url = f"{self.BASE_URL}/trending/all/week"
            resp = self.session.get(
                url,
                params={"api_key": api_key, "language": "en-US"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"TMDB: 调用失败: {e}")
            return []

        snippets = []
        for item in data.get("results", [])[:10]:
            title = item.get("title") or item.get("name", "")
            overview = item.get("overview", "")
            media_type = item.get("media_type", "")
            vote = item.get("vote_average", 0)

            if title:
                content = f"{title} ({media_type}, 评分 {vote}/10)。{overview}"
                snippets.append(InformationSnippet(
                    title=title,
                    content=self._truncate(content),
                    source="tmdb",
                    language="en",
                ))

        return snippets


class SemanticScholarProvider(BaseProvider):
    """
    Semantic Scholar API — 获取学术论文信息。
    https://api.semanticscholar.org/ — 免费 (100 req/5min)
    无需 API key（有 key 限额更高）。

    适用话题: academic
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def fetch(self, params: dict) -> list[InformationSnippet]:
        fields = params.get("fields_of_study", ["Computer Science"])
        query = params.get("query", "code-switching multilingual")

        try:
            url = f"{self.BASE_URL}/paper/search"
            req_params = {
                "query": query,
                "limit": 10,
                "fields": "title,abstract,year,citationCount",
                "fieldsOfStudy": ",".join(fields),
                "sort": "citationCount:desc",
            }
            resp = self.session.get(url, params=req_params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Semantic Scholar: 调用失败: {e}")
            return []

        snippets = []
        for paper in data.get("data", []):
            title = paper.get("title", "")
            abstract = paper.get("abstract", "") or ""
            year = paper.get("year", "")
            citations = paper.get("citationCount", 0)

            if title:
                content = (
                    f"{title} ({year}, {citations} citations)。"
                    f"{abstract[:200]}"
                )
                snippets.append(InformationSnippet(
                    title=title,
                    content=self._truncate(content),
                    source="semantic_scholar",
                    language="en",
                ))

        return snippets


# ============================================================
# Fallback: Web Search Provider
# ============================================================

class WebSearchProvider(BaseProvider):
    """
    DuckDuckGo HTML 搜索 — 通用的 fallback 搜索。
    无需 API key。
    通过解析搜索结果页面获取信息。

    适用话题: 所有话题的最后兜底方案
    """

    SEARCH_URL = "https://html.duckduckgo.com/html"

    def fetch(self, params: dict) -> list[InformationSnippet]:
        query_template = params.get(
            "query_template", "{topic} latest news"
        )
        topic = params.get("_topic_label", "")
        query = query_template.replace("{topic}", topic)

        try:
            resp = self.session.post(
                self.SEARCH_URL,
                data={"q": query},
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SwitchLingua/2.0)"},
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            logger.warning(f"WebSearch: 搜索失败: {e}")
            return []

        # 解析 DuckDuckGo 结果（简单正则提取）
        snippets = []
        # DuckDuckGo HTML 结果中的 snippet 格式
        results = re.findall(
            r'class="result__a"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet">(.*?)</div>',
            html, re.DOTALL
        )

        for title_raw, snippet_raw in results[:10]:
            title = self._clean_html(title_raw).strip()
            snippet = self._clean_html(snippet_raw).strip()

            if title and snippet:
                snippets.append(InformationSnippet(
                    title=title,
                    content=self._truncate(f"{title}。{snippet}"),
                    source="web_search",
                    language="en",
                ))

        return snippets


# ============================================================
# Provider Registry
# ============================================================

PROVIDER_REGISTRY = {
    "hackernews": HackerNewsProvider,
    "arxiv": ArxivProvider,
    "wikipedia": WikipediaProvider,
    "newsapi": NewsAPIProvider,
    "tmdb": TMDBProvider,
    "semantic_scholar": SemanticScholarProvider,
    "web_search": WebSearchProvider,
}


# ============================================================
# TopicRouter — 话题路由器
# ============================================================

class TopicRouter:
    """
    话题信息路由器。

    根据话题类型，按配置的 provider 链依次尝试获取信息。
    第一个成功返回足够结果的 provider 的信息被采用。
    支持结果缓存，避免重复 API 调用。

    使用方式：
        router = TopicRouter("provider_config.yaml")
        snippets = router.fetch("technology")
        # → 返回 3 条科技新闻 InformationSnippet

        # 带话题标签（用于 web search 查询）
        snippets = router.fetch("technology", topic_label="人工智能")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        参数：
        - config_path: provider_config.yaml 的路径
        """
        if config_path is None:
            config_path = Path(__file__).parent / "provider_config.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.api_keys = self.config.get("api_keys", {})
        self.global_config = self.config.get("global", {})
        self.topic_providers = self.config.get("topic_providers", {})
        self.max_snippets = self.global_config.get("max_snippets_per_topic", 3)

        # 缓存：{cache_key -> (timestamp, snippets)}
        self._cache: dict[str, tuple[float, list[InformationSnippet]]] = {}
        self._cache_ttl = self.global_config.get("cache_ttl_hours", 24) * 3600

        # 实例化 provider 对象
        self._providers: dict[str, BaseProvider] = {}
        for name, cls in PROVIDER_REGISTRY.items():
            provider_cfg = self.config.get("providers", {}).get(name, {})
            self._providers[name] = cls(
                config=provider_cfg,
                api_keys=self.api_keys,
                global_config=self.global_config,
            )

    def _cache_key(self, topic_id: str, topic_label: str) -> str:
        """生成缓存 key"""
        raw = f"{topic_id}_{topic_label}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[list[InformationSnippet]]:
        """检查缓存"""
        if key in self._cache:
            ts, snippets = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return snippets
            else:
                del self._cache[key]
        return None

    def _set_cache(self, key: str, snippets: list[InformationSnippet]):
        """写入缓存"""
        self._cache[key] = (time.time(), snippets)

    def fetch(
        self,
        topic_id: str,
        topic_label: str = "",
        max_snippets: Optional[int] = None,
    ) -> list[InformationSnippet]:
        """
        获取话题相关信息。

        参数：
        - topic_id: 话题 ID（如 "technology", "academic"）
        - topic_label: 话题中文标签（用于 web search 查询增强）
        - max_snippets: 最大返回条数（默认使用全局配置）

        返回：
        - InformationSnippet 列表

        流程：
        1. 检查缓存
        2. 按配置的 provider 链依次尝试
        3. 第一个返回足够结果的 provider 终止
        4. 结果写入缓存
        """
        if max_snippets is None:
            max_snippets = self.max_snippets

        # 1. 检查缓存
        cache_key = self._cache_key(topic_id, topic_label)
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"TopicRouter: 命中缓存 ({topic_id})")
            return cached[:max_snippets]

        # 2. 获取该话题的 provider 配置
        provider_chain = self.topic_providers.get(topic_id, [])
        if not provider_chain:
            # 未配置的话题 → 使用 web search fallback
            provider_chain = [
                {"provider": "web_search",
                 "params": {"query_template": "{topic} discussion"}}
            ]

        # 3. 按 provider 链依次尝试
        all_snippets = []
        for provider_cfg in provider_chain:
            provider_name = provider_cfg.get("provider", "")
            params = dict(provider_cfg.get("params", {}))
            # 注入 topic_label 供 web_search 使用
            params["_topic_label"] = topic_label or topic_id

            provider = self._providers.get(provider_name)
            if provider is None:
                logger.warning(f"TopicRouter: 未知 provider '{provider_name}'")
                continue

            try:
                snippets = provider.fetch(params)
                if snippets:
                    all_snippets.extend(snippets)
                    logger.info(
                        f"TopicRouter: {provider_name} 返回 {len(snippets)} 条 "
                        f"({topic_id})"
                    )
                    # 如果已获取足够数量，提前终止
                    if len(all_snippets) >= max_snippets:
                        break
            except Exception as e:
                logger.warning(
                    f"TopicRouter: {provider_name} 失败 ({topic_id}): {e}"
                )
                continue

        # 去重（按 title）
        seen_titles = set()
        unique_snippets = []
        for s in all_snippets:
            title_key = s.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_snippets.append(s)

        result = unique_snippets[:max_snippets]

        # 4. 写入缓存
        if result:
            self._set_cache(cache_key, result)

        return result

    def format_for_prompt(
        self, snippets: list[InformationSnippet], language: str = "zh"
    ) -> str:
        """
        将信息片段格式化为可注入 prompt 的文本。

        参数：
        - snippets: InformationSnippet 列表
        - language: 输出语言 ("zh" 用中文包装, "en" 用英文)

        返回：
        - 格式化的文本字符串
        """
        if not snippets:
            return ""

        if language == "zh":
            header = "以下是一些与当前话题相关的真实信息，你们可以在对话中自然地讨论这些内容："
        else:
            header = (
                "Here is some real information related to the topic. "
                "You can naturally discuss these in your conversation:"
            )

        lines = [header, ""]
        for i, s in enumerate(snippets, 1):
            lines.append(f"{i}. {s.content}")
            if s.source:
                lines.append(f"   (来源: {s.source})")
            lines.append("")

        if language == "zh":
            lines.append(
                "注意：\n"
                "- 只需自然地提及其中1个你感兴趣的点即可，不要试图覆盖所有信息\n"
                "- 不要照搬原文的词汇，用你自己日常的说法来表达\n"
                "- 不要反复提及信息来源的名称或标题，提一次就够了\n"
                "- 把信息当作你已经知道的背景知识，而不是在复述给对方"
            )
        else:
            lines.append(
                "Note: You don't need to cover all of these. Just naturally "
                "mention or discuss 1-2 points that interest you. "
                "Express them in your own words."
            )

        return "\n".join(lines)


# ============================================================
# CLI 测试入口
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 Topic Information Provider"
    )
    parser.add_argument(
        "--topic", type=str, default="technology",
        help="话题 ID (technology/academic/work/daily_life/food/entertainment/travel/finance)"
    )
    parser.add_argument(
        "--topic-label", type=str, default="",
        help="话题标签（用于搜索增强）"
    )
    parser.add_argument(
        "--max-snippets", type=int, default=3,
        help="最大返回条数"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="provider_config.yaml 路径"
    )
    args = parser.parse_args()

    router = TopicRouter(args.config)
    snippets = router.fetch(
        args.topic,
        topic_label=args.topic_label,
        max_snippets=args.max_snippets,
    )

    if snippets:
        print(f"\n获取到 {len(snippets)} 条信息：\n")
        for i, s in enumerate(snippets, 1):
            print(f"[{i}] {s.title}")
            print(f"    {s.content}")
            print(f"    来源: {s.source}")
            if s.url:
                print(f"    URL: {s.url}")
            print()

        print("=" * 60)
        print("格式化 prompt 输出：")
        print("=" * 60)
        print(router.format_for_prompt(snippets))
    else:
        print("未获取到任何信息")
