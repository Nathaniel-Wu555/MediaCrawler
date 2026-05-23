# -*- coding: utf-8 -*-
# Copyright (c) 2026
# Report generator for MediaCrawler

import glob
import json
import os
import re
import pathlib
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

import config
from tools.utils import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

# DeepSeek API 配置（直接硬编码）
AI_API_KEY = "sk-c63ccef16b4446eea900ee6b42d1820a"  # 注意：硬编码 API Key，生产请改为环境变量
AI_BASE_URL = "https://api.deepseek.com/v1"
AI_MODEL = "deepseek-chat"
AI_VISION_MODEL = "deepseek-vision"
REPORT_AI_SCORE_THRESHOLD = float(os.getenv("REPORT_AI_SCORE_THRESHOLD", "5.0"))
REPORT_ENABLE_AI_SCORE = os.getenv("REPORT_ENABLE_AI_SCORE", "true").lower() in {"true", "1", "yes"}
REPORT_ENABLE_CONTENT_ENRICH = os.getenv("REPORT_ENABLE_CONTENT_ENRICH", "true").lower() in {"true", "1", "yes"}
REPORT_ENABLE_MEDIA_ANALYSIS = os.getenv("REPORT_ENABLE_MEDIA_ANALYSIS", "true").lower() in {"true", "1", "yes"}
REPORT_ENABLE_SEMANTIC_DEDUP = os.getenv("REPORT_ENABLE_SEMANTIC_DEDUP", "true").lower() in {"true", "1", "yes"}
REPORT_MAX_COMMENT_PREVIEW = int(os.getenv("REPORT_MAX_COMMENT_PREVIEW", "20"))


CONTENT_FIELD_ALIASES = {
    "note_id": ["note_id", "content_id", "aweme_id", "post_id", "id"],
    "title": ["title", "note_title", "video_title", "content_title", "subject", "name"],
    "desc": ["desc", "content", "text", "description", "note_desc", "detail", "message"],
    "note_url": ["note_url", "share_url", "url", "article_url", "link", "href"],
    "video_url": ["video_url", "video_play_url", "video_link", "play_url"],
    "nickname": ["nickname", "user_name", "author", "creator", "screen_name"],
    "ip_location": ["ip_location", "location"],
    "type": ["type", "content_type", "video_type"],
}

COMMENT_FIELD_ALIASES = {
    "comment_id": ["comment_id", "id", "reply_id"],
    "note_id": ["note_id", "content_id", "aweme_id", "post_id", "target_id"],
    "content": ["content", "comment", "text", "comment_text", "message"],
    "nickname": ["nickname", "user_name", "author", "creator", "screen_name"],
    "create_time": ["create_time", "time", "publish_time", "created_at"],
}

USELESS_WORDS = {"哦", "嗯", "额", "啊", "哎", "哼", "啥", "哈哈", "呵呵", "嘿嘿", "嘻嘻"}
PURE_SYMBOL_REGEX = re.compile(r'^[^\w\s]+$')
ONLY_AT_REGEX = re.compile(r'^@\S+\s*$')
EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002500-\U00002BEF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u200d"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\ufe0f"
    u"\u3030"
    "]+",
    flags=re.UNICODE
)


class MediaCrawlerReportGenerator:
    def __init__(self, platform: str):
        self.platform = platform
        self.base_path = self._get_data_base_path(platform)
        self.report_dir = self.base_path / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.ai_client = self._create_ai_client()
        self._whisper_model = None

    def _create_ai_client(self) -> Any:
        """创建 DeepSeek AI 客户端（强制使用 AI 模式）"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK is not installed. Please install the openai package.")

        if not AI_API_KEY:
            raise ValueError("AI API key is not configured. Please set AI_API_KEY in the source or environment.")

        try:
            return OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize AI client: {exc}")

    def _get_whisper_model(self):
        if not WHISPER_AVAILABLE:
            return None
        if getattr(self, "_whisper_model", None) is None:
            try:
                self._whisper_model = whisper.load_model("small")
            except Exception as exc:
                logger.warning(f"Unable to load Whisper model: {exc}")
                self._whisper_model = None
        return self._whisper_model

    def _get_data_base_path(self, platform: str) -> pathlib.Path:
        if config.SAVE_DATA_PATH:
            return pathlib.Path(config.SAVE_DATA_PATH) / platform
        return pathlib.Path("data") / platform

    def _find_data_files(self, item_type: str) -> List[pathlib.Path]:
        patterns = [f"{self.base_path}/**/*_{item_type}_*{ext}" for ext in [".csv", ".jsonl", ".json", ".xlsx", ".xls"]]
        file_paths = []
        for pattern in patterns:
            file_paths.extend(glob.glob(pattern, recursive=True))
        return [pathlib.Path(p) for p in sorted(set(file_paths))]

    def _load_jsonl(self, path: pathlib.Path) -> List[Dict[str, Any]]:
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items

    def _load_file(self, path: pathlib.Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
                df = df.where(pd.notnull(df), None)
                return df.to_dict(orient="records")
            except Exception as exc:
                logger.warning(f"Failed to read CSV '{path}': {exc}")
                return []

        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                df = pd.read_excel(path, dtype=str)
                df = df.where(pd.notnull(df), None)
                return df.to_dict(orient="records")
            except Exception as exc:
                logger.warning(f"Failed to read Excel '{path}': {exc}")
                return []

        if path.suffix.lower() == ".jsonl":
            return self._load_jsonl(path)

        if path.suffix.lower() == ".json":
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [data]
            except Exception as exc:
                logger.warning(f"Failed to read JSON '{path}': {exc}")
                return []

        return []

    def _find_best_field(self, record: Dict[str, Any], names: List[str]) -> Any:
        for name in names:
            if name in record and record[name] not in (None, "", "nan"):
                return record[name]
        lower_record = {k.lower(): v for k, v in record.items()}
        for name in names:
            if name.lower() in lower_record and lower_record[name.lower()] not in (None, "", "nan"):
                return lower_record[name.lower()]
        return None

    def _normalize_record(self, record: Dict[str, Any], field_aliases: Dict[str, List[str]]) -> Dict[str, Any]:
        normalized = {}
        for target, names in field_aliases.items():
            value = self._find_best_field(record, names)
            if value is None:
                normalized[target] = None
            else:
                normalized[target] = str(value).strip()
        return normalized

    def _normalize_content(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return self._normalize_record(record, CONTENT_FIELD_ALIASES)

    def _normalize_comment(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return self._normalize_record(record, COMMENT_FIELD_ALIASES)

    def _load_records(self, item_type: str) -> List[Dict[str, Any]]:
        files = self._find_data_files(item_type)
        items: List[Dict[str, Any]] = []
        for path in files:
            items.extend(self._load_file(path))
        return items

    def load_contents(self) -> pd.DataFrame:
        records = [self._normalize_content(item) for item in self._load_records("contents")]
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["note_id"] = df["note_id"].astype(str).fillna("")
        return df

    def load_comments(self) -> pd.DataFrame:
        records = [self._normalize_comment(item) for item in self._load_records("comments")]
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["note_id"] = df["note_id"].astype(str).fillna("")
        return df

    def _is_useless_comment(self, comment: str) -> bool:
        if not comment:
            return True
        c = str(comment).strip().lower()
        if c == "" or c in {"nan", "null", "none"}:
            return True
        if PURE_SYMBOL_REGEX.match(c):
            return True
        if ONLY_AT_REGEX.match(c):
            return True
        if EMOJI_PATTERN.fullmatch(c):
            return True
        if c in USELESS_WORDS:
            return True
        if c.isdigit() or c.isalpha():
            return True
        if len(c) <= 2:
            return True
        return False

    def filter_comments(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        if comments_df.empty:
            return comments_df
        # 删除评论筛选：保留原始评论用于AI分析
        return comments_df

    def normalize_url(self, url: str) -> str:
        if not url:
            return ""
        try:
            parsed = re.sub(r"^https?://", "", str(url).strip())
            parsed = parsed.split("?")[0].rstrip("/")
            return parsed
        except Exception:
            return str(url).strip()

    def deduplicate_contents(self, contents_df: pd.DataFrame) -> pd.DataFrame:
        if contents_df.empty:
            return contents_df
        contents_df = contents_df.copy()
        contents_df["norm_url"] = contents_df["note_url"].astype(str).apply(self.normalize_url)
        deduped = contents_df.drop_duplicates(subset=["norm_url"]).drop(columns=["norm_url"])
        if REPORT_ENABLE_SEMANTIC_DEDUP:
            deduped["content_signature"] = (
                deduped["title"].fillna("") + " " + deduped["desc"].fillna("")
            ).str.lower().str.replace(r"\s+", " ", regex=True)
            deduped = deduped.drop_duplicates(subset=["content_signature"]).drop(columns=["content_signature"])
            logger.info(f"Semantic dedup applied, remaining rows: {len(deduped)}")
        logger.info(f"Deduplicated contents: {len(contents_df)} -> {len(deduped)}")
        return deduped

    def _extract_keywords(self, texts: List[str], top_n: int = 5) -> List[str]:
        words: List[str] = []
        for text in texts:
            if not text:
                continue
            for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(text)):
                token = token.strip()
                if len(token) <= 1:
                    continue
                if token.lower() in USELESS_WORDS:
                    continue
                words.append(token.lower())
        if not words:
            return []
        most_common = Counter(words).most_common(top_n)
        return [word for word, _ in most_common]

    # 已删除：_estimate_comment_attitude 和 _build_fallback_summary
    # 降级逻辑已移除，所有分析均通过 AI 完成

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        return None

    def _extract_ai_score(self, value: Any) -> float:
        if value is None:
            return 0.0
        text = str(value)
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return 0.0
        return 0.0

    def _build_ai_prompt(self, title: str, desc: str, comments: List[str], extra_notes: str = "") -> str:
        comment_text = chr(10).join(comments[:REPORT_MAX_COMMENT_PREVIEW])
        prompt = (
            f"请综合标题和内容生成摘要，并分析评论区观点与整体倾向，严格返回 JSON 格式结果，字段包括 summary、comment_analysis、ai_score、tags、media_insights。"
            f" summary 80-140字，comment_analysis 100-150字，说明评论区主要观点、支持/反对态度、支持比例和总体倾向；"
            f"ai_score 0-10 的数字；tags 3-5个关键词；media_insights 40-80字，补充图文内容洞察。\n"
            f"标题：{title}\n"
            f"内容：{desc}\n"
            f"评论：{comment_text}\n"
        )
        if extra_notes:
            prompt += f"额外信息：{extra_notes}\n"
        return prompt

    def _call_ai_analysis(self, title: str, desc: str, comments: List[str], extra_notes: str = "") -> Dict[str, str]:
        """调用 DeepSeek AI 进行分析（强制使用 AI，无本地降级）。"""
        prompt = self._build_ai_prompt(title, desc, comments, extra_notes)

        response = self.ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )

        text = response.choices[0].message.content.strip()
        data = self._extract_json_object(text) or {}

        tags = data.get("tags", data.get("keywords", ""))
        if isinstance(tags, list):
            tags = ", ".join(str(item) for item in tags)

        summary = str(data.get("summary", data.get("摘要", "") or "")).strip()
        comment_analysis = str(
            data.get("comment_analysis", data.get("commentAnalysis", data.get("评论区分析", ""))) or ""
        ).strip()
        ai_score_value = data.get("ai_score", data.get("score", data.get("AI评分", "5")))
        media_insights = str(
            data.get("media_insights", data.get("mediaInsights", data.get("媒体洞察", ""))) or ""
        ).strip()

        return {
            "summary": summary,
            "comment_analysis": comment_analysis,
            "ai_score": str(ai_score_value).strip(),
            "tags": str(tags),
            "media_insights": media_insights,
        }

    def _extract_image_urls(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"https?://[^\s'\"]+\.(?:png|jpg|jpeg|gif|bmp|webp)", text)

    def _extract_video_urls(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"https?://[^\s'\"]+\.(?:mp4|m3u8|mov|flv|gifv|webm)", text)

    def _download_video_audio(self, video_url: str) -> Optional[pathlib.Path]:
        if not YTDLP_AVAILABLE:
            return None
        try:
            tmp_dir = self.report_dir / "tmp_media"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_file = tmp_dir / "video.%(ext)s"
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(out_file),
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                filename = ydl.prepare_filename(info)
            return pathlib.Path(filename)
        except Exception as exc:
            logger.warning(f"Video download failed for {video_url}: {exc}")
            return None

    def _transcribe_audio(self, audio_path: pathlib.Path) -> str:
        model = self._get_whisper_model()
        if not model or not audio_path.exists():
            return ""
        try:
            result = model.transcribe(str(audio_path))
            return result.get("text", "")
        except Exception as exc:
            logger.warning(f"Whisper transcription failed: {exc}")
            return ""

    def _analyze_media_notes(self, row: Dict[str, Any]) -> str:
        if not REPORT_ENABLE_MEDIA_ANALYSIS:
            return ""

        desc = str(row.get("desc", ""))
        image_urls = self._extract_image_urls(desc)
        notes = []
        if image_urls:
            notes.append(f"检测到图片链接：{', '.join(image_urls[:3])}。请结合图像内容补充分析。")
        return "\n".join(notes)

    def _build_report_row(self, row: pd.Series, note_comments: List[str]) -> Dict[str, Any]:
        extra_notes = self._analyze_media_notes(row)
        ai_result = self._call_ai_analysis(
            row.get("title", "") or "",
            row.get("desc", "") or "",
            note_comments,
            extra_notes=extra_notes,
        )
        try:
            score_value = float(self._extract_ai_score(ai_result.get("ai_score", "0")))
        except (TypeError, ValueError):
            score_value = 0.0

        return {
            "标题": row.get("title", ""),
            "原文链接": row.get("note_url", ""),
            "原文": row.get("desc", ""),
            "摘要": ai_result.get("summary", ""),
            "作者": row.get("nickname", ""),
            "评论数": len(note_comments),
            "AI评分": round(score_value, 2),
            "评论区分析": ai_result.get("comment_analysis", ""),
            "AI标签": ai_result.get("tags", ""),
            "媒体洞察": ai_result.get("media_insights", ""),
        }

    def generate_report(self) -> None:
        contents = self.load_contents()
        comments = self.load_comments()
        if contents.empty:
            logger.warning(f"No contents found for platform '{self.platform}'. Report generation aborted.")
            return

        contents = contents[contents["note_id"] != ""].copy()

        report_records: List[Dict[str, Any]] = []
        for _, row in contents.iterrows():
            note_id = str(row.get("note_id", ""))
            note_comments = comments[comments["note_id"] == note_id]["content"].astype(str).tolist()
            row_record = self._build_report_row(row, note_comments)
            report_records.append(row_record)

        if not report_records:
            logger.warning("No report rows could be generated.")
            return

        report_df = pd.DataFrame(report_records)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = self.report_dir / f"{self.platform}_report_{now}.md"
        excel_path = self.report_dir / f"{self.platform}_report_{now}.xlsx"

        self._write_markdown(report_df, md_path)
        self._write_excel(report_df, excel_path)
        self._write_json(report_records, self.report_dir / f"{self.platform}_report_{now}.json")
        logger.info(f"Report generated: {md_path}")
        logger.info(f"Excel generated: {excel_path}")

    def _write_markdown(self, df: pd.DataFrame, path: pathlib.Path) -> None:
        lines = [
            f"# {self.platform} 爬虫分析报告",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "---",
            "",
        ]
        for _, row in df.iterrows():
            lines.append(f"## {row.get('标题', '')}")
            lines.append(f"作者：{row.get('作者', '')}")
            lines.append(f"链接：{row.get('原文链接', '')}")
            lines.append("")
            lines.append("### 摘要")
            lines.append(row.get('摘要', '') or "")
            lines.append("")
            lines.append("### 原文")
            lines.append(row.get('原文', '') or "")
            lines.append("")
            lines.append("### AI 评分")
            lines.append(str(row.get('AI评分', '')))
            lines.append("")
            lines.append("### AI 标签")
            lines.append(row.get('AI标签', '') or "")
            lines.append("")
            lines.append("### 评论区分析")
            lines.append(row.get('评论区分析', '') or "")
            lines.append("")
            lines.append("### 媒体洞察")
            lines.append(row.get('媒体洞察', '') or "")
            lines.append("")
            lines.append("---")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_excel(self, df: pd.DataFrame, path: pathlib.Path) -> None:
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="报告")
                ws = writer.sheets["报告"]
                widths = {
                    "标题": 40,
                    "原文链接": 45,
                    "摘要": 60,
                    "原文": 80,
                    "AI评分": 10,
                    "评论区分析": 80,
                    "AI标签": 35,
                    "媒体洞察": 60,
                }
                for column_name, width in widths.items():
                    if column_name in df.columns:
                        idx = list(df.columns).index(column_name)
                        excel_col = chr(ord("A") + idx)
                        ws.column_dimensions[excel_col].width = width
        except Exception as exc:
            logger.warning(f"Unable to write report Excel: {exc}")

    def _write_json(self, records: List[Dict[str, Any]], path: pathlib.Path) -> None:
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"Unable to write report JSON: {exc}")


def generate_report(platform: str) -> None:
    generator = MediaCrawlerReportGenerator(platform=platform)
    generator.generate_report()
