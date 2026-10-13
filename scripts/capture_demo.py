"""Capture full-page UI screenshots and a narrated demo video.

Uses the installed Google Chrome through Playwright. The video is the viewport
recording; each screenshot is the whole Streamlit or docs page after the inner
scroll container is expanded.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
VIDEO_DIR = ROOT / "docs" / "demo_video"
UI = "http://127.0.0.1:8601"
DOCS = "http://127.0.0.1:8600/docs"

EXPAND = """
() => {
  const nodes = document.querySelectorAll(
    '.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"], section.main'
  );
  nodes.forEach((node) => {
    node.style.height = 'auto';
    node.style.maxHeight = 'none';
    node.style.overflow = 'visible';
  });
  document.documentElement.style.height = 'auto';
  document.body.style.height = 'auto';
  document.body.style.overflow = 'visible';
  return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
}
"""

RESTORE = """
() => {
  const nodes = document.querySelectorAll(
    '.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"], section.main'
  );
  nodes.forEach((node) => {
    node.style.height = '';
    node.style.maxHeight = '';
    node.style.overflow = '';
  });
  document.documentElement.style.height = '';
  document.body.style.height = '';
  document.body.style.overflow = '';
}
"""


def shoot(page: Page, name: str) -> None:
    page.wait_for_timeout(700)
    height = page.evaluate(EXPAND)
    page.set_viewport_size({"width": 1440, "height": max(int(height) + 40, 900)})
    page.wait_for_timeout(400)
    path = SHOTS / name
    page.screenshot(path=str(path), full_page=True)
    print(f"shot {path.name} height={height}", flush=True)
    page.evaluate(RESTORE)
    page.set_viewport_size({"width": 1440, "height": 900})


def scroll_story(page: Page) -> None:
    page.evaluate(
        """async () => {
          const el = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
          const step = 500;
          for (let y = 0; y <= el.scrollHeight; y += step) {
            el.scrollTo(0, y);
            await new Promise((resolve) => setTimeout(resolve, 450));
          }
          el.scrollTo(0, 0);
        }"""
    )


def click_tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(800)


def score(page: Page, example: str | None, title: str | None, label: str) -> None:
    click_tab(page, "Predict")
    if example:
        page.get_by_role("button", name=example).last.click()
        page.wait_for_timeout(1200)
    if title:
        page.get_by_label("Title", exact=True).fill(title)
        page.get_by_label("Synopsis", exact=True).fill(
            "A catalogue title scored from its metadata. The synopsis is not a model feature."
        )
    page.get_by_role("button", name="Score this title").last.click()
    page.wait_for_timeout(2500)
    page.get_by_text("Looking up a poster", exact=False).wait_for(state="hidden", timeout=20000)
    try:
        page.locator("img").first.wait_for(state="visible", timeout=8000)
    except Exception:
        pass
    page.wait_for_timeout(800)
    shoot(page, label)
    scroll_story(page)


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(UI, wait_until="networkidle", timeout=60000)
        page.get_by_role("tab", name="Predict").wait_for()
        shoot(page, "01_predict_tab.png")

        score(page, "Obvious movie", None, "02_predict_obvious_movie.png")
        score(page, "Obvious series", None, "03_predict_obvious_series.png")
        score(page, "Ambiguous case", None, "04_predict_ambiguous.png")
        score(page, "Obvious movie", "The Matrix", "05_predict_matrix_poster.png")
        score(page, "Obvious series", "Ganglands", "06_predict_ganglands_poster.png")

        click_tab(page, "Batch")
        scroll_story(page)
        shoot(page, "07_batch_tab.png")

        click_tab(page, "Model performance")
        scroll_story(page)
        shoot(page, "08_model_performance.png")

        click_tab(page, "The leakage story")
        scroll_story(page)
        shoot(page, "09_leakage_story.png")

        click_tab(page, "About")
        scroll_story(page)
        shoot(page, "10_about.png")

        page.goto(DOCS, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(1000)
        scroll_story(page)
        shoot(page, "11_api_docs.png")

        video = page.video
        context.close()
        browser.close()
        if video is not None:
            saved = video.path()
            target = VIDEO_DIR / "demo.webm"
            Path(saved).replace(target)
            print(f"video {target}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CAPTURE FAILED: {exc}", file=sys.stderr)
        raise
