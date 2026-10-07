"""Visit the Streamlit app in a real browser so Streamlit counts real traffic.

Plain HTTP pings do NOT work on Streamlit Cloud (they only get a static
shell page). A real browser runs the page JavaScript, opens the live
connection, and the app stays awake. If the app is already asleep, this
clicks the "wake up" button for you.
"""

import sys

from playwright.sync_api import TimeoutError as PwTimeout
from playwright.sync_api import sync_playwright

APP_URL = "https://rbi-sentinel.streamlit.app/"
WAKE_TEXT = "Yes, get this app back up!"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=120000)
        except PwTimeout:
            print("LOAD TIMEOUT (checking page state anyway)")
        page.wait_for_timeout(8000)
        if "gone to sleep" in page.content() or page.get_by_role(
            "button", name=WAKE_TEXT
        ).count() > 0:
            print("ASLEEP -> clicking wake button")
            page.get_by_role("button", name=WAKE_TEXT).first.click()
            page.wait_for_timeout(90000)
            print("WAKE done, page title now:", page.title())
        else:
            print("OK - app awake, page title:", page.title())
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - log everything, fail the run loudly
        print("KEEPALIVE ERROR:", e)
        sys.exit(1)
