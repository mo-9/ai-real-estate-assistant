import { test, expect } from '@playwright/test';

const SCREENSHOT_DIR = 'artifacts/test_run_2025-12-09/screenshots';

async function openDetailsBySummary(
  page: Parameters<Parameters<typeof test>[2]>[0]['page'],
  root: Parameters<Parameters<typeof test>[2]>[0]['page'] | any,
  summaryText: string | RegExp
): Promise<void> {
  const details = root
    .locator('details')
    .filter({ has: root.locator('summary').filter({ hasText: summaryText }) })
    .first();
  if (await details.count()) {
    await details.locator('summary').first().scrollIntoViewIfNeeded();
    await details.evaluate((el: any) => {
      el.open = true;
    });
    return;
  }

  const summary = root.locator('summary').filter({ hasText: summaryText }).first();
  if (await summary.count()) {
    await summary.click({ force: true });
    return;
  }

  const expanderHeader = root
    .locator('.streamlit-expanderHeader')
    .filter({ hasText: summaryText })
    .first();
  if (await expanderHeader.count()) {
    await expanderHeader.scrollIntoViewIfNeeded();
    const expanderButton = expanderHeader.locator('button').first();
    if (await expanderButton.count()) {
      await expanderButton.click({ force: true });
    } else {
      await expanderHeader.click({ force: true });
    }
    return;
  }

  const header = root.getByText(summaryText).first();
  if (await header.count()) {
    const headerContainer = header.locator('xpath=ancestor::*[self::div][1]').first();
    if (await headerContainer.count()) {
      await headerContainer.click({ force: true });
      return;
    }
    await header.click({ force: true });
  }
}

async function clickStreamlitTab(tabsRoot: any, name: RegExp): Promise<void> {
  const tab = tabsRoot.getByRole('tab', { name }).first();
  await tab.scrollIntoViewIfNeeded();
  await tab.evaluate((el: HTMLElement) => el.click());
}

async function ensureVisibleInSidebar(
  scrollContainer: any,
  target: any,
  timeoutMs: number
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await target.isVisible()) {
      return;
    }
    await scrollContainer.evaluate((el: any) => {
      el.scrollTop = (el.scrollTop || 0) + 300;
    });
    await target.page().waitForTimeout(50);
  }
}

test.describe('UI Smoke', () => {
  test('home renders and app title visible @smoke', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('[data-testid="stAppViewBlockContainer"]')).toBeVisible({ timeout: 10000 });
    await expect(page.getByRole('heading', { name: /AI Real Estate Assistant/i }).first()).toBeVisible();
    await page.screenshot({ path: `${SCREENSHOT_DIR}/home.png`, fullPage: true });
  });

  test('load data from default URLs via sidebar @smoke', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('[data-testid="stAppViewBlockContainer"]')).toBeVisible({ timeout: 10000 });
    const sidebar = page.locator('[data-testid="stSidebar"]');
    await expect(sidebar).toBeVisible({ timeout: 10000 });
    let sidebarScrollContainer: any = sidebar.locator('[data-testid="stSidebarContent"]').first();
    if (!(await sidebarScrollContainer.count())) {
      sidebarScrollContainer = sidebar;
    }

    const dataSourcesDetails = sidebar
      .locator('details')
      .filter({ has: sidebar.locator('summary').filter({ hasText: /Data Sources/i }) })
      .first();
    if (await dataSourcesDetails.count()) {
      const dataSourcesSummary = dataSourcesDetails.locator('summary').first();
      await ensureVisibleInSidebar(sidebarScrollContainer, dataSourcesSummary, 10_000);
      await dataSourcesDetails.evaluate((el: any) => {
        el.open = true;
      });
    } else {
      const dataSourcesHeader = sidebar.getByText(/Data Sources/i).first();
      await ensureVisibleInSidebar(sidebarScrollContainer, dataSourcesHeader, 10_000);
      await dataSourcesHeader.click({ force: true });
    }

    const dataSourceGroup = sidebar
      .getByRole('radiogroup', { name: /Data Source/i })
      .first()
      .or(sidebar.locator('[role="radiogroup"]').first());
    await expect(dataSourceGroup).toHaveCount(1, { timeout: 10000 });
    await ensureVisibleInSidebar(sidebarScrollContainer, dataSourceGroup, 10_000);
    await expect(dataSourceGroup).toBeVisible({ timeout: 10000 });
    const urlOption = dataSourceGroup.getByText('URL', { exact: true });
    if (await urlOption.count()) {
      await ensureVisibleInSidebar(sidebarScrollContainer, urlOption, 10_000);
      await urlOption.click({ force: true });
    } else {
      const fallback = sidebar.locator('div[data-baseweb="radio"]').filter({ hasText: 'URL' }).first();
      await ensureVisibleInSidebar(sidebarScrollContainer, fallback, 10_000);
      await fallback.click({ force: true });
    }
    const ta = sidebar.getByLabel(/Data URLs/);
    await ensureVisibleInSidebar(sidebarScrollContainer, ta, 10_000);
    await expect(ta).toBeVisible();
    await sidebar.getByRole('button', { name: /Load Data/i }).click({ force: true });
    await page.screenshot({ path: `${SCREENSHOT_DIR}/data_loaded.png`, fullPage: true });
  });

  test('notifications tab shows email inputs @smoke', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('[data-testid="stAppViewBlockContainer"]')).toBeVisible({ timeout: 10000 });
    const tabsRoot = page.locator('.stTabs').first();
    await expect(tabsRoot).toBeVisible({ timeout: 10000 });

    await clickStreamlitTab(tabsRoot, /Notifications/i);
    if (!(await page.getByLabel(/Email Username/i).first().isVisible())) {
      await openDetailsBySummary(page, page, /Configure Email/i);
    }

    await expect(page.getByLabel(/Email Username/i).first()).toBeVisible();
    await expect(page.getByLabel(/Email Password\/App Password/i).first()).toBeVisible();
    await page.screenshot({ path: `${SCREENSHOT_DIR}/notifications_inputs.png`, fullPage: true });
  });
  test('tabs navigate and render content @smoke', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('[data-testid="stAppViewBlockContainer"]')).toBeVisible({ timeout: 10000 });
    const tabsRoot = page.locator('.stTabs').first();
    await expect(tabsRoot).toBeVisible({ timeout: 10000 });
    const tabsToCheck = [
      {
        name: /Chat/i,
        visibleAssertion: page.getByRole('heading', { name: /What You Can Do/i }).first(),
        screenshot: `${SCREENSHOT_DIR}/tab_chat.png`,
      },
      {
        name: /Market Insights/i,
        visibleAssertion: page.getByText(/to view market insights/i).first(),
        screenshot: `${SCREENSHOT_DIR}/tab_insights.png`,
      },
      {
        name: /Compare/i,
        visibleAssertion: page.getByText(/to compare properties/i).first(),
        screenshot: `${SCREENSHOT_DIR}/tab_compare.png`,
      },
      {
        name: /Export/i,
        visibleAssertion: page.getByText(/from the sidebar to export/i).first(),
        screenshot: `${SCREENSHOT_DIR}/tab_export.png`,
      },
      {
        name: /Analytics/i,
        visibleAssertion: page.getByRole('heading', { name: /Session Analytics/i }).first(),
        screenshot: `${SCREENSHOT_DIR}/tab_analytics.png`,
      },
      {
        name: /Notifications/i,
        visibleAssertion: page.getByRole('heading', { name: /Notification Settings/i }).first(),
        screenshot: `${SCREENSHOT_DIR}/tab_notifications.png`,
      },
    ];

    for (const tabToCheck of tabsToCheck) {
      await clickStreamlitTab(tabsRoot, tabToCheck.name);
      await expect(tabToCheck.visibleAssertion).toBeVisible({ timeout: 10000 });
      await page.screenshot({ path: tabToCheck.screenshot, fullPage: true });
    }
  });

  test('model configuration expander and provider select @smoke', async ({ page }) => {
    await page.goto('/');
    if (!(await page.getByText(/Provider/i).first().isVisible())) {
      await openDetailsBySummary(page, page, /Model Configuration/i);
    }
    await expect(page.getByText(/Provider/i).first()).toBeVisible();
    await page.screenshot({ path: `${SCREENSHOT_DIR}/sidebar_model_config.png`, fullPage: true });
  });
});
