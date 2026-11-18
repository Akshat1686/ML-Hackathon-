const path = require('node:path')
const { existsSync } = require('node:fs')
const { pathToFileURL } = require('node:url')

const [modulePath, executablePath, apiPort, uiPort] = process.argv.slice(2)
const { chromium } = require(modulePath)

async function capture() {
  const browser = await chromium.launch({ executablePath, headless: true, args: ['--no-sandbox'] })
  try {
    const page = await browser.newPage({ viewport: { width: 1220, height: 880 }, ignoreHTTPSErrors: true })
    await page.goto('http://127.0.0.1:' + uiPort)
    await page.locator('#api').fill('http://127.0.0.1:' + apiPort)
    await page.getByRole('button', {name: 'Connect'}).click()
    await page.getByText('10 trips available.').waitFor()
    await page.locator('#trip').selectOption('00000052')
    await page.getByRole('button', {name: 'Reconstruct route'}).click()
    await page.getByText('Trip 00000052 reconstructed.').waitFor()
    await page.screenshot({ path: 'results/ui_capture.png', fullPage: true })
    try {
      for (const ext of ['js', 'css']) {
        const asset = path.resolve('.cache/leaflet/leaflet.' + ext)
        if (existsSync(asset)) {
          await page.route('https://cdn.jsdelivr.net/npm/leaflet@*/dist/leaflet.' + ext,
            route => route.fulfill({path: asset}))
        }
      }
      await page.goto(pathToFileURL(path.resolve('results/folium_map.html')).href, { timeout: 20000 })
      await page.locator('.leaflet-overlay-pane svg path').first().waitFor({ timeout: 10000 })
      await page.screenshot({ path: 'results/folium_capture.png' })
    } catch (error) {
      console.warn('Folium browser capture unavailable: ' + error.message)
    }
  } finally {
    await browser.close()
  }
}

capture().catch(error => { console.error(error); process.exitCode = 1 })
