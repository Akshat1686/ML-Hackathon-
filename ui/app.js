const byId = id => document.getElementById(id)
const canvas = byId('map')
const ctx = canvas.getContext('2d')
const fields = [
  ['observation_accuracy', 'Accuracy', '%'],
  ['route_continuity', 'Route continuity', '%'],
  ['ordered_arc_pair_f1', 'Ordered arc-pair F1', '%'],
  ['alignment_coverage', 'Reference alignment coverage', '%'],
  ['gps_observations', 'GPS observations', ''],
  ['route_parts', 'Route parts', ''],
  ['inference_ms', 'Inference time', 'ms'],
]
let base = ''
let coordinateLayers = []
let zoom = 1, shiftX = 0, shiftY = 0, drag

function showMetrics(metrics = {}) {
  byId('metrics').replaceChildren(...fields.flatMap(([key, label, unit]) => {
    const name = document.createElement('dt')
    const value = document.createElement('dd')
    name.textContent = label
    const number = metrics[key]
    value.textContent = number == null ? '—' : unit === '%' ? (number * 100).toFixed(1) + '%' :
      number.toLocaleString() + (unit ? ' ' + unit : '')
    return [name, value]
  }))
}

function draw() {
  const { width, height } = canvas.getBoundingClientRect()
  const ratio = window.devicePixelRatio || 1
  canvas.width = Math.round(width * ratio)
  canvas.height = Math.round(height * ratio)
  ctx.scale(ratio, ratio)
  ctx.fillStyle = '#f4f8fa'
  ctx.fillRect(0, 0, width, height)
  ctx.strokeStyle = '#e4edf2'
  ctx.beginPath()
  for (let x = 0; x < width; x += 28) { ctx.moveTo(x, 0); ctx.lineTo(x, height) }
  for (let y = 0; y < height; y += 28) { ctx.moveTo(0, y); ctx.lineTo(width, y) }
  ctx.stroke()
  if (!coordinateLayers.length) return
  const bounds = [Infinity, Infinity, -Infinity, -Infinity]
  for (const layer of coordinateLayers) {
    for (const part of layer.parts) {
      for (const [x, y] of part) {
        bounds[0] = Math.min(bounds[0], x); bounds[1] = Math.min(bounds[1], y)
        bounds[2] = Math.max(bounds[2], x); bounds[3] = Math.max(bounds[3], y)
      }
    }
  }
  const [left, bottom, right, top] = bounds
  const cosLat = Math.cos((top + bottom) * Math.PI / 360)
  const scale = zoom * Math.min((width - 60) / Math.max(1e-6, (right - left) * cosLat),
    (height - 60) / Math.max(1e-6, top - bottom))
  for (const layer of coordinateLayers) {
    ctx.strokeStyle = layer.color
    ctx.fillStyle = layer.color
    ctx.lineWidth = layer.width
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    for (const part of layer.parts) {
      ctx.beginPath()
      part.forEach(([lon, lat], i) => {
        const x = width / 2 + shiftX + (lon - (left + right) / 2) * cosLat * scale
        const y = height / 2 + shiftY - (lat - (top + bottom) / 2) * scale
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
        if (part.length === 1) ctx.arc(x, y, 2, 0, Math.PI * 2)
      })
      ctx.stroke()
    }
  }
}

function reset() {
  zoom = 1
  shiftX = shiftY = 0
  draw()
}

async function read(url, options) {
  const response = await fetch(url, options)
  if (!response.ok) throw new Error('API returned HTTP ' + response.status)
  return response.json()
}

byId('connect').addEventListener('submit', async event => {
  event.preventDefault()
  const button = event.currentTarget.querySelector('button')
  button.disabled = byId('run').disabled = byId('trip').disabled = true
  byId('status').textContent = 'Connecting…'
  try {
    const address = byId('api').value.replace(/\/+$/, '')
    const catalog = await read(address + '/trips')
    base = address
    byId('trip').replaceChildren(...catalog.trip_ids.map(id => new Option(id, id)))
    byId('mode').textContent = catalog.mode === 'evaluation' ? 'Evaluator trips' : 'Student-test trips'
    byId('status').textContent = catalog.trip_ids.length + ' trips available.'
    byId('run').disabled = byId('trip').disabled = false
  } catch (error) {
    byId('status').textContent = 'Connection failed. ' + error.message
  } finally {
    button.disabled = false
  }
})

byId('predict').addEventListener('submit', async event => {
  event.preventDefault()
  const selected = byId('trip').value
  const connect = byId('connect').querySelector('button')
  connect.disabled = byId('run').disabled = byId('trip').disabled = true
  byId('status').textContent = 'Reconstructing trip ' + selected + '…'
  try {
    const result = await read(base + '/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({trip_id: selected}),
    })
    coordinateLayers = [
      {parts: result.geometry.gps, color: '#d37225', width: 2},
      {parts: result.geometry.predicted_route, color: '#087e77', width: 3},
    ]
    showMetrics(result.metrics)
    reset()
    byId('reset').disabled = false
    byId('status').textContent = 'Trip ' + result.trip_id + ' reconstructed.'
  } catch (error) {
    byId('status').textContent = 'Prediction failed. ' + error.message
  } finally {
    connect.disabled = byId('run').disabled = byId('trip').disabled = false
  }
})

canvas.addEventListener('wheel', event => {
  if (!coordinateLayers.length) return
  event.preventDefault()
  zoom = Math.max(0.5, Math.min(15, zoom * Math.exp(-event.deltaY * .001)))
  draw()
}, {passive: false})
canvas.addEventListener('pointerdown', event => {
  if (!coordinateLayers.length) return
  canvas.setPointerCapture(event.pointerId)
  drag = [event.clientX - shiftX, event.clientY - shiftY]
})
canvas.addEventListener('pointermove', event => {
  if (!drag) return
  shiftX = event.clientX - drag[0]
  shiftY = event.clientY - drag[1]
  draw()
})
canvas.addEventListener('lostpointercapture', () => { drag = null })
byId('reset').addEventListener('click', reset)
new ResizeObserver(draw).observe(canvas)
showMetrics()
